"""Joint SRAM/array search under the common native NeuSim energy model.

Search space: native divisor tiles plus dyadic extents with explicit remainders, legal architecture
geometries, existing output-stationary SRAM reuse and resident FP32 reduction.
The vectorized counters are checked against scalar forced-tile execution.
This is per-GEMM EDP minimization, not global graph or loop-order optimization.
"""
from functools import lru_cache
import json
import numpy as np

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.util import get_factors, get_size_bytes_from_dtype


def divup(a, b):
    return (a+b-1)//b


@lru_cache(maxsize=256)
def factor_tiles(m, n, k, word, planes, capacity):
    """Native divisor candidates plus dyadic extents; tails use scalar-kernel accounting."""
    # chenyi9: decision start -- prime dimensions must not restrict tiles to 1/full.
    # Keep the original native candidates and add power-of-two tile extents,
    # matching the array's dyadic geometry family. This is a finite candidate
    # search, not an assertion of global optimality over every integer extent.
    def extents(size):
        return sorted(set(get_factors(size)) | {1 << i for i in range(size.bit_length())})
    mt, nt, kt = np.meshgrid(extents(m), extents(n), extents(k), indexing="ij")
    # chenyi9: decision end
    mt, nt, kt = mt.ravel(), nt.ravel(), kt.ravel()
    # Source: tessera_partitioned.checked_tile: two operand buffers, FP32 C.
    peak = 2*word*(mt*kt+kt*nt) + 4*(planes+1)*mt*nt
    valid = peak <= capacity
    return mt[valid], nt[valid], kt[valid], peak[valid]


def array_vectors(arch, mt, nt, kt, g, variant, grain, side, active_pe_budget=None):
    """Divisor-tile counterpart of scalar array kernels; integer arithmetic."""
    h, w = g[-2:]
    active_pes = side**2 if active_pe_budget is None else active_pe_budget
    if arch in ("WS", "SOSA", "WS-independent", "Planaria-32", "Planaria"):
        nk, nn = divup(kt,h), divup(nt,w)
        rounds = divup(nk*nn, active_pes//(h*w))
        # chenyi9: decision start -- scalar and vector ranking share the physical grain.
        minimum = grain if arch in ("Planaria-32", "Planaria") else 2*h-1
        cycles = rounds*np.where(rounds>=3,np.maximum(mt,minimum),mt)
        cycles += grain+h+w-2 if arch in ("Planaria-32", "Planaria") else 3*h-2
        # chenyi9: decision end
        return cycles, mt*kt*nn, kt*nt, mt*nt*nk, nk, rounds
    if arch == "SISA":
        full, rem = mt//128, mt%128
        nn = divup(nt,128)
        height = np.full_like(mt,16)
        for value in (32,64,128):
            height = np.where(rem>value//2,value,height)
        extra = np.where(rem>0,divup(nn,128//height),0)
        rounds = full*nn+extra
        cycles = full*nn*(kt+254)+extra*(kt+height+126)
        return cycles, mt*kt*nn, kt*nt*divup(mt,128), mt*nt, np.ones_like(mt), rounds
    if arch == "FlexSA":
        kb, kr, nb, nr = kt//128, kt%128, nt//128, nt%128
        ks, ns = divup(kr,64), divup(nr,64)
        a, b, c, d = kb*nb, kb*ns, ks*nb, ks*ns
        cycles = np.where(a>0,a*np.where(a>=3,np.maximum(mt,191),mt)+128,0)
        rb = divup(b,2)
        cycles += np.where(b>0,rb*np.where(rb>=3,np.maximum(mt,127),mt)+64,0)
        per = np.stack(((c+1)//2,c//2))
        stream = np.where(per.max(axis=0)>=3,np.maximum(mt,191),mt)
        finish = np.where(c>0,per*stream+128,0)
        ends = np.repeat(finish,2,axis=0)
        provisional, counts = ends.copy(), np.zeros_like(ends)
        columns = np.arange(len(mt))
        # combo1 corner has at most four 64x64 regions.
        for step in range(4):
            idx = provisional.argmin(axis=0)
            mask = d>step
            provisional[idx,columns] += mask*(mt+np.where(counts[idx,columns]==0,64,0))
            counts[idx,columns] += mask
        stream = np.where(counts.max(axis=0)>=3,np.maximum(mt,127),mt)
        counts[:] = 0
        for step in range(4):
            idx = ends.argmin(axis=0)
            mask = d>step
            ends[idx,columns] += mask*(stream+np.where(counts[idx,columns]==0,64,0))
            counts[idx,columns] += mask
        cycles += np.where((c>0)|(d>0),ends.max(axis=0)+126,np.where((a>0)|(b>0),190,0))
        nk = kb+ks
        return cycles,mt*kt*(nb+ns),kt*nt,mt*nt*nk,nk,a+divup(b,2)+divup(c+d,4)
    rk, rn, _, kp, h, w = g
    nk, nn = divup(kt,rk), divup(nt,rn)
    slots = active_pes//(h*w)
    # chenyi9: decision start — Algorithm 2 fills the unused end of a K block.
    # Source: hpca2027-paper1832.pdf, p. 8; kp FP32 producer planes remain
    # capacity-checked, so K parallelism never becomes unbounded implicitly.
    rounds = divup(nk*nn,np.minimum(slots,kp*nn))
    # chenyi9: decision end
    traditional = variant in ("independent","ws")
    release, drain = (h-1,2*h-2) if traditional else (0,h-1)
    last, interval = rounds-1,np.maximum(mt,h)
    cycles = h+last*interval+(last//2)*np.maximum(0,mt+release+h-2*interval)+mt+drain
    seams = 0 if variant in ("independent","independent_noskew","ws") else w//grain-1
    cycles += w-h+h+seams+(h-1 if variant=="skew" else 0)
    return cycles,mt*kt*nn,kt*nt,mt*nt*nk,nk,rounds


def candidate_vectors(b, m, n, k, word, geometry, config, tiles, resident=False):
    mt, nt, kt, peak = tiles
    arch = config.tessera_parameters.get("baseline_architecture")
    grain = int(config.tessera_parameters["grain"])
    # chenyi9: decision start -- include every full tile and residual tile.
    # Independent reference: tessera_partitioned.counts / tessera_baselines.counts.
    def pieces(size, tile):
        yield tile, size//tile
        rem=size%tile
        if np.any(rem):
            yield np.maximum(rem,1), (rem>0).astype(np.int64)
    cycles=aw=bw=pw=rounds=reductions=padded_macs=padded_a=padded_b=np.zeros_like(mt)
    for mm,mc in pieces(m,mt):
        for nn,nc in pieces(n,nt):
            groups=np.zeros_like(mt)
            kparts=[(np.full_like(kt,k),np.ones_like(kt))] if arch=="SISA" else pieces(k,kt)
            for kk,kc in kparts:
                repeats=b*mc*nc*kc
                tc,ta,tb,tp,kg,tr=array_vectors(arch,mm,nn,kk,geometry,config.tessera_variant,
                    int(config.tessera_parameters.get("baseline_grain",32)) if arch in ("Planaria-32","Planaria") else grain,config.sa_dim,
                    config.tessera_parameters.get("active_pe_budget"))
                cycles=cycles+repeats*tc;aw=aw+repeats*ta;bw=bw+repeats*tb
                pw=pw+repeats*tp;rounds=rounds+repeats*tr;groups=groups+kc*kg
                if arch=="SISA":
                    full,rem=mm//128,mm%128
                    height=np.full_like(mm,16)
                    for value in (32,64,128):height=np.where(rem>value//2,value,height)
                    rows=full*128+np.where(rem>0,height,0)
                    regions=divup(nn,128)
                    pa,pb,pm=rows*kk*regions,kk*regions*128*divup(mm,128),rows*kk*regions*128
                elif arch=="FlexSA":
                    kb,nb=kk//128,nn//128
                    ks,ns=divup(kk%128,64),divup(nn%128,64)
                    pk,pn=kb*128+ks*64,nb*128+ns*64
                    pa,pb,pm=mm*pk*(nb+ns),pk*pn,mm*pk*pn
                else:
                    if arch:
                        kh,nw=geometry[-2:];rk,rn=kh,nw
                    else:
                        from neusim.npusim.backend.tessera_partitioned import logical_region_shape
                        kh,nw=logical_region_shape(n,k,geometry,grain);rk,rn=geometry[:2]
                    kr,nr=divup(kk,rk),divup(nn,rn)
                    pa,pb,pm=mm*kr*nr*kh,kr*nr*kh*nw,mm*kr*nr*kh*nw
                padded_macs=padded_macs+repeats*pm
                padded_a=padded_a+repeats*pa;padded_b=padded_b+repeats*pb
            reductions=reductions+b*mc*nc*mm*nn*(groups-1)
    hbm_a, hbm_b = b*m*k*divup(n,nt)*word, b*k*n*divup(m,mt)*word
    hbm = hbm_a+hbm_b+b*m*n*word
    # chenyi9: decision end
    # chenyi9: decision start — padded input reads participate in EDP selection.
    policy = config.tessera_parameters.get("sa_energy_accounting", "useful")
    read_policy = config.tessera_parameters.get("sram_read_accounting", "useful")
    if policy not in ("useful", "padded_tiles") or read_policy not in ("useful", "padded_tiles"):
        raise ValueError("unknown padded energy accounting")
    padding_reads = np.zeros_like(mt)
    if policy == "padded_tiles" or read_policy == "padded_tiles":
        assert np.all(padded_macs>=b*m*n*k)
        assert np.all(padded_macs<=cycles*config.sa_dim**2)
        if read_policy == "padded_tiles":
            padding_reads = ((padded_a-aw)+(padded_b-bw))*word
            assert np.all(padding_reads>=0)
            aw,bw = padded_a,padded_b
    # chenyi9: decision end
    sram = aw*word+bw*word+pw*4+reductions*12+b*m*n*4
    if not resident:
        sram += hbm_a+hbm_b
    else:
        hbm = np.zeros_like(hbm)
    sa = np.ceil(cycles/config.freq_GHz).astype(np.int64)
    # User decision start — joint EDP ranks padded activity, matching native execution.
    charged_macs = np.full_like(sa,b*m*n*k)
    if policy == "padded_tiles":
        charged_macs = padded_macs
    elif policy != "useful":
        raise ValueError("unknown SA energy accounting")
    # User decision end
    # Source: issue_ns -> native compute_node_cost_vpu_time_from_num_ops.
    vu = np.where(reductions>0,np.maximum(1,(divup(divup(reductions,1024),config.num_vu)/config.freq_GHz).astype(np.int64)),0)
    return dict(sa_ns=sa,vu_ns=vu,reduction_ops=reductions,sram_bytes=sram,hbm_bytes=hbm,
                peak_live_bytes=peak,rounds=rounds,charged_sa_macs=charged_macs,
                padding_sram_read_bytes=padding_reads, array_a_read_bytes=aw*word, array_b_read_bytes=bw*word)


def native_objective(v, macs, config, bandwidth):
    """NoPG, no DVFS algebra of native power_model and op_analysis_lib.

    HBM power scales with bandwidth exactly as ChipConfig does; its 500 ns
    activity floor remains. SRAM dynamic energy charges actual transferred bytes.
    """
    hbm = np.where(v["hbm_bytes"]>0,np.maximum(np.ceil(v["hbm_bytes"]/(bandwidth/1e9)),config.hbm_latency_ns),0)
    time = np.maximum(np.maximum(v["sa_ns"],v["vu_ns"]),hbm)
    sa_energy = config.dynamic_power_sa_W*2*v.get("charged_sa_macs",macs)/(2*config.num_sa*config.sa_dim**2*config.freq_GHz*1e9)
    vu_energy = config.dynamic_power_vu_W*np.minimum(v["vu_ns"]/1e9,v["reduction_ops"]/(1024*config.num_vu*config.freq_GHz*1e9))
    sram_energy = config.dynamic_power_vmem_W*v["sram_bytes"]/config.vmem_bw_GBps/1e9
    hbm_energy = config.dynamic_power_hbm_W_per_GBps*(bandwidth/1024**3)*hbm/1e9
    # Native fixed-voltage regulator losses depend on activity, even with
    # DVFS disabled. Read the same table; never assume 100% efficiency.
    from neusim.npusim.backend.dvfs_power_getter import FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
    table = FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
    activity = np.array([r.activity_factor for r in table])
    efficiency = np.array([r.power_efficiency_percent for r in table])
    def loss(util):
        return 100/efficiency[np.searchsorted(activity,np.minimum(1,util),side="left")]
    flops_util = 2*macs/time/1e3/config.peak_tflops_per_sec
    array_loss = loss(flops_util)
    service = np.minimum(np.ceil(v["sram_bytes"]/(config.num_sa*config.sa_dim**2*8*config.freq_GHz)),time)
    seconds = time/1e9
    energy = ((config.static_power_sa_W*seconds+sa_energy)*array_loss
              +(config.static_power_vu_W*seconds+vu_energy)*array_loss
              +(config.static_power_vmem_W*seconds+sram_energy)*loss(service/time)
              +(config.static_power_hbm_W*seconds+hbm_energy)*loss(hbm/time)
              +config.static_power_ici_W*seconds*loss(0)
              +(config.static_power_other_W+config.dynamic_power_other_W)*seconds)
    return energy*time,time,energy


def array_objective(v, macs, config):
    """Isolated SA energy times SA latency for the requested granularity study.

    Source: native power_model.compute_sa_flops_util and fixed-voltage regulator
    table. Static SA energy is integrated over SA time; padded MACs have the same
    dynamic coefficient. SRAM/HBM/VU/chip-background terms are outside this
    array-only metric. Capacity feasibility remains enforced by factor_tiles.
    """
    # chenyi9: decision start — Fig. 9 selects by array-only EDP.
    from neusim.npusim.backend.dvfs_power_getter import FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
    time = v["sa_ns"]
    peak_macs_per_ns = config.num_sa * config.sa_dim**2 * config.freq_GHz
    work = v.get("charged_sa_macs", macs)
    dynamic = config.dynamic_power_sa_W * work / peak_macs_per_ns / 1e9
    table = FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
    activity = np.array([r.activity_factor for r in table])
    efficiency = np.array([r.power_efficiency_percent for r in table])
    util = np.minimum(1, macs / (time * peak_macs_per_ns))
    loss = 100 / efficiency[np.searchsorted(activity, util, side="left")]
    energy = (config.static_power_sa_W * time / 1e9 + dynamic) * loss
    return energy*time,time,energy
    # chenyi9: decision end


@lru_cache(maxsize=32768)
def search_sweep(b, m, n, k, dtype, config_json, bandwidths, resident=False):
    """Enumerate each legal candidate once, then rank at each requested bandwidth."""
    from neusim.npusim.backend.tessera_partitioned import geometries
    from neusim.npusim.backend.tessera_baselines import geometries as baseline_geometries
    config = ChipConfig.model_validate_json(config_json)
    objective = config.tessera_parameters.get("mapping_objective", "e2e_edp")
    if objective not in ("e2e_edp", "array_edp", "e2e_latency"):
        raise ValueError("unknown joint mapping objective")
    if config.enable_dvfs or config.pg_config!="NoPG" or config.use_vu_for_small_matmul:
        raise ValueError("joint search currently requires primary SA/NoPG/no-DVFS accounting")
    if config.tessera_parameters.get("sram_bandwidth_model")!="per_pe_double_buffer":
        raise ValueError("joint search requires the requested ideal SRAM bandwidth policy")
    if config.tessera_variant=="skew":
        full = config.model_copy(update={"tessera_variant":"full"})
        return search_sweep(b,m,n,k,dtype,full.model_dump_json(),bandwidths,resident)
    # Resident QK/PV have zero HBM activity. Ranking is exactly independent of
    # bandwidth; compute it once while retaining all sweep selections.
    if resident and len(bandwidths)>1:
        one = search_sweep(b,m,n,k,dtype,config_json,bandwidths[:1],True)[0]
        return (one,)*len(bandwidths)
    word = get_size_bytes_from_dtype(dtype)
    choices = baseline_geometries(config) if config.tessera_parameters.get("baseline_architecture") else geometries(m,n,k,config)
    best = [None]*len(bandwidths)
    tiles_by_planes = {}
    candidate_count = 0
    for geometry in sorted(choices):
        planes = geometry[3]
        if planes not in tiles_by_planes:
            tiles_by_planes[planes] = factor_tiles(m,n,k,word,planes,config.vmem_size_MB*1024**2)
        tiles = tiles_by_planes[planes]
        if len(tiles[0])==0:
            continue
        v = candidate_vectors(b,m,n,k,word,geometry,config,tiles,resident)
        candidate_count += len(tiles[0])
        for bi,bw in enumerate(bandwidths):
            if objective == "array_edp":
                edp,time,energy = array_objective(v,b*m*n*k,config)
            else:
                edp,time,energy = native_objective(v,b*m*n*k,config,bw)
            # Codex: Planaria's profiler minimizes cycles, using energy for ties
            # (planaria.code/src/optimizer/optimizer.py:429-440). Both QoS sides
            # use that same choice rule; other evaluations retain their objective.
            primary, secondary = (time,energy) if objective == "e2e_latency" else (edp,time)
            # /root: decision start — use the scheduler's shared VU budget during QoS mapping too.
            # Source: active PE share c/C, also used by the QoS HBM/SRAM budgets
            # and MatchedTask.get_shared_resource_estimated_time. This only ranks
            # mappings; native VU throughput, traffic and energy are unchanged.
            vu_fraction = config.tessera_parameters.get("selection_vu_fraction", 1.)
            if not 0 < vu_fraction <= 1:
                raise ValueError("selection VU fraction must be in (0, 1]")
            if vu_fraction != 1:
                if objective != "e2e_latency":
                    raise ValueError("shared VU selection is a QoS latency objective")
                primary = np.maximum(time, v["vu_ns"] / vu_fraction)
            # /root: decision end
            # Factor tiles are lexicographic. Resolve exact EDP ties by time,
            # then geometry/tile, consistently across worker scheduling.
            tied = np.flatnonzero(primary==primary.min())
            index = int(tied[np.argmin(secondary[tied])])
            tile = tuple(int(axis[index]) for axis in tiles[:3])
            key = float(primary[index]),float(secondary[index]),geometry,tile
            if best[bi] is None or key<best[bi][0]:
                best[bi] = key,(geometry,tile)
    if any(x is None for x in best):
        raise ValueError("no capacity-feasible joint mapping")
    return tuple(x[1] for x in best)


def select_mapping(b, m, n, k, dtype, config_json, resident=False):
    # chenyi9: decision start — re-optimize every bandwidth, with HBM energy.
    config = ChipConfig.model_validate_json(config_json)
    bandwidth = config.hbm_bw_GBps*1024**3
    grid = tuple(config.tessera_parameters.get("selection_bandwidths",(bandwidth,)))
    if bandwidth not in grid:
        raise ValueError("current bandwidth missing from joint selection grid")
    canonical = config.model_copy(update={"hbm_bw_GBps":grid[0]/1024**3})
    result = search_sweep(b,m,n,k,dtype,canonical.model_dump_json(),grid,resident)
    return result[grid.index(bandwidth)]
    # chenyi9: decision end
