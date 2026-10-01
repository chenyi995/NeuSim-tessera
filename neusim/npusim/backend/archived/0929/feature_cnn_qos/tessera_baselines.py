"""Paper baseline array kernels embedded in the common native NeuSim runtime.

References (read-only): FissionSA/fissionsa/modes/combo{0,1,4,10}.py,
simulate_layer_{planaria_grid,ws_connected_grid,ws_disconnected_grid,os_sisa}.
Only array cycles and useful on-chip words are ported, not the DRAM cadence.
Planaria's legal compositions are from experiments/planaria_native/
run_native_workloads.py::COMPOSITIONS. Native NeuSim EDP selects among them.
"""
from functools import lru_cache

from neusim.npusim.backend.tessera import ceildiv

ARCHITECTURES = ("WS", "Planaria-32", "SOSA", "FlexSA", "SISA")


def geometries(config):
    """Published equal-PE instances used by the paper's main comparison."""
    arch = config.tessera_parameters["baseline_architecture"]
    if config.sa_dim != 128 or config.num_sa != 1:
        raise ValueError("paper baseline instances require 128x128 total PEs")
    # Codex: decision start — use the paper baseline geometries, not Tessera cuts.
    if arch == "Planaria-32":
        for a in (1, 2, 4, 8, 16):
            for b in (1, 2, 4, 8, 16):
                if a * b <= 16:
                    yield (32*a, 32*b, True, 16//(a*b), 32*a, 32*b)
    elif arch in ARCHITECTURES:
        dim = {"WS": 128, "SOSA": 32, "FlexSA": 64, "SISA": 128}[arch]
        yield (dim, dim, True, (128//dim)**2, dim, dim)
    else:
        raise ValueError(f"unknown paper baseline: {arch}")
    # Codex: decision end


@lru_cache(maxsize=131072)
def array_cost(arch, m, n, k, geometry):
    """Return cycles, A words, B words, partial words, K groups and folds."""
    h, w = geometry[-2:]
    slots = 128**2 // (h*w)
    if arch in ("WS", "SOSA", "Planaria-32"):
        nk, nn = ceildiv(k, h), ceildiv(n, w)
        folds = ceildiv(nk*nn, slots)
        if arch == "Planaria-32":
            # combo0's executable formula uses max(M,d), not the stale module
            # docstring's d+W-1. Keep the reference implementation exactly.
            stream = max(m, 32) if folds >= 3 else m
            cycles = folds*stream + 32 + h + w - 2
        else:
            stream = max(m, 2*h-1) if folds >= 3 else m
            cycles = folds*stream + 3*h - 2
        return cycles, m*k*nn, k*n, m*n*nk, nk, folds
    if arch == "SISA":
        # combo10: eight OS slabs; K chunks accumulate in PE registers, without
        # a new fill/drain at SRAM K boundaries. Each output fold has full drain.
        full, rem = divmod(m, 128)
        nt = ceildiv(n, 128)
        cycles = full * nt * (k + 128 + 128 - 2)
        folds = full * nt
        if rem:
            height = 16
            while height < rem:
                height *= 2
            groups = 128 // height
            extra = ceildiv(nt, groups)
            cycles += extra * (k + height + 128 - 2)
            folds += extra
        return cycles, m*k*nt, k*n*ceildiv(m, 128), m*n, 1, folds
    if arch != "FlexSA":
        raise ValueError(arch)
    # combo1: full blocks, vertical strips, horizontal strips, then corners.
    kb, kr = divmod(k, 128)
    nb, nr = divmod(n, 128)
    ks, ns = ceildiv(kr, 64), ceildiv(nr, 64)
    a, b, c, d = kb*nb, kb*ns, ks*nb, ks*ns
    cycles = a*(max(m, 191) if a >= 3 else m) + 128 if a else 0
    if b:
        folds_b = ceildiv(b, 2)
        cycles += folds_b*(max(m, 127) if folds_b >= 3 else m) + 64
    if c or d:
        if c:
            per = [(c+1-i)//2 for i in range(2)]
            stream = max(m, 191) if max(per) >= 3 else m
            finish = [count*stream + 128 for count in per]
        else:
            finish = [0, 0]
        ends = [finish[i//2] for i in range(4)]
        provisional, counts = list(ends), [0]*4
        for _ in range(d):
            i = min(range(4), key=lambda j: provisional[j])
            provisional[i] += m + (64 if counts[i] == 0 else 0)
            counts[i] += 1
        stream = max(m, 127) if max(counts) >= 3 else m
        counts = [0]*4
        for _ in range(d):
            i = min(range(4), key=lambda j: ends[j])
            ends[i] += stream + (64 if counts[i] == 0 else 0)
            counts[i] += 1
        cycles += max(ends) + 64 + 64 - 2
    elif a or b:
        cycles += 64 + 128 - 2
    return cycles, m*k*(nb+ns), k*n, m*n*(kb+ks), kb+ks, a+ceildiv(b, 2)+ceildiv(c+d, 4)


def padded_activity(arch, m, n, k, geometry):
    """MACs and operand reads of assigned physical tiles; unused tiles stay idle.

    Schedule sources: combo1.py WS phase shapes and combo10.py SISA slab shapes.
    Zero lanes use the same energy coefficients, per chenyi9's padding ruling.
    """
    if arch == "SISA":
        full, rem = divmod(m, 128)
        height = max(16, 1 << (rem-1).bit_length()) if rem else 0
        rows = full*128 + height
        nt = ceildiv(n, 128)
        return rows*k*nt*128, rows*k*nt, k*nt*128*ceildiv(m, 128)
    if arch == "FlexSA":
        kb, kr = divmod(k, 128)
        nb, nr = divmod(n, 128)
        kp, np = kb*128+ceildiv(kr,64)*64, nb*128+ceildiv(nr,64)*64
        return m*kp*np, m*kp*(nb+ceildiv(nr,64)), kp*np
    h, w = geometry[-2:]
    nk, nn = ceildiv(k, h), ceildiv(n, w)
    return m*nk*nn*h*w, m*nk*nn*h, nk*nn*h*w


@lru_cache(maxsize=262144)
def counts(b, m, n, k, a_bytes, b_bytes, out_bytes, geometry, arch,
           capacity, freq, bw, resident=False, forced_tile=None, sa_energy_accounting="useful", sram_read_accounting="useful"):
    """Use native SRAM tiling/HBM traffic with each baseline's array schedule."""
    from neusim.npusim.backend.tessera_partitioned import memory_tile, parts, BACKEND
    planes = geometry[3]
    # chenyi9: decision start — use the common joint EDP SRAM mapping.
    if forced_tile is None:
        mt, nt, kt, peak = memory_tile(m, n, k, a_bytes, b_bytes, 4, planes, capacity, 128, freq, bw)
    else:
        from neusim.npusim.backend.tessera_partitioned import checked_tile
        mt, nt, kt = forced_tile
        peak = checked_tile(m, n, k, a_bytes, b_bytes, 4, planes, capacity, forced_tile)
    # chenyi9: decision end
    cycles = aw = bw_words = pw = reductions = rounds = padded_macs = 0
    padded_a = padded_b = 0
    if sa_energy_accounting not in ("useful", "padded_tiles"):
        raise ValueError("unknown SA energy accounting")
    if sram_read_accounting not in ("useful", "padded_tiles"):
        raise ValueError("unknown SRAM read accounting")
    for mm, mc in parts(m, mt):
        for nn, nc in parts(n, nt):
            groups = 0
            # Reference SISA retains running sums in PEs across K transfers.
            kparts = [(k, 1)] if arch == "SISA" else parts(k, kt)
            for kk, kc in kparts:
                timing, ra, rb, partial, kg, folds = array_cost(arch, mm, nn, kk, geometry)
                repeat = b*mc*nc*kc
                # chenyi9: decision start — charge compute and reads for padded lanes.
                cm, pa, pb = padded_activity(arch, mm, nn, kk, geometry)
                padded_macs += repeat*cm
                padded_a += repeat*pa
                padded_b += repeat*pb
                # chenyi9: decision end
                cycles += repeat*timing
                aw += repeat*ra
                bw_words += repeat*rb
                pw += repeat*partial
                rounds += repeat*folds
                groups += kc*kg
            reductions += b*mc*nc*mm*nn*(groups-1)
    hbm_a = b*m*k*ceildiv(n, nt)*a_bytes
    hbm_b = b*k*n*ceildiv(m, mt)*b_bytes
    hbm_c = b*m*n*out_bytes
    if resident:
        hbm_a = hbm_b = hbm_c = 0
    final = b*m*n*4
    padding_reads = 0
    if sram_read_accounting == "padded_tiles":
        padding_reads = (padded_a-aw)*a_bytes + (padded_b-bw_words)*b_bytes
        assert padding_reads >= 0
        aw, bw_words = padded_a, padded_b
    sram = aw*a_bytes + bw_words*b_bytes + pw*4 + reductions*12 + final + hbm_a + hbm_b
    macs = b*m*n*k
    assert cycles*128**2 >= macs
    charged_macs = padded_macs if sa_energy_accounting == "padded_tiles" else macs
    assert macs <= charged_macs <= cycles*128**2
    return dict(model=BACKEND, engine="SA", baseline_architecture=arch,
                geometry=list(geometry), variant=arch, B=b, M=m, N=n, K=k,
                useful_macs=macs, sa_cycles=cycles, sa_arithmetic_ops=2*charged_macs,
                charged_sa_macs=charged_macs,padding_macs=charged_macs-macs,
                sa_energy_accounting=sa_energy_accounting,
                padding_sram_read_bytes=padding_reads, sram_read_accounting=sram_read_accounting,
                reduction_ops=reductions, vu_arithmetic_ops=reductions,
                array_a_read_bytes=aw*a_bytes, array_b_read_bytes=bw_words*b_bytes,
                partial_write_bytes=pw*4, reduction_read_bytes=reductions*8,
                reduction_write_bytes=reductions*4, final_read_bytes=final,
                sram_bytes=sram, hbm_a_read_bytes=hbm_a, hbm_b_read_bytes=hbm_b,
                hbm_output_write_bytes=hbm_c, hbm_partial_read_bytes=0,
                hbm_partial_write_bytes=0, hbm_bytes=hbm_a+hbm_b+hbm_c,
                peak_live_bytes=peak, memory_tile=[mt, nt, kt], rounds=rounds,
                skew_tail_cycles=0)
