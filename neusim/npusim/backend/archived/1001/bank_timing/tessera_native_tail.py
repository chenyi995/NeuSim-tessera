"""Native SRAM tiling, full-size bulk arrays, and fracturable residual packing.

The block decomposition follows FissionSA combo1's A/B/C/D weight blocks.
Timing retains the revision Tessera bank recurrence and Planaria array kernel.
Only the residual blocks fracture; no energy or EDP mapping search runs here.
"""
from functools import lru_cache
import heapq
from math import ceil

from neusim.npusim.backend.tessera import ceildiv


def geometry(k, n, grain, side, planes, family):
    h,w=(min(k,n),max(k,n)) if family=='Tessera' else (k,n)
    return k,n,k==n,min(planes,side*side//(h*w)),h,w


def groups(n,k,side,grain,planes,family):
    kb,kr=divmod(k,side);nb,nr=divmod(n,side)
    ks,ns=ceildiv(kr,grain),ceildiv(nr,grain)
    return tuple((geometry(kk,nn,grain,side,planes,family),nk,nt) for kk,nn,nk,nt in
        ((side,grain,kb,ns),(grain,side,ks,nb),(grain,grain,ks,ns)) if nk and nt)


def lane_time(m,g,folds,grain,family):
    if family=='Tessera':
        from neusim.npusim.backend.tessera_partitioned import lane_cycles
        return lane_cycles(m,g,'full',grain,folds)
    h,w=g[-2:]
    # Source: tessera_baselines.array_cost -> combo0 native Planaria formula.
    return folds*(max(m,grain) if folds>=3 else m)+grain+h+w-2


@lru_cache(maxsize=16384)
def tail_plan(occupied,side,grain,m,work,family,audit=False):
    """Pack mixed residual groups into available physical regions.

    Each homogeneous group distributes its folds over available first-fit lanes.
    Other groups fill remaining holes and newly retired lanes. Returned masks
    pin each used grain cell until its last planned use; this permits subsequent
    requests to occupy retired cells without stealing a later residual's cell.
    External occupied cells are never assumed to retire during this local plan.
    """
    from neusim.npusim.backend.tessera_partitioned import FirstFitFabric,packed_lanes
    fabric=FirstFitFabric(side,grain);fabric.rows=list(occupied)
    pending=list(work);events=[];serial=0;now=0
    last=[[0]*(side//grain) for _ in occupied];trace=[]
    while pending:
        progressed=False
        for group in pending[:]:
            g,nk,nn=group;h,w=g[-2:]
            limit=packed_lanes(nk,nn,side*side//(h*w),g[3])
            boxes=fabric.available(h,w,limit)
            if not boxes:continue
            width=len(boxes)
            for lane,box in enumerate(boxes):
                r,c,hh,ww=box;folds=ceildiv(nk*nn-lane,width)
                end=now+lane_time(m,g,folds,grain,family)
                mask=((1<<(ww//grain))-1)<<(c//grain)
                for row in range(r//grain,(r+hh)//grain):
                    assert not fabric.rows[row]&mask
                    fabric.rows[row]|=mask
                    for col in range(c//grain,(c+ww)//grain):last[row][col]=end
                heapq.heappush(events,(end,serial,box));serial+=1
                if audit:trace.append((now,end,box,g,folds))
            pending.remove(group);progressed=True
        if not pending:break
        if not events:return None
        now=events[0][0]
        while events and events[0][0]==now:
            _,_,(r,c,h,w)=heapq.heappop(events)
            mask=((1<<(w//grain))-1)<<(c//grain)
            for row in range(r//grain,(r+h)//grain):
                assert fabric.rows[row]&mask==mask
                fabric.rows[row]^=mask
    retire={};final=list(occupied)
    for row,values in enumerate(last):
        for col,end in enumerate(values):
            if end:
                masks=retire.setdefault(end,[0]*len(occupied));masks[row]|=1<<col;final[row]|=1<<col
    assert all((a&b)==a for a,b in zip(occupied,final))
    result=(tuple(final),tuple((dt,tuple(mask)) for dt,mask in sorted(retire.items())))
    return (*result,tuple(trace)) if audit else result


def required_planes(n,k,side,grain):
    kb,kr=divmod(k,side);nr=n%side;ks=ceildiv(kr,grain)
    # Full-size bulk folds have one producer at a time. Residual N strips
    # can overlap K producers; the corner can coexist with those strips.
    return max(1,ks+(min(kb,side//grain) if nr else 0))


@lru_cache(maxsize=65536)
def memory_tile(m,n,k,a_bytes,b_bytes,capacity,side,grain,freq,bw,transfer_only=False):
    from neusim.npusim.backend.tessera_partitioned import memory_tile as native,parts
    # chenyi9: decision start -- reserve the actual global tail producer count.
    planes=required_planes(n,k,side,grain) if transfer_only else 1
    while True:
        mt,nt,kt,_=native(m,n,k,a_bytes,b_bytes,4,planes,capacity,side,freq,bw)
        needed=planes if transfer_only else max(required_planes(nn,kk,side,grain) for nn,_ in parts(n,nt) for kk,_ in parts(k,kt))
        if needed<=planes:
            peak=2*a_bytes*mt*kt+2*b_bytes*kt*nt+(planes+1)*4*mt*nt
            assert peak<=capacity
            return mt,nt,kt,planes,peak
        planes=needed
    # chenyi9: decision end


@lru_cache(maxsize=131072)
def tile_cycles(m,n,k,side,grain,planes,family):
    full=(k//side)*(n//side)
    cycles=lane_time(m,geometry(side,side,grain,side,1,family),full,grain,family) if full else 0
    work=groups(n,k,side,grain,planes,family)
    if work:
        plan=tail_plan((0,)*(side//grain),side,grain,m,work,family)
        assert plan is not None and plan[1]
        cycles+=max(dt for dt,_ in plan[1])
    return cycles


@lru_cache(maxsize=131072)
def counts(b,m,n,k,a_bytes,b_bytes,out_bytes,config_json,resident=False):
    from neusim.configs.chips.ChipConfig import ChipConfig
    from neusim.npusim.backend.tessera_partitioned import parts,BACKEND
    config=ChipConfig.model_validate_json(config_json)
    grain=int(config.tessera_parameters['grain']);side=config.sa_dim
    arch=config.tessera_parameters.get('baseline_architecture')
    transfer_only=config.tessera_parameters.get('sram_tiling_model')=='transfer_only'
    if arch in ('WS','WS-independent'):
        from neusim.npusim.backend import tessera_baselines as base
        g=next(base.geometries(config))
        return base.counts(b,m,n,k,a_bytes,b_bytes,out_bytes,g,arch,config.vmem_size_MB*1024**2,
            config.freq_GHz,config.hbm_bw_GBps,resident=resident,sa_energy_accounting='padded_tiles',
            sram_read_accounting='padded_tiles',grain=grain,transfer_only=transfer_only)
    family='Planaria' if arch in ('Planaria','Planaria-32') else 'Tessera'
    if family=='Tessera' and config.tessera_variant!='full':
        raise ValueError('native bulk/tail mode currently requires full Tessera')
    mt,nt,kt,planes,peak=memory_tile(m,n,k,a_bytes,b_bytes,config.vmem_size_MB*1024**2,
        side,grain,config.freq_GHz,config.hbm_bw_GBps,transfer_only)
    cycles=aw=bw=pw=pa=pb=charged=reductions=rounds=0
    # chenyi9: decision start -- fracture actual tails, not HBM transfer edges.
    am,an,ak=(m,n,k) if transfer_only else (mt,nt,kt)
    for mm,mc in parts(m,am):
        for nn,nc in parts(n,an):
            kgroups=0
            for kk,kc in parts(k,ak):
                repeat=b*mc*nc*kc;kb,kr=divmod(kk,side);nb,nr=divmod(nn,side)
                ks,ns=ceildiv(kr,grain),ceildiv(nr,grain)
                kp,np=kb*side+ks*grain,nb*side+ns*grain
                cycles+=repeat*tile_cycles(mm,nn,kk,side,grain,planes,family)
                aw+=repeat*mm*kk*(nb+ns);bw+=repeat*kk*nn
                pa+=repeat*mm*kp*(nb+ns);pb+=repeat*kp*np
                pw+=repeat*mm*nn*(kb+ks);kgroups+=kc*(kb+ks)
                charged+=repeat*mm*kp*np
                rounds+=repeat*((kb*nb)+sum(ceildiv(nk*nnn,min(side*side//(g[-2]*g[-1]),g[3]*nnn))
                    for g,nk,nnn in groups(nn,kk,side,grain,planes,family)))
            reductions+=b*mc*nc*mm*nn*(kgroups-1)
    ha=b*m*k*ceildiv(n,nt)*a_bytes;hb=b*k*n*ceildiv(m,mt)*b_bytes;hc=b*m*n*out_bytes
    if resident:ha=hb=hc=0
    padding=(pa-aw)*a_bytes+(pb-bw)*b_bytes
    sram=pa*a_bytes+pb*b_bytes+pw*4+reductions*12+b*m*n*4+ha+hb
    macs=b*m*n*k
    assert 0<=padding and macs<=charged<=cycles*side*side
    phase=dict(native_tail=True,engine='SA',M=m,N=n,K=k,B=b,memory_tile=[mt,nt,kt],producer_planes=planes,
        family=family,sa_ns=ceil(cycles/config.freq_GHz),array_tile=[am,an,ak])
    return dict(model=BACKEND,engine='SA',variant=config.tessera_variant,packing_policy='native_bulk_tail_v1',
        B=b,M=m,N=n,K=k,useful_macs=macs,sa_cycles=cycles,sa_arithmetic_ops=2*charged,
        charged_sa_macs=charged,padding_macs=charged-macs,sa_energy_accounting='padded_tiles',
        sram_read_accounting='padded_tiles',padding_sram_read_bytes=padding,reduction_ops=reductions,
        vu_arithmetic_ops=reductions,array_a_read_bytes=pa*a_bytes,array_b_read_bytes=pb*b_bytes,
        partial_write_bytes=pw*4,reduction_read_bytes=reductions*8,reduction_write_bytes=reductions*4,
        final_read_bytes=b*m*n*4,sram_bytes=sram,hbm_a_read_bytes=ha,hbm_b_read_bytes=hb,
        hbm_output_write_bytes=hc,hbm_partial_read_bytes=0,hbm_partial_write_bytes=0,hbm_bytes=ha+hb+hc,
        peak_live_bytes=peak,memory_tile=[mt,nt,kt],producer_planes=planes,rounds=rounds,skew_tail_cycles=0,
        phase_plans=[phase],array_tile=[am,an,ak],
        sram_tiling_model='transfer_only' if transfer_only else 'restart_per_tile')
    # chenyi9: decision end


def compute_matmul(I,op,config,b,m,n,k):
    from neusim.npusim.backend import util
    from neusim.npusim.backend.tessera_partitioned import issue_ns
    if config.use_vu_for_small_matmul:raise ValueError('native bulk/tail comparison uses the SA engine')
    detail=dict(counts(b,m,n,k,*(util.get_size_bytes_from_dtype(x) for x in
        (I.input_axes[0][0].data_type,I.input_axes[1][0].data_type,I.output_axes[0].data_type)),
        config.model_dump_json(),(op.tessera_spec or {}).get('resident',False)))
    op.stats.tessera_details=detail;op.stats.num_sa_ops=detail['rounds']
    op.stats.einsum_B_size,op.stats.einsum_M_size=b,m
    op.stats.einsum_N_size,op.stats.einsum_K_size=n,k
    return ceil(detail['sa_cycles']/config.freq_GHz),issue_ns(detail['reduction_ops'],config)
