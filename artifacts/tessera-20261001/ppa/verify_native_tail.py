"""Independent native-tile, literal operand, physical-placement and replay checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,json,math,random,sys,traceback
from pathlib import Path

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
sys.path.insert(0,str(ROOT))
from neusim.npusim.backend import tessera_native_tail as tail,tessera_partitioned as part
from neusim.run_scripts.run_tessera_ppa import configurations,sha
from neusim.run_scripts.run_tessera_partitioned import native_record,ENERGIES
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op,create_multi_head_flash_attention_op
from neusim.npusim.backend.tests.test_tessera import golden_bank_events
from neusim.run_scripts.replay_tessera_arrivals import Costs
from neusim.run_scripts.tessera_request_dispatch import segments,chains_run,dispatch


def literal(shape,detail,grain):
    b,m,n,k=shape;mt,nt,kt=detail['memory_tile']
    mac=charged=pa=pb=pw=reductions=ha=hb=hc=aw=bw=0
    def weights(size):
        full=size//128
        for _ in range(full):yield 128,128
        rem=size-full*128
        for start in range(0,rem,grain):yield min(grain,rem-start),grain
    for _ in range(b):
        for mi in range(0,m,mt):
            mm=min(mt,m-mi)
            for ni in range(0,n,nt):
                nn=min(nt,n-ni);kgroups=0
                for ki in range(0,k,kt):
                    kk=min(kt,k-ki);ha+=mm*kk*2;hb+=kk*nn*2
                    for ka,kpad in weights(kk):
                        kgroups+=1
                        for na,npad in weights(nn):
                            mac+=mm*ka*na;charged+=mm*kpad*npad
                            aw+=mm*ka*2;bw+=ka*na*2
                            pa+=mm*kpad*2;pb+=kpad*npad*2;pw+=mm*na*4
                reductions+=mm*nn*(kgroups-1);hc+=mm*nn*2
    return dict(useful_macs=mac,charged_sa_macs=charged,array_a_read_bytes=pa,array_b_read_bytes=pb,
        partial_write_bytes=pw,reduction_ops=reductions,hbm_bytes=ha+hb+hc,
        padding_sram_read_bytes=pa+pb-aw-bw,sram_bytes=pa+pb+pw+12*reductions+4*b*m*n+ha+hb)


def placement(m,n,k,grain,family):
    side=128;planes=tail.required_planes(n,k,side,grain)
    work=tail.groups(n,k,side,grain,planes,family)
    if not work:return 0
    tested=0
    for occupied in ((0,)*(side//grain),tuple(1 if row==0 else 0 for row in range(side//grain))):
        plan=tail.tail_plan(occupied,side,grain,m,work,family,True)
        if plan is None:continue
        final,retire,events=plan
        live={};expected_last={};charged=0
        for start,end,box,g,folds in events:
            r,c,h,w=box
            if family=='Tessera':
                expected=golden_bank_events(m,h,folds,False)[-1][2]+w+(w//grain-1)
            else:
                from neusim.npusim.backend.tessera_baselines import array_cost
                one=(h,w,True,1,h,w)
                expected=array_cost('Planaria',m,w,h*folds,one,grain)[0]
            assert end-start==expected
            charged+=m*h*w*folds
            for y in range(r//grain,(r+h)//grain):
                for x in range(c//grain,(c+w)//grain):
                    assert not occupied[y]&(1<<x)
                    for a,b in live.get((y,x),[]):assert end<=a or start>=b
                    live.setdefault((y,x),[]).append((start,end));expected_last[y,x]=max(end,expected_last.get((y,x),0))
        assert charged==sum(m*g[-2]*g[-1]*nk*nn for g,nk,nn in work)
        for dt,mask in retire:
            for y,bits in enumerate(mask):
                for x in range(side//grain):
                    if bits&(1<<x):assert expected_last[y,x]==dt
        assert max(dt for dt,_ in retire)==max(end for _,end,*_ in events)
        tested+=len(events)
    return tested


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    out=parser.parse_args().out;out.mkdir(exist_ok=False)
    result=dict(status='RUNNING',native_tiles=0,literal_counter_cases=0,physical_lane_checks=0,
        native_operators=0,attention_cases=0,source_segments=0,concurrent_replays=0)
    try:
        for (name,_),chip in configurations(2_800_000_000_000,'native_tail')[0].items():
            grain=chip.tessera_parameters['grain']
            shapes=[(1,17,193,259),(1,31,256,384),(2,9,27,43)]
            for seed in (7,29,101):
                rng=random.Random(seed)
                shapes.append((1,rng.randrange(1,40),rng.randrange(1,512),rng.randrange(1,512)))
            records={};metadata={}
            for oid,shape in enumerate(shapes):
                b,m,n,k=shape
                detail=tail.counts(*shape,2,2,2,chip.model_dump_json())
                op=create_einsum_op([b,m,k],[b,k,n],'BMK;BKN->BMN')
                r=native_record(op,chip);records[oid]=r
                metadata[str(oid)]=dict(mxu=True,flops=2*math.prod(shape))
                assert r['useful_macs']==math.prod(shape)
                assert math.isclose(r['energy_J'],sum(r[x] for x in ENERGIES),rel_tol=2e-12)
                result['native_operators']+=1
                if name.startswith(('Tessera','Planaria')):
                    gold=literal(shape,detail,grain)
                    for key,value in gold.items():assert detail[key]==value,(name,shape,key,detail[key],value)
                    result['literal_counter_cases']+=1
                    tile=part.memory_tile(m,n,k,2,2,4,detail['producer_planes'],chip.vmem_size_MB*1024**2,
                        chip.sa_dim,chip.freq_GHz,chip.hbm_bw_GBps)
                    assert tuple(detail['memory_tile'])==tile[:3]
                    result['native_tiles']+=1
                    result['physical_lane_checks']+=placement(m,n,k,grain,'Tessera' if name.startswith('Tessera') else 'Planaria')
            if name.startswith('Tessera'):
                costs=Costs(records,metadata,chip)
                for oid in records:
                    ss=list(segments(costs.get(oid),costs));assert ss
                    result['source_segments']+=len(ss)
                for chain_order in ([[0,1],[2,3]],[[4],[5],[1]]):
                    trace=[];r=chains_run(chain_order,costs,'region_async',trace)
                    assert r['time_ns']>0 and len(trace)==sum(map(len,chain_order))
                    assert all(t['finish_ns']>t['start_ns'] for t in trace)
                    result['concurrent_replays']+=1
            for q,kv,hq,hkv,d in ((1,271,8,2,80),(19,145,8,8,64)):
                r=native_record(create_multi_head_flash_attention_op([1,q,hq,d],[1,kv,hkv,d],[1,kv,hkv,d]),chip)
                assert r['useful_macs']==2*hq*q*kv*d
                assert math.isclose(r['energy_J'],sum(r[x] for x in ENERGIES),rel_tol=2e-12)
                result['attention_cases']+=1
            print('PASS',name,result,flush=True)
        # Explicit over-capacity input: native SRAM tiling must fit live buffers.
        c=configurations(2_800_000_000_000,'native_tail')[0]['Tessera-8',2_800_000_000_000]
        shape=(1,257,2048,2048)
        detail=tail.counts(*shape,2,2,2,c.model_dump_json())
        assert detail['peak_live_bytes']<=c.vmem_size_MB*1024**2
        assert detail['memory_tile']!=list(shape[1:])
        gold=literal(shape,detail,8)
        for key,value in gold.items():assert detail[key]==value,(key,detail[key],value)
        result['literal_counter_cases']+=1
        result['status']='PASS'
    except BaseException:
        result['status']='FAIL';(out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        files=[Path(__file__),Path(tail.__file__),Path(part.__file__),ROOT/'neusim/run_scripts/tessera_request_dispatch.py',
            ROOT/'neusim/run_scripts/run_tessera_ppa.py',ROOT/'neusim/run_scripts/run_tessera_joint.py',
            ROOT/'neusim/npusim/backend/npusim_lib.py']
        result['source_sha256']={str(p):sha(p) for p in files}
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    assert json.loads((out/'verification.json').read_text())==result


if __name__=='__main__':main()
