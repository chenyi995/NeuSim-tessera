"""Independent physical-tile enumeration for padded SRAM and compute energy."""
import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import shutil
import traceback

from neusim.npusim.backend import tessera_joint as joint, tessera_partitioned as part, tessera_baselines as base
from neusim.run_scripts import run_tessera_joint as runner
from neusim.run_scripts.compare_tessera_native import ROOT, sha
from neusim.npusim.frontend.llm_ops_lib import create_multi_head_flash_attention_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info


def golden(shape,g,tile,arch,grain):
    """Enumerate assigned tiles without calling production count helpers."""
    b,m,n,k=shape;mt,nt,kt=tile
    def extents(size,step):
        return [min(step,size-i) for i in range(0,size,step)]
    def allocated(size,minimum):
        p=minimum
        while p<size:p*=2
        return p
    macs=aw=bw=0
    for mm in extents(m,mt):
        for nn in extents(n,nt):
            for kk in ([k] if arch=='SISA' else extents(k,kt)):
                if arch=='SISA':
                    for r in extents(mm,128):
                        h=allocated(r,16)
                        for _ in extents(nn,128):
                            macs+=h*128*kk;aw+=h*kk;bw+=128*kk
                elif arch=='FlexSA':
                    kh=[128]*(kk//128)+[64]*len(extents(kk%128,64))
                    nw=[128]*(nn//128)+[64]*len(extents(nn%128,64))
                    for h in kh:
                        for w in nw:
                            macs+=mm*h*w;aw+=mm*h;bw+=h*w
                else:
                    if arch:
                        rk,rn=g[-2:];h,w=rk,rn
                    else:
                        rk,rn=g[:2]
                        h,w=allocated(min(k,rk),grain),allocated(min(n,rn),grain)
                        if g[2]:h=w=max(h,w)
                    for _ in extents(kk,rk):
                        for _ in extents(nn,rn):
                            macs+=mm*h*w;aw+=mm*h;bw+=h*w
    return b*macs,b*aw*2,b*bw*2


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=1e-18),(a,b)


def scalar(shape,g,tile,c,resident=False):
    common=(*shape,2,2,2,g)
    suffix=(c.vmem_size_MB*1024**2,c.freq_GHz,c.hbm_bw_GBps,resident,tile,'padded_tiles','padded_tiles')
    arch=c.tessera_parameters.get('baseline_architecture')
    return base.counts(*common,arch,*suffix) if arch else part.counts(*common,c.tessera_variant,c.tessera_parameters['grain'],c.sa_dim,*suffix)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out.resolve();assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    shutil.copy2(__file__,out/Path(__file__).name)
    report=dict(status='running',integer_checks=0,native_checks=0,sram_delta_checks=0,winner_checks=0,attention_checks=0)
    grid=[394_000_000_000,2_800_000_000_000]
    runner.initialize(grid,padded_energy=True)
    try:
        for (name,bw),c in runner.CONFIGS.items():
            arch=c.tessera_parameters.get('baseline_architecture');grain=c.tessera_parameters['grain']
            pj=4.41 if name=='WS' or name.startswith('Tessera') else 4.54
            close(c.dynamic_power_sa_W/c.peak_SA_tflops_per_sec,pj)
            for shape in ((1,3,14,16),(2,5,13,23),(1,16,32,8),(1,129,193,257)):
                b,m,n,k=shape
                gs=list(base.geometries(c) if arch else part.geometries(m,n,k,c))
                if m>100:gs=gs[::max(1,len(gs)//8)]
                for g in gs:
                    tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                    v=joint.candidate_vectors(*shape,2,g,c,tiles)
                    indices=range(len(tiles[0])) if m<100 else sorted({0,len(tiles[0])//2,len(tiles[0])-1})
                    for i in indices:
                        tile=tuple(int(t[i]) for t in tiles[:3]);expected=golden(shape,g,tile,arch,grain)
                        s=scalar(shape,g,tile,c)
                        for field,value in zip(('charged_sa_macs','array_a_read_bytes','array_b_read_bytes'),expected):
                            assert s[field]==int(v[field][i])==value,(name,shape,g,tile,field,s[field],v[field][i],value)
                        for field in ('sram_bytes','hbm_bytes','padding_sram_read_bytes','reduction_ops'):
                            assert s[field]==int(v[field][i]),(name,shape,g,tile,field)
                        report['integer_checks']+=1
                    # Non-divisor SRAM boundary tiles exercise scalar edge counting.
                    tile=tuple(max(1,x//2+1) for x in (m,n,k))
                    s=scalar(shape,g,tile,c)
                    assert tuple(s[f] for f in ('charged_sa_macs','array_a_read_bytes','array_b_read_bytes'))==golden(shape,g,tile,arch,grain)
                    report['integer_checks']+=1
                    for i in sorted({0,len(tiles[0])-1}):
                        tile=tuple(int(t[i]) for t in tiles[:3])
                        for resident in (False,True):
                            vv=joint.candidate_vectors(*shape,2,g,c,tiles,resident)
                            edp,times,energies=joint.native_objective(vv,b*m*n*k,c,bw)
                            op=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,c,resident,tile);s=op.stats
                            assert s.execution_time_ns==times[i]
                            close(s.total_energy_J,energies[i]);close(s.total_energy_J*s.execution_time_ns,edp[i])
                            close(s.dynamic_energy_sa_J*op.dvfs_sa.voltage_conversion_power_efficiency_percent/100,
                                  2*int(vv['charged_sa_macs'][i])*pj*1e-12)
                            report['native_checks']+=1
                            old=c.model_copy(deep=True);old.tessera_parameters['sram_read_accounting']='useful'
                            before=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,old,resident,tile)
                            assert before.stats.execution_time_ns==s.execution_time_ns
                            assert before.stats.memory_traffic_bytes==s.memory_traffic_bytes
                            close(before.stats.dynamic_energy_sa_J,s.dynamic_energy_sa_J)
                            delta=s.tessera_details['sram_bytes']-before.stats.tessera_details['sram_bytes']
                            assert delta==int(vv['padding_sram_read_bytes'][i])
                            after_j=s.dynamic_energy_sram_J*op.dvfs_sram.voltage_conversion_power_efficiency_percent/100
                            before_j=before.stats.dynamic_energy_sram_J*before.dvfs_sram.voltage_conversion_power_efficiency_percent/100
                            close(after_j-before_j,c.dynamic_power_vmem_W*delta/c.vmem_bw_GBps/1e9)
                            report['sram_delta_checks']+=1
            shape=(1,3,5,7);b,m,n,k=shape
            for resident in (False,True):
                best=float('inf')
                for g in (base.geometries(c) if arch else part.geometries(m,n,k,c)):
                    tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                    for tt in zip(*tiles[:3]):
                        s=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,c,resident,tuple(map(int,tt))).stats
                        best=min(best,s.total_energy_J*s.execution_time_ns)
                g,tile=joint.select_mapping(*shape,'DT_BFLOAT16',c.model_dump_json(),resident)
                s=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,c,resident,tile).stats
                close(best,s.total_energy_J*s.execution_time_ns);report['winner_checks']+=1
            op=create_multi_head_flash_attention_op([1,3,4,16],[1,19,2,16],[1,19,2,16])
            with redirect_stdout(part.Sink()):s=fill_operators_execution_info([op],c)[0].stats
            d=s.tessera_details;expected=[0,0,0];padding=0
            for pp in d['phase_plans']:
                shape=(1,pp['M'],pp['N'],pp['K']);gg=tuple(pp['geometry']);tt=tuple(pp['memory_tile'])
                expected=[a+pp['copies']*bb for a,bb in zip(expected,golden(shape,gg,tt,arch,grain))]
                padding+=pp['copies']*scalar(shape,gg,tt,c,True)['padding_sram_read_bytes']
            assert expected==[d[f] for f in ('charged_sa_macs','array_a_read_bytes','array_b_read_bytes')]
            assert padding==d['padding_sram_read_bytes'];report['attention_checks']+=1
            print(name,bw,report,flush=True)
        c=runner.CONFIGS['Tessera-8-square',max(grid)]
        s=scalar((1,3,14,16),(16,16,True,1,16,16),(3,14,16),c)
        assert s['array_a_read_bytes']==3*16*2 and s['array_b_read_bytes']==16*16*2
        assert s['charged_sa_macs']==3*16*16 and s['padding_sram_read_bytes']==16*(16-14)*2
        report['example_16x14']={f:s[f] for f in ('useful_macs','charged_sa_macs','array_a_read_bytes','array_b_read_bytes','padding_sram_read_bytes')}
        files=[Path(__file__),ROOT/'neusim/run_scripts/run_tessera_joint.py',ROOT/'neusim/run_scripts/run_tessera_partitioned.py',
               ROOT/'neusim/configs/chips/ChipConfig.py',ROOT/'neusim/npusim/backend/power_model.py',
               ROOT/'neusim/npusim/backend/tessera_joint.py',ROOT/'neusim/npusim/backend/tessera_partitioned.py',ROOT/'neusim/npusim/backend/tessera_baselines.py']
        report.update(status='PASS',source_sha256={str(p):sha(p) for p in files})
    except BaseException:
        report['status']='FAIL';(out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        with (out/'verification.json').open('x') as f:json.dump(report,f,indent=2)
        assert json.loads((out/'verification.json').read_text())==report
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
