"""Audit nondivisor search against scalar native kernels and literal tile ordering."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from neusim.npusim.backend import tessera_joint as joint, tessera_partitioned as part, tessera_baselines as baseline
from neusim.run_scripts.run_tessera_partitioned import config
from neusim.run_scripts.run_tessera_joint import apply_array_energy
from neusim.run_scripts.tessera_request_dispatch import segments
from neusim.run_scripts.compare_tessera_native import ROOT,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    checks=energy_checks=schedule_checks=0
    configs=[]
    for grain in (2,4,8,16,32,64,128):
        for variant in ('full','independent_noskew'):
            configs.append((f'Tessera-{grain}-{variant}',config(variant,grain=grain)))
    for arch in baseline.ARCHITECTURES:
        c=config('full',grain=32);c.tessera_parameters['baseline_architecture']=arch;configs.append((arch,c))
    for name,c in configs:
        apply_array_energy(c,name)
        c.tessera_parameters.update(sa_energy_accounting='padded_tiles',sram_read_accounting='padded_tiles',mapping_policy='joint_edp')
        arch=c.tessera_parameters.get('baseline_architecture')
        for shape in ((1,3,5,7),(2,17,33,67),(1,257,130,259)):
            b,m,n,k=shape
            gs=list(baseline.geometries(c) if arch else part.geometries(m,n,k,c))
            for g in gs:
                tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                v=joint.candidate_vectors(*shape,2,g,c,tiles)
                # Uniform index coverage includes both exact and remainder candidates.
                indices=np.unique(np.linspace(0,len(tiles[0])-1,min(17,len(tiles[0])),dtype=int))
                # Literal trace enumeration is bounded; full-size scalar counters
                # remain checked for every sampled candidate, including tile=1.
                schedules=[i for i in indices if b*math.ceil(m/tiles[0][i])*math.ceil(n/tiles[1][i])*math.ceil(k/tiles[2][i])<=4096]
                schedules=set(schedules[::max(1,len(schedules)//3)])
                for i in indices:
                    tile=tuple(int(x[i]) for x in tiles[:3])
                    kwargs=dict(forced_tile=tile,sa_energy_accounting='padded_tiles',sram_read_accounting='padded_tiles')
                    if arch:
                        r=baseline.counts(*shape,2,2,2,g,arch,c.vmem_size_MB*1024**2,c.freq_GHz,c.hbm_bw_GBps,**kwargs)
                    else:
                        r=part.counts(*shape,2,2,2,g,c.tessera_variant,c.tessera_parameters['grain'],c.sa_dim,
                                      c.vmem_size_MB*1024**2,c.freq_GHz,c.hbm_bw_GBps,**kwargs)
                    for field in ('sram_bytes','hbm_bytes','charged_sa_macs','padding_sram_read_bytes','rounds',
                                  'reduction_ops','peak_live_bytes','array_a_read_bytes','array_b_read_bytes'):
                        assert int(v[field][i])==r[field],(name,shape,g,tile,field,int(v[field][i]),r[field])
                    assert int(v['sa_ns'][i])==math.ceil(r['sa_cycles']/c.freq_GHz)
                    assert int(v['vu_ns'][i])==part.issue_ns(r['reduction_ops'],c)
                    checks+=1
                    if not arch and i in schedules:
                        phase=dict(B=b,M=m,N=n,K=k,geometry=g,memory_tile=tile,sa_ns=int(v['sa_ns'][i]))
                        costs=SimpleNamespace(side=c.sa_dim,grain=c.tessera_parameters['grain'],chip=c)
                        got=list(segments(dict(phases=[phase]),costs))
                        expanded=[(s['m'],s['nk'],s['nn']) for s in got for _ in range(s['groups'])]
                        expected=[]
                        # Separate literal original-index traversal, not vector counter reuse.
                        for _ in range(b):
                            for ms in range(0,m,tile[0]):
                                for ns in range(0,n,tile[1]):
                                    for ks in range(0,k,tile[2]):
                                        expected.append((min(tile[0],m-ms),math.ceil(min(tile[2],k-ks)/g[0]),math.ceil(min(tile[1],n-ns)/g[1])))
                        assert expanded==expected,(name,shape,g,tile)
                        schedule_checks+=1
                for i in indices[:1]:
                    tile=tuple(int(x[i]) for x in tiles[:3])
                    s=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,c,forced_tile=tile).stats
                    edp,time,energy=joint.native_objective(v,b*m*n*k,c,c.hbm_bw_GBps*1024**3)
                    assert s.execution_time_ns==time[i]
                    assert math.isclose(s.total_energy_J,energy[i],rel_tol=2e-12),(name,shape,g,tile,s.total_energy_J,energy[i])
                    energy_checks+=1
        print('PASS',name,checks,energy_checks,schedule_checks,flush=True)
    # Reproduce the two actual prime-length failures with the saved energy configuration.
    original=ROOT/'results/tessera/20260930_async_full_v1'
    cases=json.loads((original/'cacheblend_tile_search_diagnostic_v1/diagnostic.json').read_text())
    saved=json.loads((original/'main_costs/configs.json').read_text())
    from neusim.configs.chips.ChipConfig import ChipConfig
    winners=[]
    for case in cases['results']:
        m,n,k=case['shape']
        for name in ('Tessera-8','Tessera-8-independent_noskew'):
            c=ChipConfig.model_validate(saved[name+'@2800000000000'])
            g,t=joint.select_mapping(1,m,n,k,'DT_BFLOAT16',c.model_dump_json())
            s=part.evaluate_candidate(1,m,n,k,'DT_BFLOAT16',g,c,forced_tile=t).stats
            olds=[r for r in case['variants'] if r['architecture']==name]
            assert s.total_energy_J*s.execution_time_ns<=min(r['edp_J_ns'] for r in olds)*(1+1e-12)
            winners.append(dict(cohort=case['cohort'],arch=name,geometry=g,tile=t,time_ns=s.execution_time_ns,
                energy_J=s.total_energy_J,edp_J_ns=s.total_energy_J*s.execution_time_ns))
    files=[Path(__file__),ROOT/'neusim/npusim/backend/tessera_joint.py',ROOT/'neusim/run_scripts/run_tessera_joint.py',
        ROOT/'neusim/run_scripts/run_tessera_partitioned.py',ROOT/'neusim/run_scripts/tessera_request_dispatch.py',
        ROOT/'neusim/npusim/backend/tessera_partitioned.py',ROOT/'neusim/npusim/backend/tessera_baselines.py']
    payload=dict(status='PASS',scalar_counter_checks=checks,native_energy_checks=energy_checks,
        literal_schedule_checks=schedule_checks,real_prime_cases=winners,source_sha256={str(p):sha(p) for p in files})
    (out/'verification.json').write_text(json.dumps(payload,indent=2))
    assert json.loads((out/'verification.json').read_text())==json.loads(json.dumps(payload))
    print(json.dumps(payload,indent=2),flush=True)


if __name__=='__main__':main()
