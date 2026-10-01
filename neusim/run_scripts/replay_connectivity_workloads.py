"""Parallelize the existing connectivity replay across independent workloads."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import math
from pathlib import Path
import resource
import time
from neusim.run_scripts import replay_tessera_arrivals as replay
from neusim.run_scripts.prepare_feature_cnn_qos import rows,sha,save_json,save_csv


def worker(args,job):
    arch,bw,wid=job
    out=args.out/'jobs'/wid/arch;out.mkdir(parents=True,exist_ok=False)
    replay.initialize_run(args.costs,args.store,args.programs,out)
    replay.RUN_INPUT['totals']=[r for r in replay.RUN_INPUT['totals'] if r['workload_id']==wid]
    stem=replay.run_job((arch,bw,'region_async'))
    data=json.loads((out/(stem+'.json')).read_text())
    assert data['status']=='PASS' and len(data['totals'])==1
    return data['totals'][0]


def main():
    p=argparse.ArgumentParser()
    for key in ('costs','store','programs','out','reuse'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--memory-gb',type=int,default=160)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    assert 1<=a.workers<=64 and 0<a.memory_gb<=200
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers])
    cap=int(a.memory_gb*1e9)//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    manifest=json.loads((a.costs/'manifest.json').read_text());assert manifest['status']=='PASS'
    previous=json.loads((a.reuse/'scenario.json').read_text())
    for path,digest in previous['source_sha256'].items():assert sha(Path(path))==digest,path
    for folder in (a.store,a.programs):assert json.loads((folder/'verification.json').read_text())['status']=='PASS'
    originals=list(rows(a.costs/'totals.csv'))
    jobs=[(r['architecture'],int(r['bandwidth_bytes_per_second']),r['workload_id']) for r in originals]
    assert len(jobs)==len(set(jobs))==24
    inputs=[Path(__file__),Path(replay.__file__),Path(replay.__file__).with_name('tessera_request_dispatch.py'),
        a.costs/'manifest.json',a.store/'verification.json',a.programs/'verification.json',a.reuse/'scenario.json']
    hashes={str(f.resolve()):sha(f) for f in inputs}
    save_json(a.out/'scenario.json',dict(jobs=[list(j) for j in jobs],workers=a.workers,memory_gb=a.memory_gb,
        source_sha256=hashes,accounting=previous['accounting'],admission=previous['admission'],
        spatial_scope=previous['spatial_scope'],energy=previous['energy']))
    done=[];pending=[];reused=[];start=time.monotonic()
    for arch,bw,wid in jobs:
        f=a.reuse/(arch+'-'+str(bw)+'-region_async-'+wid+'-workload.json')
        if f.exists():
            data=json.loads(f.read_text());assert data['status']=='PASS'
            done.append(data['total']);reused.append(dict(source=str(f.resolve()),sha256=sha(f)))
        else:pending.append((arch,bw,wid))
    print('Verified reused checkpoints:',len(reused),'remaining:',len(pending),flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures={pool.submit(worker,a,job):job for job in pending}
        for future in as_completed(futures):
            r=future.result();done.append(r)
            print('PASS',len(done),'/',len(jobs),r['architecture'],r['workload_id'],round(time.monotonic()-start,1),flush=True)
    lookup={(r['architecture'],int(r['bandwidth_bytes_per_second']),r['workload_id']):r for r in done}
    assert set(lookup)==set(jobs)
    for old in originals:
        r=lookup[old['architecture'],int(old['bandwidth_bytes_per_second']),old['workload_id']]
        assert int(r['useful_macs'])==int(old['useful_macs'])
        assert r['time_ns']>=r['initial_idle_ns']>=0
        assert math.isclose(r['edp_J_s'],r['system_energy_J']*r['seconds'],rel_tol=2e-12)
        assert math.isclose(r['energy_J'],sum(float(r[k]) for k in replay.ENERGIES),rel_tol=2e-12)
    keys=list(dict.fromkeys(k for r in done for k in r))
    save_csv(a.out/'totals.csv',[{k:lookup[job].get(k,0) for k in keys} for job in jobs])
    assert all(sha(Path(path))==digest for path,digest in hashes.items())
    save_json(a.out/'verification.json',dict(status='PASS',jobs=len(done),records=len(done),reused=reused,
        source_sha256=hashes,elapsed_s=time.monotonic()-start,
        artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print('PASS: all matched connectivity E2E workloads',flush=True)


if __name__=='__main__':main()
