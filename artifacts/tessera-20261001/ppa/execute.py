"""Run native bulk/tail and joint EDP independently within 64 CPUs / 200 GB."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,math,resource,subprocess,sys,time
from pathlib import Path

RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];sys.path.insert(0,str(ROOT))
PYTHON=ROOT/'.venv/bin/python'
PROGRAMS=ROOT/'results/tessera/20260930_energy_corrected_v1/trace_programs_v2'


def stage(folder,name,args):
    command=[str(PYTHON),'-B','-m',*args]
    with (folder/(name+'_command.json')).open('x') as f:json.dump(command,f,indent=2)
    print('START',folder.name,name,flush=True)
    with (folder/(name+'.log')).open('x') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS',folder.name,name,flush=True)


def replay_worker(folder,job):
    from neusim.run_scripts import replay_tessera_arrivals as replay
    arch,bw,mode,wid=job
    out=folder/'arrival_replay/jobs'/wid/arch;out.mkdir(parents=True,exist_ok=False)
    replay.initialize_run(folder/'main_costs',folder/'cost_store',PROGRAMS,out)
    replay.RUN_INPUT['totals']=[r for r in replay.RUN_INPUT['totals'] if r['workload_id']==wid]
    stem=replay.run_job((arch,bw,mode));result=json.loads((out/(stem+'.json')).read_text())
    assert result['status']=='PASS' and len(result['totals'])==1
    return result['totals'][0]


def run_mode(mapping,cpus):
    os.sched_setaffinity(0,cpus)
    # Per mode: profiling has <=92.16 GB of address-space caps plus this 4-GB
    # supervisor; replay has 12 workers plus supervisor, each capped at 4 GB.
    # Two modes use disjoint <=32-CPU sets and remain below the 200-GB budget.
    resource.setrlimit(resource.RLIMIT_AS,(4_000_000_000,4_000_000_000))
    folder=RUN/mapping;folder.mkdir(exist_ok=False);start=time.monotonic()
    stage(folder,'main_costs',['neusim.run_scripts.run_tessera_joint','--base',
        'results/tessera/archived/0929/20260929_paper_sram_per_pe','--verification',str(RUN/'combined_verification.json'),
        '--equal-pe-ppa','--ppa-tiling',mapping,'--workers','32','--memory-gb','96','--out',str(folder/'main_costs')])
    stage(folder,'cost_store',['neusim.run_scripts.replay_tessera_arrivals','index','--costs',str(folder/'main_costs'),
        '--out',str(folder/'cost_store'),'--workers','1','--memory-gb','4'])
    from neusim.run_scripts.prepare_feature_cnn_qos import rows,sha,save_json,save_csv
    from neusim.run_scripts import replay_tessera_arrivals as replay,tessera_request_dispatch as dispatch
    out=folder/'arrival_replay';out.mkdir(exist_ok=False)
    for p in (PROGRAMS,folder/'cost_store'):
        assert json.loads((p/'verification.json').read_text())['status']=='PASS'
    originals=list(rows(folder/'main_costs/totals.csv'))
    jobs=[(r['architecture'],int(r['bandwidth_bytes_per_second']),
        'region_async' if r['architecture'].startswith('Tessera') else 'serial',r['workload_id']) for r in originals]
    assert len(jobs)==len(set(jobs))==156
    files=[Path(__file__),Path(replay.__file__),Path(dispatch.__file__),ROOT/'neusim/npusim/backend/tessera_native_tail.py',
        folder/'main_costs/manifest.json',folder/'main_costs/configs.json',PROGRAMS/'verification.json',folder/'cost_store/verification.json']
    hashes={str(p):sha(p) for p in files}
    save_json(out/'scenario.json',dict(jobs=[list(j) for j in jobs],workers=12,cpus=list(cpus),
        per_process_limit_bytes=4_000_000_000,mapping_policy=mapping,source_sha256=hashes,
        accounting='Complete separate workloads; original arrivals, idle and dependencies. All Tessera grains pack within rounds and asynchronously admit independent adjacent requests. Revision NeuSim energy; RTL-scaled area only.'))
    print('START',mapping,'arrival_replay',len(jobs),flush=True)
    completed=[]
    with ProcessPoolExecutor(max_workers=12) as pool:
        futures=[pool.submit(replay_worker,folder,j) for j in jobs]
        for future in as_completed(futures):
            r=future.result();completed.append(r)
            print('PASS replay',mapping,len(completed),len(jobs),r['architecture'],r['workload_id'],flush=True)
    by={(r['architecture'],r['workload_id']):r for r in completed};assert len(by)==len(originals)
    for old in originals:
        r=by[old['architecture'],old['workload_id']]
        assert int(r['useful_macs'])==int(old['useful_macs'])
        assert float(r['hbm_bytes'])==float(old['hbm_bytes'])
        assert r['time_ns']>=r['initial_idle_ns']>=0
        assert math.isclose(r['edp_J_s'],r['system_energy_J']*r['seconds'],rel_tol=2e-12)
        assert math.isclose(r['energy_J'],math.fsum(float(r[k]) for k in replay.ENERGIES),rel_tol=2e-12)
    keys=list(dict.fromkeys(k for r in completed for k in r))
    save_csv(out/'totals.csv',[{k:by[r['architecture'],r['workload_id']].get(k,0) for k in keys} for r in originals])
    assert all(sha(Path(p))==v for p,v in hashes.items())
    save_json(out/'verification.json',dict(status='PASS',jobs=len(completed),records=len(completed),source_sha256=hashes,
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
    stage(folder,'report',['neusim.run_scripts.report_tessera_ppa','--run',str(folder),'--out',str(folder/'analysis')])
    save_json(folder/'completion.json',dict(status='PASS',mapping_policy=mapping,elapsed_seconds=time.monotonic()-start,
        configurations=13,workloads=12,records=len(completed)))
    return mapping


def main():
    from neusim.run_scripts.prepare_feature_cnn_qos import sha
    stage(RUN,'common_verification',['neusim.run_scripts.run_tessera_ppa','verify','--out',str(RUN/'common_verification')])
    checks=[RUN/'common_verification/verification.json',RUN/'native_verification_v2/verification.json']
    hashes={}
    for path in checks:
        data=json.loads(path.read_text());assert data['status']=='PASS'
        for p,digest in data['source_sha256'].items():
            assert sha(Path(p))==digest,p
            assert p not in hashes or hashes[p]==digest
            hashes[p]=digest
        hashes[str(path)]=sha(path)
    with (RUN/'combined_verification.json').open('x') as f:
        json.dump(dict(status='PASS',source_sha256=hashes),f,indent=2)
    cpus=sorted(os.sched_getaffinity(0))[:64]
    assert len(cpus)>=2
    midpoint=len(cpus)//2;sets=(cpus[:midpoint],cpus[midpoint:])
    assert not set(sets[0])&set(sets[1]) and max(map(len,sets))<=32
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run_mode,m,c) for m,c in zip(('native_tail','joint_edp'),sets)]
        for future in as_completed(futures):print('PASS complete',future.result(),flush=True)
    with (RUN/'completion.json').open('x') as f:
        json.dump(dict(status='PASS',policies=['native_tail','joint_edp'],records=312,
            cpu_budget=64,memory_budget_GB=200),f,indent=2)


if __name__=='__main__':main()
