"""Native-cost spatial replay with arrival gaps and request dependencies.

Native SRAM tiles acquire available FCRs and reconsider the available lanes at
each tile boundary. Each region retires after its own last native fold.
FlashAttention phases retain their source ordering and native internal head loop.
No intermediate phase is illegally moved and no dependent operators overlap.
HBM, VU and ICI use shared FIFO service under NeuSim's component overlap model.
This is an analytical replay, not a cycle-accurate operand-ready simulation.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse,csv,heapq,json,math,resource,sqlite3,time
from collections import Counter,OrderedDict
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
from pathlib import Path
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.tessera_partitioned import FirstFitFabric,packed_lanes,lane_cycles
from neusim.npusim.backend.dvfs_power_getter import FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,rows,sha,save_json,save_csv
from neusim.run_scripts.run_tessera_partitioned import ENERGIES

COMPONENTS=('sa','vu','sram','hbm','ici')
FREQUENCY_NS=1e9  # source: SI nanoseconds per second, no fitted latency.

def efficiency(activity):
    activity=min(1.,max(0.,activity))
    return next(r.power_efficiency_percent/100 for r in FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE if r.activity_factor>=activity)

class Costs:
    def __init__(self,records,metadata,chip):
        self.rows=records;self.meta=metadata;self.chip=chip
        self.side=chip.sa_dim;self.grain=int(chip.tessera_parameters['grain'])
        self.capacity=chip.vmem_size_MB*1024**2
        self.power={c:getattr(chip,'static_power_'+('vmem' if c=='sram' else c)+'_W') for c in COMPONENTS}
        self.power['other']=chip.static_power_other_W
        assert not chip.enable_dvfs
        self.profiles={}

    def get(self,oid):
        if oid in self.profiles:return self.profiles[oid]
        r=self.rows[oid];t=float(r['time_ns']);assert t>0
        # Native regulator activity uses op.stats.flops_util for SA and VU.
        meta=self.meta[str(oid)]
        peak=self.chip.peak_tflops_per_sec if meta['mxu'] else self.chip.peak_VU_tflops_per_sec
        work=dict(sa=min(t,meta['flops']/peak/1e3),vu=min(t,meta['flops']/peak/1e3))
        for c,k in [('sram','sram_ns'),('hbm','hbm_ns'),('ici','ici_ns')]:work[c]=min(t,float(r[k]))
        dynamic={}
        for c in COMPONENTS:
            eta=efficiency(work[c]/t)
            gold=self.power[c]*t/1e9/eta
            assert math.isclose(gold,float(r['static_energy_'+c+'_J']),rel_tol=2e-12,abs_tol=1e-18),(oid,c,gold,r['static_energy_'+c+'_J'])
            dynamic[c]=float(r['dynamic_energy_'+c+'_J'])*eta
        assert math.isclose(float(r['static_energy_other_J']),self.power['other']*t/1e9,rel_tol=2e-12)
        mapping=json.loads(r['mapping_json'])
        phases=mapping.get('phase_plans') or ([{**mapping,'sa_ns':float(r['sa_ns'])}] if 'geometry' in mapping else [])
        p=dict(row=r,time=t,work=work,dynamic=dynamic,phases=phases,
          sa=float(r['sa_ns']),peak=int(r['peak_live_bytes']),hbm=float(r['hbm_ns']),
          vu=float(r['vu_ns']),ici=float(r['ici_ns']))
        self.profiles[oid]=p;return p

    def integrate(self,dt,active):
        """One whole-chip static term; work-proportional native dynamic terms.

        Native fixed-voltage efficiencies are recomputed from concurrent
        average activity. Empty intervals retain the NoPG idle power.
        """
        ans=Counter()
        for c in COMPONENTS:
            activity=math.fsum(p['work'][c]/duration for p,duration in active)
            eta=efficiency(activity)
            ans['static_energy_'+c+'_J']=self.power[c]*dt/1e9/eta
            ans['dynamic_energy_'+c+'_J']=math.fsum(p['dynamic'][c]/duration for p,duration in active)*dt/eta
        ans['static_energy_other_J']=self.power['other']*dt/1e9
        ans['dynamic_energy_other_J']=self.chip.dynamic_power_other_W*dt/1e9
        return ans

    def serial(self,oids):
        acc=Counter()
        for oid in oids:
            r=self.rows[oid]
            for k in ENERGIES:acc[k]+=float(r[k])
            acc['time_ns']+=float(r['time_ns'])
            acc['array_busy_ns']+=float(r['sa_ns'])
        return acc

from neusim.run_scripts.tessera_request_dispatch import chains_run,batch_cost,replay_trace

def index_costs(source,out):
    """One streaming pass creates a read-only cost store for all sweep jobs."""
    manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='PASS'
    assert sha(source/'operator_costs.csv')==manifest['output_sha256']['operator_costs.csv']
    db=sqlite3.connect(out/'costs.sqlite');db.execute('pragma journal_mode=OFF')
    db.execute('create table costs (arch text, bw integer, kind text, oid integer, payload text, primary key(arch,bw,kind,oid)) without rowid')
    batch=[];count=0
    for r in rows(source/'operator_costs.csv'):
        batch.append((r['architecture'],int(r['bandwidth_bytes_per_second']),r['kind'],int(r['operator_id']),json.dumps(r,separators=(',',':'))))
        if len(batch)==10000:db.executemany('insert into costs values (?,?,?,?,?)',batch);count+=len(batch);batch=[]
    db.executemany('insert into costs values (?,?,?,?,?)',batch);count+=len(batch);db.commit()
    assert db.execute('select count(*) from costs').fetchone()[0]==count
    check=next(rows(source/'operator_costs.csv'))
    assert json.loads(db.execute('select payload from costs where arch=? and bw=? and kind=? and oid=?',(check['architecture'],int(check['bandwidth_bytes_per_second']),check['kind'],int(check['operator_id']))).fetchone()[0])==check
    db.close();save_json(out/'verification.json',dict(status='PASS',records=count,source_sha256=sha(source/'operator_costs.csv'),store_sha256=sha(out/'costs.sqlite')))

RUN_INPUT={}
def initialize_run(costs,store,programs,out):
    global RUN_INPUT
    source=Path(costs);programs=Path(programs)
    manifest=json.loads((source/'manifest.json').read_text())
    base=Path(manifest['base'])/'inputs_snapshot'
    metadata=json.loads((base/'workload_metadata.json').read_text())
    RUN_INPUT=dict(source=source,store=Path(store),programs=programs,out=Path(out),
      configs=json.loads((source/'configs.json').read_text()),
      operator_meta=json.loads((programs/'operator_metadata.json').read_text()),
      totals=list(rows(source/'totals.csv')),metadata=metadata,
      tags={(r['dataset'],r['cohort']):r['trace_tag'] for r in metadata['cohorts']})

def run_job(job):
    arch,bw,mode=job;state=RUN_INPUT;begun=time.monotonic()
    db=sqlite3.connect('file:'+str(state['store']/'costs.sqlite')+'?mode=ro',uri=True)
    records={oid:json.loads(payload) for oid,payload in db.execute('select oid,payload from costs where arch=? and bw=? and kind=?',(arch,bw,'graph'))};db.close()
    chip=ChipConfig.model_validate(state['configs'][arch+'@'+str(bw)])
    costs=Costs(records,state['operator_meta'],chip)
    answers=[];batches_report=[]
    for original in state['totals']:
        if original['architecture']!=arch or int(original['bandwidth_bytes_per_second'])!=bw:continue
        tag=state['tags'][original['dataset'],original['cohort']]
        if tag:
            batches=[json.loads(line) for line in (state['programs']/(tag+'.jsonl')).open()]
            result,done=replay_trace(batches,costs,mode)
            for b in batches:
                assert done[b['batch_id']]>=b['release_ns']
                assert all(done[d]<=done[b['batch_id']] for d in b['dependencies'])
                batches_report.append(dict(model=tag,batch_id=b['batch_id'],release_ns=b['release_ns'],finish_ns=done[b['batch_id']]))
        else:
            # Complete single-operator sourced workloads have no independent
            # request-arrival metadata. Preserve their native invocation sum.
            result=Counter({k:float(original[k]) for k in ENERGIES})
            result.update(time_ns=float(original['time_ns']),energy_J=float(original['energy_J']),
              array_busy_ns=float(original['sa_ns']),idle_ns=0.)
        r=dict(original);r.update(result)
        r['architecture']=arch if mode!='rounds' else arch+'-rounds'
        r['dispatch']=mode;r['source_architecture']=arch
        r['boundary']='original_arrival_to_last_completion' if tag else 'complete_sourced_operator_workload'
        r['seconds']=float(r['time_ns'])/1e9
        r['static_J']=math.fsum(float(r[k]) for k in ENERGIES if k.startswith('static_'))
        r['dynamic_J']=math.fsum(float(r[k]) for k in ENERGIES if k.startswith('dynamic_'))
        assert math.isclose(r['energy_J'],r['static_J']+r['dynamic_J'],rel_tol=2e-12)
        r['system_energy_J']=r['energy_J']*int(r['chips']);r['edp_J_s']=r['system_energy_J']*r['seconds']
        r['array_active_utilization']=float(r['useful_macs'])/(chip.sa_dim**2*chip.freq_GHz*r['array_busy_ns'])
        r['e2e_array_utilization']=float(r['useful_macs'])/(chip.sa_dim**2*chip.freq_GHz*r['time_ns'])
        answers.append(r)
        checkpoint=arch+'-'+str(bw)+'-'+mode+'-'+r['workload_id']
        save_json(state['out']/(checkpoint+'-workload.json'),dict(status='PASS',total=r))
        print('PASS workload',checkpoint,round(time.monotonic()-begun,1),flush=True)
    stem=arch+'-'+str(bw)+'-'+mode
    save_json(state['out']/(stem+'.json'),dict(status='PASS',job=list(job),totals=answers,
      elapsed_s=time.monotonic()-begun))
    if batches_report:save_csv(state['out']/(stem+'-batches.csv'),batches_report)
    return stem

def run_all(a):
    for name in ('programs','store'):
        assert json.loads((getattr(a,name)/'verification.json').read_text())['status']=='PASS'
    manifest=json.loads((a.costs/'manifest.json').read_text());assert manifest['status']=='PASS'
    configs=json.loads((a.costs/'configs.json').read_text());jobs=[]
    high=max(manifest['bandwidth_bytes_per_second'])
    for key in configs:
        arch,bw=key.split('@');bw=int(bw)
        if arch.endswith('-skew'):continue
        jobs.append((arch,bw,'region_async' if arch.startswith('Tessera') else 'serial'))
        if arch in ('Tessera-8','Tessera-32') and bw==high:jobs.append((arch,bw,'rounds'))
    source=[Path(__file__),Path(__file__).with_name('tessera_request_dispatch.py'),ROOT/'neusim/npusim/backend/tessera_partitioned.py',a.costs/'manifest.json',a.programs/'verification.json',a.store/'verification.json']
    hashes={str(p.resolve()):sha(p) for p in source}
    save_json(a.out/'scenario.json',dict(jobs=[list(j) for j in jobs],workers=a.workers,memory_gb=a.memory_gb,
      source_sha256=hashes,
      accounting='EDP = whole-chip NoPG energy including original arrival idle gaps * (last completion - first request arrival). Tensor-parallel energy includes all ranks.',
      admission='Source-order adjacent batches and chains only; preserve actual releases and autoregressive dependencies; no arbitrary future-job lookahead.',
      spatial_scope='Native SRAM-tile dispatch, with per-region final-fold retirement and lane reassignment at the next tile. FlashAttention keeps native QK/PV/head dependencies; shared HBM/VU/ICI FIFO service.',
      energy='All native traffic and padded compute/SRAM dynamic work retained; concurrent static energy charged once; fixed-voltage regulator efficiencies recomputed from concurrent mean activity. NoPG idle efficiency uses the native zero-activity point.'))
    completed=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize_run,initargs=tuple(str(getattr(a,k).resolve()) for k in ('costs','store','programs','out'))) as pool:
        futures=[pool.submit(run_job,job) for job in jobs]
        for future in as_completed(futures):
            stem=future.result()
            completed.append(stem);print('PASS replay',len(completed),'/',len(jobs),stem,round(time.monotonic()-started,1),flush=True)
    combined=[]
    for stem in completed:
        data=json.loads((a.out/(stem+'.json')).read_text());assert data['status']=='PASS';combined.extend(data['totals'])
    keys=list(dict.fromkeys(k for r in combined for k in r))
    combined=[{k:r.get(k,0) for k in keys} for r in combined]
    save_csv(a.out/'totals.csv',combined)
    for r in rows(a.out/'totals.csv'):
        assert math.isclose(float(r['edp_J_s']),float(r['system_energy_J'])*float(r['seconds']),rel_tol=2e-12)
    assert all(sha(Path(p))==h for p,h in hashes.items())
    save_json(a.out/'verification.json',dict(status='PASS',jobs=len(completed),records=len(combined),
      source_sha256=hashes,artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},elapsed_s=time.monotonic()-started))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['index','run']);p.add_argument('--costs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--store',type=Path);p.add_argument('--programs',type=Path);p.add_argument('--workers',type=int,default=1);p.add_argument('--memory-gb',type=float,default=1)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    assert 1<=a.workers<=64 and 0<a.memory_gb<=200
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers]);cap=int(a.memory_gb*1e9)//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
    if a.mode=='index':index_costs(a.costs,a.out)
    else:run_all(a)
if __name__=='__main__':main()
