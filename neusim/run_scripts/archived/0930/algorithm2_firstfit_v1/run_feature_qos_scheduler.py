"""Planaria allocation-policy replay over NeuSim profiles, identical on both sides.

Import allocation functions directly from the user's planaria.code. Event
bookkeeping handles simultaneous arrivals/completions and idle queues. At an
event every active task finishes its current native SRAM tile; all then wait for
the last tile boundary, matching the original scheduler's barrier semantics.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,ast,csv,contextlib,hashlib,importlib.util,io,json,math,random,resource,sys,time
from collections import Counter,OrderedDict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,PROJECT,rows,sha,save_csv,save_json
INPUT=ROOT/'results/tessera/20260929_feature_cnn_qos/inputs_v1'
SRC=PROJECT/'planaria.code/scheduler'

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
POLICY=load('original_planaria_policy',SRC/'scheduler.py')
GEN=load('original_planaria_generator',SRC/'generator.py')
# Parse only original scenario assignments: no FissionSA simulator import/run.
CONSTANTS={}
WANTED={'NET_CSVS','WORKLOADS','QOS_BASE_MS','QOS_LEVELS','NUM_TASKS','NUM_CORES','S_SEARCH','BISECT_ROUNDS','LAM_LO_MS','LAM_HI_MS','SLA_TARGET'}
for node in ast.parse((PROJECT/'FissionSA/multitenant/run_abc_qos.py').read_text()).body:
 if isinstance(node,ast.Assign):
  names={x.id for target in node.targets for x in ast.walk(target) if isinstance(x,ast.Name)}
  if names & WANTED:exec(compile(ast.Module(body=[node],type_ignores=[]),'sourced_scenario','exec'),CONSTANTS)
C={k:CONSTANTS[k] for k in WANTED}
FREQ=1e9 # source: inputs_v1/sources/configs.json, freq_GHz=1
INFO={}
# chenyi9: preserve actual fission grains under the same Planaria policy.
UNITS={'Planaria-32':16,'Tessera-8':256}

class Task:
 """Exact progress fractions with cached suffix times for original policy API."""
 def __init__(self,key,name,start,current,priority,sla,info):
  self.key=key;self.task_name=name;self.start_time=start;self.current_time=current;self.priority=priority;self.sla=sla
  self.info=info;self.layer=0;self.progress=Fraction(0);self.count=len(info[max(info)][0]);self.remaining_cache={}
 def remaining_tiles(self,c):
  n=self.info[c][0][self.layer][1];p=self.progress;return ((p.denominator-p.numerator)*n+p.denominator-1)//p.denominator
 def get_remaining_estimated_time(self,c):
  if c==0:return math.inf
  if c in self.remaining_cache:return self.remaining_cache[c]
  layers,suffix=self.info[c]
  if self.layer==self.count:return 0
  answer=self.remaining_tiles(c)*layers[self.layer][2]+suffix[self.layer+1]
  self.remaining_cache[c]=answer
  return answer
 def propose(self,dt,c):
  if c==0:return dt,self.layer,self.progress
  layers,_=self.info[c];i=self.layer;p=self.progress;spent=0.
  while i<len(layers):
   _,n,period,_=layers[i];left=math.ceil((1-p)*n);need=left*period
   if spent+need<=dt:
    spent+=need;i+=1;p=Fraction(0)
    if spent>=dt:break
   else:
    # Quantize to a tile boundary without adding a tile for float roundoff.
    q=(dt-spent)/period
    nearest=round(q)
    steps=nearest if math.isclose(q,nearest,rel_tol=2e-15,abs_tol=2e-15) else math.ceil(q)
    steps=min(left,max(1,steps));spent+=steps*period;p+=Fraction(steps,n)
    if p>=1:i+=1;p=Fraction(0)
    break
  return spent,i,p

def run(work,infos,qos,seed):
 random.seed(seed);queue=OrderedDict();done=[];cycle=0.;idx=0;events=0;allocations=0
 num_cores=max(next(iter(infos.values())))
 while idx<len(work) or queue:
  if not queue:cycle=max(cycle,work[idx][0])
  while idx<len(work) and work[idx][0]<=cycle:
   arrival,name,priority=work[idx];queue[idx]=Task(idx,name,arrival,cycle,priority,qos[name],infos[name]);idx+=1
  possible={key:POLICY.get_possible_num_cores(t,FREQ,num_cores) for key,t in queue.items()}
  if POLICY.check_if_tasks_all_fit(possible,num_cores):alloc=POLICY.assign_cores_if_tasks_fit(queue,possible,num_cores)
  else:alloc=POLICY.assign_cores_if_tasks_not_fit(queue,possible,num_cores,FREQ)
  assert set(alloc)==set(queue)
  assert 0<sum(alloc.values())<=num_cores,('original allocation has no progress',alloc)
  assert all(isinstance(v,int) and v>=0 for v in alloc.values());allocations+=1
  dt=min(t.get_remaining_estimated_time(alloc[key]) for key,t in queue.items())
  if idx<len(work):dt=min(dt,work[idx][0]-cycle)
  assert dt>0 and math.isfinite(dt)
  proposed={key:t.propose(dt,alloc[key]) for key,t in queue.items()}
  step=max(v[0] for v in proposed.values());assert step>0
  cycle+=step
  for key,t in list(queue.items()):
   _,new_layer,new_progress=proposed[key]
   if (new_layer,new_progress)!=(t.layer,t.progress):t.remaining_cache.clear()
   t.layer,t.progress=new_layer,new_progress;t.current_time=cycle
   if t.layer==t.count:done.append(dict(task_id=key,network=t.task_name,start_cycle=t.start_time,finish_cycle=cycle,latency_ms=(cycle-t.start_time)/FREQ*1e3,sla_ms=t.sla));del queue[key]
  events+=1;assert events<100000
 assert sorted(x['task_id'] for x in done)==list(range(len(work)))
 return done,events,allocations

def make_info(profile_path):
 by={(r['kind'],r['key'],r['architecture'],int(r['cores'])):r for r in rows(profile_path/'operator_costs.csv')}
 layers=list(rows(INPUT/'cnn_layers.csv'));result={}
 for arch in ('Planaria-32','Tessera-8'):
  info={}
  for name,stems in C['NET_CSVS'].items():
   info[name]={}
   for c in range(1,UNITS[arch]+1):
    ll=[]
    for stem in stems:
     for r in layers:
      if r['network']!=stem:continue
      keys=('B','M','N','K') if r['kind']=='matrix' else ('input_bytes','output_bytes','vector_ops')
      rr=by[r['kind'],'x'.join(r[k] for k in keys),arch,c]
      n=int(rr['tiles']);tm=float(rr['time_ns']);period=tm*FREQ/1e9/n
      assert math.isclose(n*period,tm*FREQ/1e9,rel_tol=2e-15)
      ll.append((stem+'/'+r['layer'],n,period,float(rr['energy_J'])))
    suffix=[0.]*(len(ll)+1)
    for i in reversed(range(len(ll))):suffix[i]=suffix[i+1]+ll[i][1]*ll[i][2]
    info[name][c]=(ll,suffix)
  result[arch]=info
 return result

def init(path):
 global INFO;INFO=make_info(Path(path))

def pool_job(job):
 wk,qk,arch,lam,seed=job;np.random.seed(seed)
 gen=GEN.WorkloadGenerator(distribution='poisson',args={'max_cycles':0,'arrival_time':lam,'frequency':FREQ,'N':C['NUM_TASKS']})
 for name in C['WORKLOADS'][wk]:gen.add_task_type(name)
 with contextlib.redirect_stdout(io.StringIO()):work=gen.generate()
 work=[(int(t),str(n),int(p)) for t,n,p in work]
 qos={n:C['QOS_BASE_MS'][n]*C['QOS_LEVELS'][qk] for n in C['WORKLOADS'][wk]}
 records,events,alloc=run(work,INFO[arch],qos,seed)
 count=Counter()
 for r in records:
  group='tr' if r['network']=='GNMT' else 'cd';count[group+'_n']+=1;count[group+'_ok']+=int(r['latency_ms']<=r['sla_ms'])
 return dict(workload=wk,qos=qk,architecture=arch,lambda_ms=lam,seed=seed,cd_n=count['cd_n'],cd_ok=count['cd_ok'],tr_n=count['tr_n'],tr_ok=count['tr_ok'],
  events=events,allocations_checked=alloc,arrival_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest(),
  completion_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())

def feasible(rr):
 return all(sum(r[g+'_n'] for r in rr)==0 or sum(r[g+'_ok'] for r in rr)/sum(r[g+'_n'] for r in rr)>=threshold for g,threshold in [('cd',C['SLA_TARGET']['RESNET-50']),('tr',C['SLA_TARGET']['GNMT'])])

def check():
 # Independent hand-computed single-task and separated-arrival golden cases.
 for duration in (1600,3200):
  info={'test':{c:([('layer',16,duration/c/16,0.)],[duration/c,0.]) for c in range(1,17)}}
  for w,gold in [([(0,'test',1)],[duration/16]), ([(0,'test',1),(10000,'test',1)],[duration/16,10000+duration/16])]:
   rr,_,_=run(w,info,{'test':1},123)
   assert [r['finish_cycle'] for r in rr]==gold
  rr,_,_=run([(0,'test',1),(0,'test',1)],info,{'test':1},123)
  assert [r['finish_cycle'] for r in rr]==[duration/8,duration/8]
  t=Task(0,'test',0,0,1,1,info['test']);queue=OrderedDict([(0,t)])
  possible={0:POLICY.get_possible_num_cores(t,FREQ,16)}
  assert possible[0]==list(range(16,0,-1));assert POLICY.assign_cores_if_tasks_fit(queue,possible,16)=={0:16}
 return dict(status='PASS',hand_golden_cases=6,original_allocation_golden_cases=4)

def main():
 p=argparse.ArgumentParser();p.add_argument('--profiles',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=15);p.add_argument('--check',action='store_true');a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers]);cap=90_000_000_000//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
 save_json(out/'checks.json',check())
 sources=[Path(__file__),SRC/'scheduler.py',SRC/'generator.py',PROJECT/'FissionSA/multitenant/run_abc_qos.py'];hashes={str(x):sha(x) for x in sources}
 save_json(out/'scenario.json',dict(constants=C,frequency_Hz=FREQ,policy_sha256=hashes,allocation_units=UNITS,resource_policy='Same total PEs: Planaria 16 x 32x32, Tessera 256 x 8x8; proportional SRAM and HBM BW; native minimum E2E latency mapping; tile-boundary preemption',arrival_policy='Original integer-ms Poisson interarrival samples, 50 tasks/pool; not exponential intervals'))
 if a.check:print('PASS scheduler unit checks');return
 assert json.loads((a.profiles/'verification.json').read_text())['status']=='PASS'
 allrows=[];cfgs=[(w,q,arch) for w in C['WORKLOADS'] for q in C['QOS_LEVELS'] for arch in ('Planaria-32','Tessera-8')];state={};t=time.monotonic()
 def evaluate(pool,jobs):
  rr=list(pool.map(pool_job,jobs,chunksize=1));allrows.extend(rr);return rr
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(a.profiles.resolve()),)) as pool:
  rr=evaluate(pool,[(*cfg,C['LAM_HI_MS'],5000+i) for cfg in cfgs for i in range(C['S_SEARCH'])])
  for cfg in cfgs:
   subset=[r for r in rr if (r['workload'],r['qos'],r['architecture'])==cfg]
   state[cfg]=dict(lo=C['LAM_LO_MS'],hi=C['LAM_HI_MS'],attainable=feasible(subset),capped=False)
  print('Feasibility pass',round(time.monotonic()-t,1),flush=True)
  # Check the maximum offered-load endpoint explicitly; label capped results.
  live=[c for c in cfgs if state[c]['attainable']]
  rr=evaluate(pool,[(*c,C['LAM_LO_MS'],5500+i) for c in live for i in range(C['S_SEARCH'])])
  for c in live:
   if feasible([r for r in rr if (r['workload'],r['qos'],r['architecture'])==c]):state[c].update(hi=C['LAM_LO_MS'],capped=True)
  for rnd in range(C['BISECT_ROUNDS']):
   live=[c for c in cfgs if state[c]['attainable'] and not state[c]['capped']]
   jj=[]
   for c in live:
    state[c]['mid']=math.sqrt(state[c]['lo']*state[c]['hi']);jj.extend([(*c,state[c]['mid'],6000+rnd*100+i) for i in range(C['S_SEARCH'])])
   rr=evaluate(pool,jj)
   for c in live:
    good=feasible([r for r in rr if (r['workload'],r['qos'],r['architecture'])==c]);state[c]['hi' if good else 'lo']=state[c]['mid']
   print('Bisection',rnd+1,'seconds',round(time.monotonic()-t,1),flush=True)
 save_csv(out/'pool_results.csv',allrows)
 summary=[dict(workload=c[0],qos=c[1],architecture=c[2],qps=1000/s['hi'] if s['attainable'] else 0,lambda_feasible_ms=s['hi'],lambda_infeasible_ms=s['lo'],search_capped=s['capped'],sla_attainable=s['attainable']) for c,s in state.items()]
 save_csv(out/'qps.csv',summary)
 # Identical seeds at identical offered load must generate identical pools.
 identities={}
 for r in allrows:
  k=(r['workload'],r['qos'],r['lambda_ms'],r['seed'])
  if k in identities:assert identities[k]==r['arrival_sha256']
  identities[k]=r['arrival_sha256']
 assert all(sha(Path(x))==h for x,h in hashes.items())
 save_json(out/'verification.json',dict(status='PASS',pools=len(allrows),tasks_completed=len(allrows)*C['NUM_TASKS'],allocation_checks=sum(r['allocations_checked'] for r in allrows),source_sha256=hashes,profiles_sha256=sha(a.profiles/'operator_costs.csv'),artifacts={n:sha(out/n) for n in ('qps.csv','pool_results.csv','scenario.json')},elapsed_s=time.monotonic()-t))
 print('PASS QoS',len(allrows),'pools',flush=True)
if __name__=='__main__':main()
