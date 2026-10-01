"""Planaria allocation-policy replay over NeuSim profiles, identical on both sides.

Import allocation functions directly from the user's planaria.code. Event
bookkeeping handles simultaneous arrivals/completions and idle queues. The
default packs physical FCRs within synchronous rounds. Optional region_async
dispatch keeps the same allocation policy and admits independent tasks without
a global barrier. The original barrier replay remains explicit. It uses native NeuSim
component-overlap accounting, not a cycle-accurate operand-ready pipeline.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,ast,csv,contextlib,hashlib,heapq,importlib.util,io,json,math,random,resource,sys,time
from collections import Counter,OrderedDict
from concurrent.futures import ProcessPoolExecutor,as_completed
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
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
REGION_PROFILES={}
REGION_CHIP={}
DISPATCH='rounds'
PRUNE_INFEASIBLE=False
# chenyi9: preserve actual fission grains under the same Planaria policy.
UNITS={'Planaria-32':16,'Tessera-8':256}

@lru_cache(maxsize=65536)
def mapping_from_json(value):return json.loads(value)

@lru_cache(maxsize=65536)
def available_boxes(occupied,side,grain,h,w,limit):
 from neusim.npusim.backend.tessera_partitioned import FirstFitFabric
 fabric=FirstFitFabric(side,grain);fabric.rows=list(occupied)
 return tuple(fabric.available(h,w,limit))

class ProvenInfeasible(Exception):
 def __init__(self,done,events,allocations,late,cutoff):
  self.done=done;self.events=events;self.allocations=allocations;self.late=dict(late);self.cutoff=cutoff

def failure_budgets():
 # Exact upper bound: each point has S_SEARCH pools of NUM_TASKS requests.
 # Even if every other request succeeds, more than this many known failures
 # makes the original aggregate SLA target impossible. No completion estimate
 # or simulated latency is substituted for an observed deadline miss.
 population=C['S_SEARCH']*C['NUM_TASKS']
 return {g:math.floor(population*(1-Fraction(str(C['SLA_TARGET'][name]))))
   for g,name in [('cd','RESNET-50'),('tr','GNMT')]}

def possible_cores(task,frequency,total):
 # Codex: decision start — cache only a provably unchanged original-policy result.
 # Waiting-task estimates are constant until its layer/progress changes. The
 # exact original slack remains within [max accepted, min rejected), so every
 # original <= comparison has the same answer.
 flight=getattr(task,'inflight',None)
 if flight is not None:
  state=(task.layer,task.progress,flight,total,frequency,task.sla,task.start_time)
  saved=getattr(task,'inflight_possible_cache',None)
  finish=flight[0]
  if saved is not None and saved[0]==state and task.start_time<=task.current_time<=finish:
   return list(saved[1])
  result=POLICY.get_possible_num_cores(task,frequency,total)
  if not 0<=task.start_time<=task.current_time<=finish:return result
  operands=[task.remaining_cache[c] for c in range(1,total+1)]
  sla=task.sla*1.e-3*frequency
  # Each source predicate is ((finish-now)+first)+second <= sla-(now-start).
  # Its exact-real difference is independent of now. Eight times the largest
  # nonnegative operand bounds every intermediate magnitude. Five source
  # rounded operations plus fsum's rounding contribute less than six ULPs
  # at that bound. Cache only when every sign is outside this conservative
  # error interval; near ties always execute the original source function.
  magnitude=8*max(finish,task.start_time,sla,*(v for pair in operands for v in pair))
  uncertainty=6*math.ulp(magnitude)
  differences=[math.fsum((finish,first,second,-sla,-task.start_time)) for first,second in operands]
  if all(abs(d)>uncertainty for d in differences):
   assert result==[c for c in range(total,0,-1) if differences[c-1]<0]
   task.inflight_possible_cache=(state,tuple(result))
  return result
 state=(task.layer,task.progress,total)
 slack=task.sla*1.e-3*frequency-(task.current_time-task.start_time)
 saved=getattr(task,'possible_cache',None)
 if saved is not None and saved[0]==state and saved[1]<=slack<saved[2]:return list(saved[3])
 result=POLICY.get_possible_num_cores(task,frequency,total)
 accepted=set(result)
 lo=max((task.get_remaining_estimated_time(c) for c in accepted),default=-math.inf)
 hi=min((task.get_remaining_estimated_time(c) for c in range(1,total+1) if c not in accepted),default=math.inf)
 task.possible_cache=(state,lo,hi,tuple(result))
 return result
 # Codex: decision end

class Task:
 """Exact progress fractions with cached suffix times for original policy API."""
 def __init__(self,key,name,start,current,priority,sla,info):
  self.key=key;self.task_name=name;self.start_time=start;self.current_time=current;self.priority=priority;self.sla=sla
  self.info=info;self.layer=0;self.progress=Fraction(0);self.count=len(info[max(info)][0]);self.remaining_cache={}
 def remaining_tiles(self,c):
  n=self.info[c][0][self.layer][1];p=self.progress;return ((p.denominator-p.numerator)*n+p.denominator-1)//p.denominator
 def get_remaining_estimated_time(self,c):
  if c==0:return math.inf
  layers,suffix=self.info[c]
  if self.layer==self.count:return 0
  # Codex: decision start — credit in-flight progress at asynchronous events.
  # A pinned group has an already scheduled finish, not another full tile left.
  flight=getattr(self,'inflight',None)
  if flight is not None:
   finish,groups=flight
   # Cache only time-invariant operands; retain the original operation order.
   if c not in self.remaining_cache:
    pending=max(Fraction(0),1-self.progress-Fraction(1,groups))
    self.remaining_cache[c]=(math.ceil(pending*layers[self.layer][1])*layers[self.layer][2],suffix[self.layer+1])
   first,second=self.remaining_cache[c]
   return max(0,finish-self.current_time)+first+second
  if c in self.remaining_cache:return self.remaining_cache[c]
  answer=self.remaining_tiles(c)*layers[self.layer][2]+suffix[self.layer+1]
  # Codex: decision end
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

# chenyi9: decision start — asynchronous backfill is separate from the round fix.
def run_regions(work,infos,qos,seed,profiles,chip,trace=None,asynchronous=True,fail_budget=None):
 """Nonpreemptive FCR chains with original Planaria allocation targets.

 A GEMM chooses one profile when admitted; its SRAM tile stays fixed for that
 layer, avoiding fractional progress being remapped into different real tiles.
 Each SRAM tile packs K-major/N-minor weights into homogeneous lane chains.
 Shorter chains retire independently. Work ready at a round's start can fill
 its partial tail. With asynchronous=True, later arrivals can also use holes;
 otherwise they wait for the current ready-work cohort to finish.
 Native HBM/VU service and the selected profile's component floors overlap SA
 execution; a shared FIFO server additionally prevents overbooking HBM/VU.
 This is the native roofline abstraction, NOT proof that operands are ready
 at every physical cycle. Layer dependencies wait for all component service.
 """
 from neusim.npusim.backend.tessera_partitioned import FirstFitFabric,packed_lanes,lane_cycles
 from neusim.npusim.backend.tessera import ceildiv
 random.seed(seed)
 side,grain=chip['side'],chip['grain'];units=(side//grain)**2
 fabric=FirstFitFabric(side,grain)
 queue=OrderedDict();state={};done=[];pending=[];now=0.;idx=0;serial=0;events=0;allocations=0
 hbm_end=vu_end=0.
 round_members=set();round_waiting=set()
 def log(event,**kw):
  if trace is not None:trace.append(dict(event=event,cycle=now,**kw))
 def push(when,kind,key,payload):
  nonlocal serial
  serial+=1;heapq.heappush(pending,(when,serial,kind,key,payload))
 def used_memory():return sum(s['held'] for s in state.values())
 def get_profile(key,c):return profiles[queue[key].task_name][c][queue[key].layer]
 @lru_cache(maxsize=65536)
 def feasible_profiles(name,layer,top,available_memory):
  return tuple(profiles[name][c][layer] for c in range(top,0,-1)
    if int(profiles[name][c][layer]['peak_live_bytes'])<=available_memory)
 def prepare(key,row,target,borrow):
  """Probe both geometry and live SRAM before committing any state."""
  s=state[key];mapping=mapping_from_json(row['mapping_json'])
  if row['kind']=='vector':
   return dict(row=row,lanes=[],peak=int(row['peak_live_bytes']),count=1,ktiles=1,accum=0,
    hbm=int(row['hbm_bytes']),vu=float(row['vu_ns'])*FREQ/1e9,
    floor=float(row['time_ns'])*FREQ/1e9,weight_tiles=0)
  b,m,n,k=map(int,row['key'].split('x'));mt,nt,kt=map(int,mapping['memory_tile'])
  if any(v%t for v,t in zip((m,n,k),(mt,nt,kt))):
   raise ValueError('Region dispatcher requires the native divisor SRAM tiles')
  g=tuple(mapping['geometry']);h,w=g[-2:];nk,nn=ceildiv(kt,g[0]),ceildiv(nt,g[1])
  need=h*w//grain**2
  limit=side*side//(h*w) if borrow else target//need
  if limit==0:return None
  boxes=available_boxes(tuple(fabric.rows),side,grain,h,w,min(limit,nk*nn,g[3]*nn))
  if not boxes:return None
  width=packed_lanes(nk,nn,len(boxes),g[3])
  weights=nk*nn
  lanes=[(box,lane_cycles(mt,g,'full',grain,(weights+width-1-i)//width)) for i,box in enumerate(boxes[:width])]
  ktiles=k//kt
  group=s['completed']
  # Native output-stationary traffic: each A/B tile is fetched, final C only
  # after the last K tile. Running C remains reserved while a task waits.
  hbm=2*(mt*kt+kt*nt)+(2*mt*nt if group%ktiles==ktiles-1 else 0)
  if int(row['hbm_bytes'])==0:hbm=0  # explicitly resident validation inputs
  red=mt*nt*(nk-1+(group%ktiles!=0))
  vu=float(row['vu_ns'])*red/int(row['reduction_ops'])*FREQ/1e9 if int(row['reduction_ops']) else 0.
  floor=float(row['hbm_ns'])*hbm/int(row['hbm_bytes'])*FREQ/1e9 if int(row['hbm_bytes']) else 0.
  return dict(row=row,lanes=lanes,peak=int(row['peak_live_bytes']),count=b*(m//mt)*(n//nt)*ktiles,
   ktiles=ktiles,accum=4*mt*nt,hbm=hbm,vu=vu,floor=floor,weight_tiles=weights)
 def admit(key,target,borrow=False):
  nonlocal hbm_end,vu_end
  s=state[key]
  if s['active']:return False
  available_memory=chip['capacity_bytes']-used_memory()+s['held']
  if s['row'] is not None:
   choices=[s['row']]
  else:
   # Original allocation is a target, not permission to relocate live FCRs.
   # Search smaller feasible shares only when its selected map cannot fit.
   top=units if borrow else target
   choices=feasible_profiles(queue[key].task_name,queue[key].layer,top,available_memory)
  plan=None
  for row in choices:
   # The profile already records the exact SRAM reservation. Reject an
   # impossible reservation before computing identical geometry/traffic work.
   if int(row['peak_live_bytes'])>available_memory:continue
   candidate=prepare(key,row,target,borrow)
   if candidate is not None and candidate['peak']<=available_memory:
    plan=candidate;break
  if plan is None:return False
  s['row']=plan['row'];s['active']=True;s['held']=plan['peak']
  group=s['completed'];finish=now+plan['floor'];coalesced={}
  for lane,(box,cycles) in enumerate(plan['lanes']):
   token=(key,queue[key].layer,group,lane)
   release=now+cycles*FREQ/chip['frequency_Hz']
   if trace is None:
    masks=coalesced.setdefault(release,[0]*len(fabric.rows));r,c,h,w=box
    mask=((1<<(w//grain))-1)<<(c//grain)
    for y in range(r//grain,(r+h)//grain):
     assert not fabric.rows[y]&mask;fabric.rows[y]|=mask;masks[y]|=mask
   else:
    actual=fabric.place(token,*box[-2:]);assert actual==box
    push(release,'region_done',key,token)
   finish=max(finish,release)
   log('region_start',task_id=key,layer=queue[key].layer,group=group,lane=lane,
       box=list(box),finish_cycle=release,borrowed=borrow,
       first_weight=lane,weight_stride=len(plan['lanes']),
       weight_count=(plan['weight_tiles']+len(plan['lanes'])-1-lane)//len(plan['lanes']))
  for i,(release,masks) in enumerate(coalesced.items()):
   token=('coalesced',key,queue[key].layer,group,i);fabric.regions[token]=masks
   push(release,'region_done',key,token)
  if plan['hbm']:
   begin=max(now,hbm_end);hbm_end=begin+plan['hbm']/chip['hbm_bytes_per_cycle'];finish=max(finish,hbm_end)
   log('hbm_service',task_id=key,start_cycle=begin,finish_cycle=hbm_end,bytes=plan['hbm'])
  if plan['vu']:
   begin=max(now,vu_end);vu_end=begin+plan['vu'];finish=max(finish,vu_end)
   log('vu_service',task_id=key,start_cycle=begin,finish_cycle=vu_end)
  assert finish>now and used_memory()<=chip['capacity_bytes']
  queue[key].inflight=(finish,plan['count']);queue[key].remaining_cache.clear()
  push(finish,'group_done',key,plan)
  log('group_start',task_id=key,layer=queue[key].layer,group=group,finish_cycle=finish,
      held_bytes=used_memory(),weight_tiles=plan['weight_tiles'],borrowed=borrow)
  return True
 while idx<len(work) or queue:
  if not queue and not pending:now=max(now,work[idx][0])
  while pending and pending[0][0]<=now:
   _,_,kind,key,payload=heapq.heappop(pending)
   if kind=='region_done':
    if payload[0]=='coalesced':
     masks=fabric.regions.pop(payload)
     for y,mask in enumerate(masks):assert fabric.rows[y]&mask==mask;fabric.rows[y]^=mask
    else:
     box=fabric.regions[payload];fabric.release(payload);log(kind,task_id=key,box=list(box))
   else:
    t=queue[key];s=state[key];s['active']=False;s['completed']+=1;t.inflight=None
    round_members.discard(key)
    t.progress=Fraction(s['completed'],payload['count']);t.remaining_cache.clear()
    s['held']=payload['accum'] if s['completed']%payload['ktiles'] else 0
    log(kind,task_id=key,layer=t.layer,group=s['completed']-1)
    if s['completed']==payload['count']:
     t.layer+=1;t.progress=Fraction(0);s.update(row=None,completed=0,held=0)
    if t.layer==t.count:
     done.append(dict(task_id=key,network=t.task_name,start_cycle=t.start_time,finish_cycle=now,
       latency_ms=(now-t.start_time)/FREQ*1e3,sla_ms=t.sla))
     del queue[key];del state[key];log('task_done',task_id=key)
  while idx<len(work) and work[idx][0]<=now:
   arrival,name,priority=work[idx];queue[idx]=Task(idx,name,arrival,now,priority,qos[name],infos[name])
   state[idx]=dict(active=False,held=0,row=None,completed=0);log('arrival',task_id=idx);idx+=1
  if fail_budget is not None:
   late=Counter()
   for record in done:
    if record['latency_ms']>record['sla_ms']:late['tr' if record['network']=='GNMT' else 'cd']+=1
   for task in queue.values():
    if (now-task.start_time)/FREQ*1e3>task.sla:late['tr' if task.task_name=='GNMT' else 'cd']+=1
   if any(late[g]>limit for g,limit in fail_budget.items()):
    raise ProvenInfeasible(done,events,allocations,late,now)
  if queue:
   if not asynchronous and not round_members:
    round_members=set(queue);round_waiting=set(queue)
    log('round_open',members=list(queue))
   for t in queue.values():t.current_time=now
   admitted=queue if asynchronous else OrderedDict((key,t) for key,t in queue.items() if key in round_members)
   possible={key:possible_cores(t,FREQ,units) for key,t in admitted.items()}
   if POLICY.check_if_tasks_all_fit(possible,units):alloc=POLICY.assign_cores_if_tasks_fit(admitted,possible,units)
   else:alloc=POLICY.assign_cores_if_tasks_not_fit(admitted,possible,units,FREQ)
   assert set(alloc)==set(admitted) and 0<sum(alloc.values())<=units
   allocations+=1;log('allocation',targets=dict(alloc))
   eligible=list(queue) if asynchronous else [key for key in queue if key in round_waiting]
   for key in eligible:
    if alloc[key]>0 and admit(key,alloc[key]):round_waiting.discard(key)
   # User-requested extra behavior: fill remaining legal holes with ready
   # independent work. Running regions are pinned until their own completion.
   for key in eligible:
    if admit(key,alloc[key],borrow=True):round_waiting.discard(key)
  future=[pending[0][0]] if pending else []
  if idx<len(work):future.append(work[idx][0])
  if not future:
   if queue:raise RuntimeError('No feasible region/SRAM placement makes progress')
   break
  next_event=min(future);assert next_event>now;now=next_event;events+=1
 assert not fabric.regions and not any(fabric.rows)
 assert sorted(x['task_id'] for x in done)==list(range(len(work)))
 return done,events,allocations


def make_region_profiles(profile_path):
 """Load real maps and full-device resources; reject obsolete timing tables."""
 from neusim.npusim.backend.tessera_partitioned import PACKING_POLICY
 allrows=list(rows(profile_path/'operator_costs.csv'))
 for r in allrows:
  if r['architecture']=='Tessera-8' and r.get('packing_policy')!=PACKING_POLICY:
   raise ValueError('Tessera profiles predate the Algorithm 2 packing fix')
 by={(r['kind'],r['key'],int(r['cores'])):r for r in allrows if r['architecture']=='Tessera-8'}
 layers=list(rows(INPUT/'cnn_layers.csv'));result={}
 for name,stems in C['NET_CSVS'].items():
  result[name]={}
  for c in range(1,UNITS['Tessera-8']+1):
   ll=[]
   for stem in stems:
    for r in layers:
     if r['network']!=stem:continue
     keys=('B','M','N','K') if r['kind']=='matrix' else ('input_bytes','output_bytes','vector_ops')
     ll.append(by[r['kind'],'x'.join(r[k] for k in keys),c])
   result[name][c]=ll
 configs=json.loads((profile_path/'configs.json').read_text())
 c=configs['Tessera-8/'+str(UNITS['Tessera-8'])]
 return result,dict(side=c['sa_dim'],grain=c['tessera_parameters']['grain'],
  capacity_bytes=c['vmem_size_MB']*1024**2,frequency_Hz=c['freq_GHz']*1e9,
  hbm_bytes_per_cycle=c['hbm_bw_GBps']*1024**3/FREQ)
# chenyi9: decision end

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

# Codex: decision start — repair QoS using exact unfinished output tiles on both architectures.
# Source: tessera_joint.factor_tiles/candidate_vectors, native output-stationary
# SRAM accounting, and the original Planaria allocation functions imported above.
# A partial C tile is never converted to a fraction of a different mapping.
@dataclass(frozen=True)
class TileProfile:
 kind: str
 shape: tuple
 geometry: tuple
 tile: tuple
 resident: bool
 vu_per_add: float
 vector_hbm: int=0
 vector_vu: float=0.
 vector_peak: int=0

def tile_profile(row):
 if row['kind']=='vector':
  return TileProfile('vector',(),(),(),False,0.,int(row['hbm_bytes']),float(row['vu_ns'])*FREQ/1e9,int(row['peak_live_bytes']))
 mp=mapping_from_json(row['mapping_json']);red=int(row['reduction_ops'])
 return TileProfile('matrix',tuple(map(int,row['key'].split('x'))),tuple(mp['geometry']),tuple(mp['memory_tile']),
  int(row['hbm_bytes'])==0,float(row['vu_ns'])*FREQ/1e9/red if red else 0.)

class MatchedCosts:
 """Common tile timing, resource limits and exact remaining-work estimates.

 Future execution estimates assume the requested PE share is available. They
 include the actual pinned partial C tile, but do not predict future arrivals
 or FIFO interference. In-flight service has its already scheduled finish.
 """
 def __init__(self,profiles,chip,architecture):
  self.chip=chip;self.arch=architecture;self.units=(chip['side']//chip['grain'])**2
  self.profiles={n:{c:tuple(tile_profile(r) for r in ll) for c,ll in pp.items()} for n,pp in profiles.items()}
  self.suffix={}
  for name,pp in self.profiles.items():
   self.suffix[name]={}
   for c,ll in pp.items():
    suffix=[0.]*(len(ll)+1)
    for i in reversed(range(len(ll))):
     p=ll[i]
     duration=self.vector_cost(p,c)[0] if p.kind=='vector' else self.rectangle(p,p.shape[1],p.shape[2],c)*p.shape[0]
     suffix[i]=max(duration,0 if p.resident else chip['hbm_latency_cycles'])+suffix[i+1]
    self.suffix[name][c]=suffix

 @lru_cache(maxsize=262144)
 def group(self,p,m,n,k,first,last,c,slots_override=None):
  """One SRAM K tile; traffic and padding use its real dimensions."""
  from neusim.npusim.backend.tessera_partitioned import lane_cycles
  from neusim.npusim.backend.tessera import ceildiv
  g=p.geometry;h,w=g[-2:];need=h*w//self.chip['grain']**2
  slots=c//need if slots_override is None else slots_override
  nk,nn=ceildiv(k,g[0]),ceildiv(n,g[1]);width=min(slots,nk*nn,g[3]*nn)
  if width<=0:return None
  folds=ceildiv(nk*nn,width)
  if self.arch=='Planaria-32':
   cycles=folds*(max(m,32) if folds>=3 else m)+32+h+w-2
  else:cycles=lane_cycles(m,g,'full',self.chip['grain'],folds)
  sa=cycles*FREQ/self.chip['frequency_Hz']
  hb=0 if p.resident else 2*(m*k+k*n)+(2*m*n if last else 0)
  vu=m*n*(nk-1+(not first))*p.vu_per_add
  peak=4*(m*k+k*n)+4*(g[3]+1)*m*n
  duration=max(sa,vu,hb/self.chip['hbm_bytes_per_cycle']*self.units/c)
  return duration,sa,hb,vu,peak,width,nk*nn

 @lru_cache(maxsize=65536)
 def output_tile(self,p,m,n,k0,c):
  """Same group sequence as execution, summed without a tile loop."""
  k=p.shape[3];kt=p.tile[2];full,tail=divmod(k-k0,kt);total=0.
  groups=[]
  if full:
   # Only the first and final K groups have different reduction/output work.
   if full==1:groups.append((kt,k0==0,tail==0,1))
   else:
    groups.append((kt,k0==0,False,1))
    if full>2:groups.append((kt,False,False,full-2))
    groups.append((kt,False,tail==0,1))
  if tail:groups.append((tail,k0==0 and not full,True,1))
  for kk,first,last,count in groups:
   v=self.group(p,m,n,kk,first,last,c)
   if v is None or v[4]>self.chip['capacity_bytes']:return math.inf
   total+=v[0]*count
  return total

 @lru_cache(maxsize=131072)
 def rectangle(self,p,m,n,c):
  mt,nt,_=p.tile;total=0.
  mm=[(mt,m//mt),(m%mt,1)]
  nn=[(nt,n//nt),(n%nt,1)]
  for h,hcount in mm:
   for w,wcount in nn:
    if h and w and hcount and wcount:total+=hcount*wcount*self.output_tile(p,h,w,0,c)
  return total

 @lru_cache(maxsize=8192)
 def vector_cost(self,p,c):
  return max(p.vector_vu,p.vector_hbm/self.chip['hbm_bytes_per_cycle']*self.units/c),p.vector_hbm,p.vector_vu,p.vector_peak

class MatchedTask:
 """Uncomputed B/M/N rectangles plus at most one partially accumulated C tile."""
 def __init__(self,key,name,start,current,priority,sla,model):
  self.key=key;self.task_name=name;self.start_time=start;self.current_time=current
  self.priority=priority;self.sla=sla;self.model=model;self.layer=0
  self.count=len(model.profiles[name][model.units]);self.held=0;self.inflight=None
  self.version=0;self.cache={};self.possible_saved=None;self.completed_macs=0;self.groups=0
  self.reset_layer()
 def reset_layer(self):
  self.partial=None;self.origin=None;self.rectangles=[];self.vector_pending=False
  if self.layer<self.count:
   p=self.model.profiles[self.task_name][self.model.units][self.layer]
   if p.kind=='matrix':
    b,m,n,k=p.shape;self.rectangles=[(0,b,0,m,0,n)]
   else:self.vector_pending=True
 def changed(self):
  self.version+=1;self.cache.clear();self.possible_saved=None
 def remaining_work(self,c):
  if c==0:return math.inf
  if c in self.cache:return self.cache[c]
  if self.layer==self.count:return 0.
  model=self.model;p=model.profiles[self.task_name][c][self.layer]
  total=0.
  if self.vector_pending:total=model.vector_cost(p,c)[0]
  elif p.kind=='matrix':
   if self.partial is not None:
    pp,b,m,n,h,w,k0=self.partial
    total+=model.output_tile(pp,h,w,k0,c)
   for b,bc,m,h,n,w in self.rectangles:total+=bc*model.rectangle(p,h,w,c)
  self.cache[c]=total
  return total
 def get_remaining_estimated_time(self,c):
  if c==0:return math.inf
  if self.layer==self.count:return 0.
  wait=max(0.,self.inflight-self.current_time) if self.inflight is not None else 0.
  current=wait+self.remaining_work(c)
  p=self.model.profiles[self.task_name][c][self.layer]
  if not p.resident:
   floor=self.model.chip['hbm_latency_cycles']
   current=max(current,floor if self.origin is None else max(0.,self.origin+floor-self.current_time))
  return current+self.model.suffix[self.task_name][c][self.layer+1]
 def possible(self):
  # Codex: decision start — reuse only predicates provably unchanged until retirement.
  # For now < finish, the two deadline predicates reduce to finish+work+suffix
  # and origin+HBM_latency+suffix <= arrival+SLA. The original function is the
  # golden. Reject caching near an IEEE-754 boundary; sixteen ulps bound the
  # rounded sums/subtractions here (fewer than eight, including comparison slack).
  if self.inflight is not None and self.origin is not None and self.current_time<self.inflight:
   saved=getattr(self,'flight_possible_saved',None)
   if saved is not None and saved[0]==self.version:return list(saved[1])
   result=POLICY.get_possible_num_cores(self,FREQ,self.model.units)
   sla=self.sla*1e-3*FREQ;stable=True;accepted=set(result)
   for c in range(1,self.model.units+1):
    work=self.remaining_work(c);suffix=self.model.suffix[self.task_name][c][self.layer+1]
    if not math.isfinite(work):
     assert c not in accepted
     continue
    p=self.model.profiles[self.task_name][c][self.layer]
    floor=self.origin+(0 if p.resident else self.model.chip['hbm_latency_cycles'])
    parts=(self.inflight,work,suffix,-self.start_time,-sla)
    margins=(math.fsum(parts),math.fsum((floor,suffix,-self.start_time,-sla)))
    uncertainty=16*math.ulp(max(1.,math.fsum(abs(x) for x in parts)+abs(floor)))
    if any(abs(x)<=uncertainty for x in margins):stable=False;break
    assert (c in accepted)==all(x<0 for x in margins)
   if stable:self.flight_possible_saved=(self.version,tuple(result))
   return result
  # Codex: decision end
  # Waiting estimates decrease only through the native per-layer latency floor.
  # Cache the original predicate only when its conservative interval is unchanged.
  slack=self.sla*1e-3*FREQ-(self.current_time-self.start_time)
  saved=self.possible_saved
  if saved is not None and saved[0]==self.version and saved[1]<=slack<saved[2]:
   return list(saved[3])
  result=POLICY.get_possible_num_cores(self,FREQ,self.model.units)
  # Only cache when there is no time-dependent in-flight/floor contribution.
  if self.inflight is None and (self.origin is None or self.current_time>=self.origin+self.model.chip['hbm_latency_cycles']):
   accepted=set(result)
   lo=max((self.get_remaining_estimated_time(c) for c in accepted),default=-math.inf)
   hi=min((self.get_remaining_estimated_time(c) for c in range(1,self.model.units+1) if c not in accepted),default=math.inf)
   self.possible_saved=(self.version,lo,hi,tuple(result))
  return result

def run_matched(work,model,qos,seed,asynchronous=False,trace=None,fail_budget=None,allocation_policy=None):
 """Shared resource/event model; Planaria retains synchronous dispatch rounds."""
 from neusim.npusim.backend.tessera_partitioned import FirstFitFabric,lane_cycles
 from neusim.npusim.backend.tessera import ceildiv
 random.seed(seed);chip=model.chip;units=model.units;grain=chip['grain'];side=chip['side']
 tessera=model.arch=='Tessera-8';fabric=FirstFitFabric(side,grain) if tessera else None
 occupied_cores=0;queue=OrderedDict();done=[];pending=[];now=0.;idx=0;serial=0;events=0;allocations=0
 hbm_end=vu_end=0.;cohort=set();waiting=set();running=set()
 def log(event,**kw):
  if trace is not None:trace.append(dict(event=event,cycle=now,**kw))
 def push(when,kind,key,payload):
  nonlocal serial
  serial+=1;heapq.heappush(pending,(when,serial,kind,key,payload))
 def used_memory():return sum(t.held for t in queue.values())
 def probe(t,c,borrow):
  available_memory=chip['capacity_bytes']-used_memory()+t.held
  if t.partial:
   choices=[t.partial[0]]
  else:
   pp=model.profiles[t.task_name]
   top=units if borrow else c
   choices=dict.fromkeys(pp[x][t.layer] for x in range(top,0,-1))
  for p in choices:
   if p.kind=='vector':
    duration,hb,vu,peak=model.vector_cost(p,max(1,c))
    if peak<=available_memory:return dict(p=p,vector=True,peak=peak,duration=duration,hbm=hb,vu=vu,lanes=[],share=max(1,c))
    continue
   if t.partial:pp,b,m,n,h,w,k0=t.partial
   else:
    b,bc,m,h0,n,w0=t.rectangles[0];h=min(h0,p.tile[0]);w=min(w0,p.tile[1]);k0=0
   k=min(p.tile[2],p.shape[3]-k0);g=p.geometry;gh,gw=g[-2:];need=gh*gw//grain**2
   requested=units//need if borrow else c//need
   if requested<=0:continue
   max_width=min(requested,ceildiv(k,g[0])*ceildiv(w,g[1]),g[3]*ceildiv(w,g[1]))
   if tessera:
    boxes=available_boxes(tuple(fabric.rows),side,grain,gh,gw,max_width)
    width=len(boxes)
   else:
    width=min(max_width,(units-occupied_cores)//need);boxes=[None]*width
   if width<=0:continue
   share=max(c,width*need) if borrow else c
   v=model.group(p,h,w,k,k0==0,k0+k==p.shape[3],max(1,share),width)
   if v[4]>available_memory:continue
   duration,sa,hb,vu,peak,width,weights=v
   lanes=[]
   for i,box in enumerate(boxes):
    folds=ceildiv(weights-i,width)
    cycles=(folds*(max(h,32) if folds>=3 else h)+32+gh+gw-2) if not tessera else lane_cycles(h,g,'full',grain,folds)
    lanes.append((box,cycles*FREQ/chip['frequency_Hz'],need))
   return dict(p=p,vector=False,b=b,m=m,n=n,h=h,w=w,k0=k0,k=k,peak=peak,
    duration=duration,hbm=hb,vu=vu,lanes=lanes,share=share,weights=weights)
  return None
 def admit(key,target,borrow=False):
  nonlocal occupied_cores,hbm_end,vu_end
  t=queue[key]
  if t.inflight is not None or (target<=0 and not borrow):return False
  plan=probe(t,target,borrow)
  if plan is None:return False
  p=plan['p'];t.held=plan['peak'];t.groups+=1
  if t.origin is None:t.origin=now
  finish=now+plan['duration'];coalesced={}
  for i,(box,cycles,need) in enumerate(plan['lanes']):
   release=now+cycles
   if tessera:
    masks=coalesced.setdefault(release,[0]*len(fabric.rows));r,c,h,w=box
    mask=((1<<(w//grain))-1)<<(c//grain)
    for y in range(r//grain,(r+h)//grain):assert not fabric.rows[y]&mask;fabric.rows[y]|=mask;masks[y]|=mask
   else:
    occupied_cores+=need;coalesced[release]=coalesced.get(release,0)+need
   log('region_start',task_id=key,layer=t.layer,group=t.groups,box=box,finish_cycle=release,borrowed=borrow,
       first_weight=i,weight_stride=len(plan['lanes']),weight_count=ceildiv(plan['weights']-i,len(plan['lanes'])))
  for release,mask in coalesced.items():push(release,'release',key,mask)
  if plan['hbm']:
   begin=max(now,hbm_end);hbm_end=begin+plan['hbm']/chip['hbm_bytes_per_cycle'];finish=max(finish,hbm_end)
   log('hbm_service',task_id=key,start_cycle=begin,finish_cycle=hbm_end,bytes=plan['hbm'])
  if plan['vu']:
   begin=max(now,vu_end);vu_end=begin+plan['vu'];finish=max(finish,vu_end)
   log('vu_service',task_id=key,start_cycle=begin,finish_cycle=vu_end)
  if plan['vector']:t.vector_pending=False
  else:
   b,m,n,h,w,k0,k=(plan[x] for x in ('b','m','n','h','w','k0','k'))
   if t.partial is None:
    rb,bc,rm,rh,rn,rw=t.rectangles.pop(0);assert (rb,rm,rn)==(b,m,n)
    new=[]
    if rw>w:new.append((b,1,m,h,n+w,rw-w))
    if rh>h:new.append((b,1,m+h,rh-h,n,rw))
    if bc>1:new.append((b+1,bc-1,m,rh,n,rw))
    t.rectangles[:0]=new
   t.partial=(p,b,m,n,h,w,k0+k) if k0+k<p.shape[3] else None
   log('tile_start',task_id=key,layer=t.layer,group=t.groups,b=b,m=m,n=n,h=h,w=w,k0=k0,k=k,
       geometry=list(p.geometry),memory_tile=list(p.tile),share=plan['share'],borrowed=borrow)
  final=t.partial is None and not t.rectangles and not t.vector_pending
  if final and not p.resident:finish=max(finish,t.origin+chip['hbm_latency_cycles'])
  t.inflight=finish;t.changed();running.add(key);waiting.discard(key)
  assert used_memory()<=chip['capacity_bytes'] and occupied_cores<=units
  push(finish,'group_done',key,(plan,final))
  log('group_start',task_id=key,layer=t.layer,group=t.groups,finish_cycle=finish,held_bytes=used_memory(),borrowed=borrow)
  return True
 while idx<len(work) or queue:
  if not queue and not pending:now=max(now,work[idx][0])
  while pending and pending[0][0]<=now:
   _,_,kind,key,payload=heapq.heappop(pending)
   if kind=='release':
    if tessera:
     for y,mask in enumerate(payload):assert fabric.rows[y]&mask==mask;fabric.rows[y]^=mask
    else:occupied_cores-=payload
   else:
    t=queue[key];plan,final=payload;t.inflight=None;running.discard(key)
    if not plan['vector']:t.completed_macs+=plan['h']*plan['w']*plan['k']
    # Reserve the output tile's ping-pong workspace until C is complete. This
    # avoids accepting new tenants into storage needed by a live K accumulator.
    if t.partial is None:t.held=0
    log('group_done',task_id=key,layer=t.layer,group=t.groups)
    if final:t.layer+=1;t.reset_layer()
    t.changed()
    if t.layer==t.count:
     done.append(dict(task_id=key,network=t.task_name,start_cycle=t.start_time,finish_cycle=now,
      latency_ms=(now-t.start_time)/FREQ*1e3,sla_ms=t.sla,useful_macs=t.completed_macs,groups=t.groups))
     del queue[key];log('task_done',task_id=key)
  while idx<len(work) and work[idx][0]<=now:
   arrival,name,priority=work[idx];queue[idx]=MatchedTask(idx,name,arrival,now,priority,qos[name],model)
   log('arrival',task_id=idx);idx+=1
  if fail_budget is not None:
   late=Counter()
   for r in done:
    if r['latency_ms']>r['sla_ms']:late['tr' if r['network']=='GNMT' else 'cd']+=1
   for t in queue.values():
    if (now-t.start_time)/FREQ*1e3>t.sla:late['tr' if t.task_name=='GNMT' else 'cd']+=1
   if any(late[g]>limit for g,limit in fail_budget.items()):raise ProvenInfeasible(done,events,allocations,late,now)
  if queue:
   if not asynchronous and not running:
    cohort=set(queue);waiting=set(queue);log('round_open',members=list(queue))
   for t in queue.values():t.current_time=now
   admitted=queue if asynchronous else OrderedDict((k,t) for k,t in queue.items() if k in cohort)
   possible={key:t.possible() for key,t in admitted.items()}
   if allocation_policy is not None:alloc=allocation_policy(admitted,possible,units)
   elif POLICY.check_if_tasks_all_fit(possible,units):alloc=POLICY.assign_cores_if_tasks_fit(admitted,possible,units)
   else:alloc=POLICY.assign_cores_if_tasks_not_fit(admitted,possible,units,FREQ)
   assert set(alloc)==set(admitted) and 0<sum(alloc.values())<=units
   allocations+=1;log('allocation',targets=dict(alloc))
   eligible=list(queue) if asynchronous else [key for key in queue if key in waiting]
   for key in eligible:admit(key,alloc[key])
   if tessera:
    for key in eligible:admit(key,alloc[key],borrow=True)
  future=[pending[0][0]] if pending else []
  if idx<len(work):future.append(work[idx][0])
  if not future:
   if queue:raise RuntimeError('No legal PE/SRAM allocation can continue an output tile')
   break
  next_event=min(future);assert next_event>now;now=next_event;events+=1
 assert occupied_cores==0 and (fabric is None or not any(fabric.rows))
 assert sorted(x['task_id'] for x in done)==list(range(len(work)))
 for r in done:
  expected=sum(math.prod(p.shape) for p in model.profiles[r['network']][units] if p.kind=='matrix')
  assert r['useful_macs']==expected
 return done,events,allocations

def make_matched_models(profile_path):
 allrows=list(rows(profile_path/'operator_costs.csv'));layers=list(rows(INPUT/'cnn_layers.csv'))
 configs=json.loads((profile_path/'configs.json').read_text());result={}
 for arch,total in UNITS.items():
  by={(r['kind'],r['key'],int(r['cores'])):r for r in allrows if r['architecture']==arch}
  profiles={}
  for name,stems in C['NET_CSVS'].items():
   definitions=[r for stem in stems for r in layers if r['network']==stem]
   profiles[name]={c:[by[r['kind'],'x'.join(r[k] for k in (('B','M','N','K') if r['kind']=='matrix' else ('input_bytes','output_bytes','vector_ops'))),c] for r in definitions] for c in range(1,total+1)}
  cfg=configs[arch+'/'+str(total)]
  chip=dict(side=cfg['sa_dim'],grain=cfg['tessera_parameters']['grain'],capacity_bytes=cfg['vmem_size_MB']*1024**2,
   frequency_Hz=cfg['freq_GHz']*1e9,hbm_bytes_per_cycle=cfg['hbm_bw_GBps']*1024**3/FREQ,hbm_latency_cycles=cfg['hbm_latency_ns']*FREQ/1e9)
  result[arch]=MatchedCosts(profiles,chip,arch)
 return result
# Codex: decision end

def init(path,dispatch='rounds',prune_infeasible=False,execution_model='matched_tiles'):
 global INFO,REGION_PROFILES,REGION_CHIP,DISPATCH,PRUNE_INFEASIBLE,MATCHED_MODELS,EXECUTION_MODEL
 EXECUTION_MODEL=execution_model
 if execution_model=='matched_tiles':
  MATCHED_MODELS=make_matched_models(Path(path));DISPATCH=dispatch;PRUNE_INFEASIBLE=prune_infeasible
  return
 INFO=make_info(Path(path));DISPATCH=dispatch;PRUNE_INFEASIBLE=prune_infeasible
 if dispatch!='barrier':REGION_PROFILES,REGION_CHIP=make_region_profiles(Path(path))

def pool_job(job):
 wk,qk,arch,lam,seed=job;np.random.seed(seed)
 gen=GEN.WorkloadGenerator(distribution='poisson',args={'max_cycles':0,'arrival_time':lam,'frequency':FREQ,'N':C['NUM_TASKS']})
 for name in C['WORKLOADS'][wk]:gen.add_task_type(name)
 with contextlib.redirect_stdout(io.StringIO()):work=gen.generate()
 work=[(int(t),str(n),int(p)) for t,n,p in work]
 qos={n:C['QOS_BASE_MS'][n]*C['QOS_LEVELS'][qk] for n in C['WORKLOADS'][wk]}
 # chenyi9: decision start — enable the extra dispatcher only by explicit flag.
 proof=None
 try:
  if EXECUTION_MODEL=='matched_tiles':
   records,events,alloc=run_matched(work,MATCHED_MODELS[arch],qos,seed,
    asynchronous=arch=='Tessera-8' and DISPATCH=='region_async',fail_budget=failure_budgets() if PRUNE_INFEASIBLE else None)
  elif arch=='Tessera-8' and DISPATCH!='barrier':
   records,events,alloc=run_regions(work,INFO[arch],qos,seed,REGION_PROFILES,REGION_CHIP,asynchronous=DISPATCH=='region_async',fail_budget=failure_budgets() if PRUNE_INFEASIBLE else None)
  else:records,events,alloc=run(work,INFO[arch],qos,seed)
 except ProvenInfeasible as exc:
  records,events,alloc=exc.done,exc.events,exc.allocations
  proof=dict(known_deadline_misses=exc.late,maximum_permitted_failures=failure_budgets(),cutoff_cycle=exc.cutoff)
 # chenyi9: decision end
 count=Counter()
 for _,name,_ in work:count['tr_n' if name=='GNMT' else 'cd_n']+=1
 for r in records:
  group='tr' if r['network']=='GNMT' else 'cd';count[group+'_ok']+=int(r['latency_ms']<=r['sla_ms'])
 return dict(workload=wk,qos=qk,architecture=arch,lambda_ms=lam,seed=seed,cd_n=count['cd_n'],cd_ok=count['cd_ok'],tr_n=count['tr_n'],tr_ok=count['tr_ok'],
  proven_infeasible=proof is not None,completed_tasks=len(records),infeasibility_proof=json.dumps(proof,sort_keys=True),
  events=events,allocations_checked=alloc,arrival_sha256=hashlib.sha256(json.dumps(work).encode()).hexdigest(),
  completion_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())

def feasible(rr):
 if any(r.get('proven_infeasible',False) for r in rr):return False
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
 # Codex: decision start — the same CLI checks the requested spatial modes.
 import unittest
 from neusim.npusim.backend.tests.test_tessera_partitioned import Algorithm2Tests
 from neusim.npusim.backend.tests.test_matched_qos import MatchedQoSTests
 log=io.StringIO()
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (Algorithm2Tests,MatchedQoSTests)])
 result=unittest.TextTestRunner(stream=log).run(suite)
 if not result.wasSuccessful():raise AssertionError(log.getvalue())
 return dict(status='PASS',hand_golden_cases=6,original_allocation_golden_cases=4,spatial_unit_tests=result.testsRun)
 # Codex: decision end

def main():
 p=argparse.ArgumentParser();p.add_argument('--profiles',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=15);p.add_argument('--check',action='store_true')
 p.add_argument('--tessera-dispatch',choices=('barrier','rounds','region_async'),default='rounds')
 p.add_argument('--exact-infeasibility-pruning',action='store_true')
 p.add_argument('--execution-model',choices=('matched_tiles','legacy_profiles'),default='matched_tiles')
 # chenyi9: decision start — the complete rerun may use 64 CPUs and 200 GB RAM.
 p.add_argument('--memory-gb',type=float,default=100);p.add_argument('--cpu-offset',type=int,default=0);a=p.parse_args()
 assert 1<=a.workers<=64 and 0<a.memory_gb<=200
 assert 0<=a.cpu_offset and a.cpu_offset+a.workers<=64
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[a.cpu_offset:a.cpu_offset+a.workers]);cap=int(a.memory_gb*0.94*1e9)//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
 # chenyi9: decision end
 save_json(out/'checks.json',check())
 sources=[Path(__file__),ROOT/'neusim/npusim/backend/tessera_partitioned.py',ROOT/'neusim/npusim/backend/tessera_baselines.py',ROOT/'neusim/npusim/backend/tessera.py',ROOT/'neusim/npusim/backend/tests/test_matched_qos.py',SRC/'scheduler.py',SRC/'generator.py',PROJECT/'FissionSA/multitenant/run_abc_qos.py'];hashes={str(x):sha(x) for x in sources}
 save_json(out/'scenario.json',dict(constants=C,frequency_Hz=FREQ,policy_sha256=hashes,allocation_units=UNITS,tessera_dispatch=a.tessera_dispatch,execution_model=a.execution_model,workers=a.workers,memory_budget_GB=a.memory_gb,cpu_offset=a.cpu_offset,exact_infeasibility_pruning=a.exact_infeasibility_pruning,failure_budgets=failure_budgets(),
  resource_policy='Matched tiles: identical finite live SRAM, shared HBM/VU queues, native array kernels and output-tile remapping boundary. Exact B/M/N/K progress, pinned partial C, and execution-consistent remaining-work estimates. Planaria retains its long compositions and synchronous rounds; Tessera uses spatial first-fit and optional asynchronous backfill. Legacy profiles are a separately labelled diagnostic control.',arrival_policy='Original integer-ms Poisson interarrival samples, 50 tasks/pool; not exponential intervals'))
 if a.check:print('PASS scheduler unit checks');return
 assert json.loads((a.profiles/'verification.json').read_text())['status']=='PASS'
 # No old fixed-grid profiles may silently enter a newly labelled experiment.
 from neusim.npusim.backend.tessera_partitioned import PACKING_POLICY
 for r in rows(a.profiles/'operator_costs.csv'):
  if r['architecture']=='Tessera-8' and r.get('packing_policy')!=PACKING_POLICY:
   raise ValueError('Corrected Algorithm 2 profiles are required')
 allrows=[];cfgs=[(w,q,arch) for w in C['WORKLOADS'] for q in C['QOS_LEVELS'] for arch in ('Planaria-32','Tessera-8')];state={};t=time.monotonic()
 (out/'pools').mkdir()
 def evaluate(pool,jobs):
  futures={pool.submit(pool_job,job):i for i,job in enumerate(jobs)};completed={}
  for future in as_completed(futures):
   i=futures[future];record=future.result();completed[i]=record
   key=hashlib.sha256(json.dumps(jobs[i]).encode()).hexdigest()
   save_json(out/'pools'/(key+'.json'),dict(status='PASS',job=list(jobs[i]),result=record))
   if len(completed)%C['S_SEARCH']==0:print('Pools',len(allrows)+len(completed),'point',len(completed),'/',len(jobs),'seconds',round(time.monotonic()-t,1),flush=True)
  rr=[completed[i] for i in range(len(jobs))]
  if rr:save_csv(out/('point_pools_'+str(len(allrows))+'.csv'),rr)
  allrows.extend(rr);return rr
 with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(a.profiles.resolve()),a.tessera_dispatch,a.exact_infeasibility_pruning,a.execution_model)) as pool:
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
 save_json(out/'verification.json',dict(status='PASS',pools=len(allrows),source_tasks=len(allrows)*C['NUM_TASKS'],tasks_completed=sum(r['completed_tasks'] for r in allrows),proven_infeasible_pools=sum(r['proven_infeasible'] for r in allrows),allocation_checks=sum(r['allocations_checked'] for r in allrows),source_sha256=hashes,profiles_sha256=sha(a.profiles/'operator_costs.csv'),artifacts={n:sha(out/n) for n in ('qps.csv','pool_results.csv','scenario.json')},elapsed_s=time.monotonic()-t))
 print('PASS QoS',len(allrows),'pools',flush=True)
if __name__=='__main__':main()
