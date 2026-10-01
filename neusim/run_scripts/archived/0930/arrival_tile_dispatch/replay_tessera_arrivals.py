"""Native-cost spatial replay with arrival gaps and request dependencies.

The operator lease is conservative: native-selected physical footprints are
reserved in advance, but each region retires after its own last native fold.
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
from concurrent.futures import ProcessPoolExecutor
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
        mapping=json.loads(r['mapping_json']);cells=self.side//self.grain
        mask=[0]*cells;retire=[[0.]*cells for _ in range(cells)]
        parents=list(range(cells*cells))
        def root(i):
            while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
            return i
        phases=mapping.get('phase_plans') or ([{**mapping,'sa_ns':float(r['sa_ns'])}] if 'geometry' in mapping else [])
        # chenyi9: decision start — a region's last fold releases its own cells.
        # FA plans are run-length summaries of ordered QK/PV pairs. The last
        # copy's QK precedes its PV; a phase must never consume future PV data.
        ends={};end=float(r['sa_ns'])
        if mapping.get('phase_plans'):
            assert len(phases)%2==0
            for j in range(len(phases)-2,-1,-2):
                qk,pv=phases[j:j+2];assert qk['phase']=='QK' and pv['phase']=='PV' and qk['copies']==pv['copies']
                ends[j+1]=end;ends[j]=end-float(pv['sa_ns'])
                end-=qk['copies']*(float(qk['sa_ns'])+float(pv['sa_ns']))
            assert abs(end)<1e-6
        elif phases:ends[0]=end
        for j,p in enumerate(phases):
            if p.get('engine')!='SA':continue
            g=p['geometry'];mt,nt,kt=p['memory_tile'];h,w=g[-2:]
            nk=(kt+g[0]-1)//g[0];nn=(nt+g[1]-1)//g[1]
            lanes=packed_lanes(nk,nn,self.side*self.side//(h*w),g[3])
            fabric=FirstFitFabric(self.side,self.grain)
            durations=[lane_cycles(mt,g,self.chip.tessera_variant,self.grain,(nk*nn+lanes-1-i)//lanes)/self.chip.freq_GHz for i in range(lanes)]
            # Earlier SRAM tiles use these same physical lanes. Only the final
            # tile's drain difference can be released before the phase ends.
            for i in range(lanes):
                box=fabric.place(i,h,w);assert box is not None
                rr,cc,hh,ww=(x//self.grain for x in box)
                release=ends[j]-max(durations)+durations[i]
                assert release>0
                for y in range(rr,rr+hh):
                    for x in range(cc,cc+ww):
                        retire[y][x]=max(retire[y][x],release)
                        parents[root(y*cells+x)]=root(rr*cells+cc)
            mask=[a|b for a,b in zip(mask,fabric.rows)]
        grouped={}
        for y in range(cells):
            for x in range(cells):
                if retire[y][x]:grouped.setdefault(retire[y][x],[0]*cells)[y]|=1<<x
        retirements=sorted(grouped.items())
        if mask and any(mask):assert math.isclose(retirements[-1][0],float(r['sa_ns']),rel_tol=1e-12)
        # Independent physical FCRs can occupy scattered holes. Regions that
        # overlap in different FA phases stay together so later interconnection
        # still joins contiguous PEs, preserving the selected native mapping.
        members={}
        for y in range(cells):
            for x in range(cells):
                if retire[y][x]:members.setdefault(root(y*cells+x),[]).append((y,x))
        pieces=[]
        for points in members.values():
            oy=min(y for y,x in points);ox=min(x for y,x in points)
            hh=max(y for y,x in points)-oy+1;ww=max(x for y,x in points)-ox+1
            local=[0]*hh
            for y,x in points:local[y-oy]|=1<<(x-ox)
            pieces.append(dict(origin=(oy,ox),height=hh,width=ww,mask=local))
        pieces.sort(key=lambda p:p['origin'])
        # chenyi9: decision end
        usedrows=[i for i,r in enumerate(mask) if r]
        if usedrows:
            height=max(usedrows)+1;width=max(r.bit_length() for r in mask)
            mask=mask[:height]
        else:height=width=0;mask=[]
        p=dict(row=r,time=t,work=work,dynamic=dynamic,mask=mask,height=height,width=width,retirements=retirements,pieces=pieces,phases=phases,
          sa=float(r['sa_ns']),peak=int(r['peak_live_bytes']),hbm=float(r['hbm_ns']),
          vu=float(r['vu_ns']),ici=float(r['ici_ns']))
        self.profiles[oid]=p;return p

    def partial(self,p,occupied):
        """Start the next adjacent request in available legal physical lanes.

        Geometry and SRAM tile remain fixed. NeuSim lane_cycles determines the
        additional folds when fewer lanes are available. Native arithmetic and
        byte counts are conserved; no fabricated request or shared operand is
        introduced. Future FA phases reserve their legal footprint in advance.
        """
        if not p['phases']:return None
        cells=self.side//self.grain;plans=[];array_ns=0.
        for phase in p['phases']:
            if phase.get('engine')!='SA':return None
            g=phase['geometry'];mt,nt,kt=phase['memory_tile'];h,w=g[-2:]
            nk=(kt+g[0]-1)//g[0];nn=(nt+g[1]-1)//g[1]
            full=packed_lanes(nk,nn,self.side*self.side//(h*w),g[3])
            fabric=FirstFitFabric(self.side,self.grain);fabric.rows=occupied[:]
            boxes=[]
            for i in range(full):
                box=fabric.place(i,h,w)
                if box is None:break
                boxes.append(box)
            if not boxes:return None
            native=lane_cycles(mt,g,self.chip.tessera_variant,self.grain,(nk*nn+full-1)//full)/self.chip.freq_GHz
            groups=float(phase['sa_ns'])/native
            assert math.isclose(groups,round(groups),rel_tol=1e-12),(groups,phase)
            durations=[lane_cycles(mt,g,self.chip.tessera_variant,self.grain,(nk*nn+len(boxes)-1-i)//len(boxes))/self.chip.freq_GHz for i in range(len(boxes))]
            phase_ns=round(groups)*max(durations)
            plans.append(dict(boxes=boxes,durations=durations,sa_ns=phase_ns,copies=phase.get('copies',1)))
            array_ns+=phase_ns*phase.get('copies',1)
        ends={};end=array_ns
        if 'phase_plans' in json.loads(p['row']['mapping_json']):
            for j in range(len(plans)-2,-1,-2):
                ends[j+1]=end;ends[j]=end-plans[j+1]['sa_ns']
                end-=plans[j]['copies']*(plans[j]['sa_ns']+plans[j+1]['sa_ns'])
        else:ends[0]=end
        retire=[[0.]*cells for _ in range(cells)];mask=[0]*cells
        for j,plan in enumerate(plans):
            for box,duration in zip(plan['boxes'],plan['durations']):
                y,x,h,w=(v//self.grain for v in box)
                release=ends[j]-max(plan['durations'])+duration
                for yy in range(y,y+h):
                    for xx in range(x,x+w):retire[yy][xx]=max(retire[yy][xx],release);mask[yy]|=1<<xx
        assert all(not (a&b) for a,b in zip(mask,occupied))
        grouped={}
        for y in range(cells):
            for x in range(cells):
                if retire[y][x]:grouped.setdefault(retire[y][x],[0]*cells)[y]|=1<<x
        # This stencil already occupies only free cells, retaining each future
        # phase's contiguous FCRs even across physically scattered holes.
        answer=dict(p,sa=array_ns,time=max(p['time'],array_ns),mask=mask,height=cells,width=cells,
          retirements=sorted(grouped.items()),pieces=[dict(origin=(0,0),height=cells,width=cells,mask=mask)],partial_lanes=True)
        return answer

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

class Fabric:
    """Translate native phase footprints without splitting any physical FCR."""
    def __init__(self,costs):self.costs=costs;self.rows=[0]*(costs.side//costs.grain);self.held={}
    def place(self,key,p):
        if not p['mask']:return True
        c=self.costs;probe=self.rows[:];placed=[]
        for piece in p['pieces']:
            found=None
            # FirstFitFabric exposes bitmaps in grain-sized cells, not PE units.
            for y in range(c.side//c.grain-piece['height']+1):
                for x in range(c.side//c.grain-piece['width']+1):
                    mask=[m<<x for m in piece['mask']]
                    if all(not (probe[y+i]&m) for i,m in enumerate(mask)):
                        found=(piece,y,x,mask);break
                if found:break
            if found is None:return False
            for i,m in enumerate(found[3]):probe[found[1]+i]|=m
            placed.append(found)
        self.rows=probe;self.held[key]=placed;return True
    def release(self,key,part=None):
        if key not in self.held:return
        for piece,y,x,mask in self.held[key]:
            oy,ox=piece['origin']
            released=mask[:] if part is None else [(((part[oy+i]>>ox)&piece['mask'][i])<<x) for i in range(len(mask))]
            for i,m in enumerate(released):
                assert self.rows[y+i]&m==m and mask[i]&m==m
                self.rows[y+i]^=m;mask[i]^=m
        if not any(any(item[3]) for item in self.held[key]):del self.held[key]

def chains_run(chains,costs,mode,trace=None):
    """Adjacent independent request chains, FCFS within each ready cohort."""
    if mode=='serial':return costs.serial(sum(chains,[]))
    fabric=Fabric(costs);positions=[0]*len(chains);running={};pending=[]
    now=0.;serial=0;result=Counter();servers=dict(hbm=0.,vu=0.,ici=0.)
    cohort=set();done=set()
    while len(done)<len(chains):
        while pending and pending[0][0]<=now:
            _,_,kind,key=heapq.heappop(pending)
            if kind=='array':fabric.release(*key)
            else:
                del running[key];positions[key]+=1;cohort.discard(key)
                if positions[key]==len(chains[key]):done.add(key)
        eligible=[k for k in range(len(chains)) if k not in done and k not in running]
        if mode=='rounds':
            if not cohort and not running:cohort=set(eligible)
            eligible=[k for k in eligible if k in cohort]
        for key in eligible:
            p=costs.get(chains[key][positions[key]])
            if sum(v[0]['peak'] for v in running.values())+p['peak']>costs.capacity:break
            if not fabric.place(key,p):
                partial=costs.partial(p,fabric.rows)
                if partial is None or not fabric.place(key,partial):break
                p=partial
            end=now+p['time']
            # Native bandwidth service is distinct from the one-op HBM latency
            # floor. Repeated fetches retain their complete native byte count.
            hbm_service=float(p['row']['hbm_bytes'])/(costs.chip.hbm_bw_GBps*1024**3)*1e9
            for comp,duration in [('hbm',hbm_service),('vu',p['vu']),('ici',p['ici'])]:
                if duration:
                    servers[comp]=max(now,servers[comp])+duration;end=max(end,servers[comp])
            running[key]=(p,end-now)
            serial+=1;heapq.heappush(pending,(end,serial,'operator',key))
            for retirement,part in p['retirements']:
                serial+=1;heapq.heappush(pending,(now+retirement,serial,'array',(key,part)))
            if trace is not None:trace.append(dict(chain=key,position=positions[key],start_ns=now,finish_ns=end,footprint=fabric.held.get(key)))
        if not pending:
            assert len(done)==len(chains),'No feasible adjacent request can progress'
            break
        following=pending[0][0];dt=following-now;assert dt>0
        result.update(costs.integrate(dt,list(running.values())))
        if fabric.held:result['array_busy_ns']+=dt
        now=following
    result['time_ns']=now
    assert not fabric.held
    return result

def batch_cost(batch,costs,mode):
    outside=costs.serial(batch['prefix']+batch['suffix'])
    body=costs.serial(batch['before']+batch['after'])
    body.update(chains_run(batch['chains'],costs,mode))
    outside.update({k:v*batch['layers'] for k,v in body.items()})
    return outside

def program_sections(batch):
    yield [batch['prefix']]
    for layer in range(batch['layers']):
        yield [batch['before']]
        yield batch['chains']
        yield [batch['after']]
    yield [batch['suffix']]

def replay_trace(batches,costs,mode,log=None):
    """Lazy source-order admission; no bypass of a dependent adjacent batch.

    A solo batch can be evaluated as repeated identical native layer bodies.
    If an independent adjacent batch arrives while it executes, materialize
    the affected programs and replay their operator/region events explicitly.
    """
    origin=min(min(b['arrival_ns']) for b in batches);now=origin
    fabric=Fabric(costs);states=OrderedDict();running={};pending=[];finished={}
    idx=serial=0;cohort=set();result=Counter();servers=dict(hbm=now,vu=now,ici=now)
    def section(state):
        try:state['chains']=next(state['iterator']);state['pos']=[0]*len(state['chains'])
        except StopIteration:return False
        return True
    def eligible_batch():
        return idx<len(batches) and all(k in finished for k in batches[idx]['dependencies'])
    while idx<len(batches) or states:
        while pending and pending[0][0]<=now:
            _,_,kind,key=heapq.heappop(pending)
            if kind=='array':fabric.release(*key)
            else:
                del running[key];states[key[0]]['pos'][key[1]]+=1;cohort.discard(key)
        for bid,state in list(states.items()):
            while all(p==len(c) for p,c in zip(state['pos'],state['chains'])):
                if not section(state):
                    finished[bid]=now;del states[bid];break
        # Exact compression is allowed only when no adjacent independent batch
        # can arrive before the solo batch's own completion.
        if not states and not pending and eligible_batch():
            b=batches[idx];start=max(now,b['release_ns'])
            if start>now:
                result.update(costs.integrate(start-now,[]));result['idle_ns']+=start-now;now=start
            own=batch_cost(b,costs,mode)
            nxt=batches[idx+1] if idx+1<len(batches) else None
            safe=(nxt is None or b['batch_id'] in nxt['dependencies'] or nxt['release_ns']>=now+own['time_ns'] or mode=='serial')
            if safe:
                result.update(own);now+=own['time_ns'];finished[b['batch_id']]=now;idx+=1
                for c in servers:servers[c]=now
                result['compressed_batches']+=1
                continue
        while eligible_batch() and batches[idx]['release_ns']<=now:
            b=batches[idx];bid=b['batch_id'];state=dict(iterator=iter(program_sections(b)))
            assert section(state);states[bid]=state;idx+=1
        ready=[]
        for bid,s in states.items():
            ready.extend((bid,k) for k,(p,c) in enumerate(zip(s['pos'],s['chains'])) if p<len(c) and (bid,k) not in running)
        if mode=='rounds':
            if not cohort and not running:cohort=set(ready)
            ready=[k for k in ready if k in cohort]
        if mode=='serial':ready=ready[:1] if not running else []
        for key in ready:
            s=states[key[0]];oid=s['chains'][key[1]][s['pos'][key[1]]];p=costs.get(oid)
            if sum(v[0]['peak'] for v in running.values())+p['peak']>costs.capacity:break
            if not fabric.place(key,p):
                partial=costs.partial(p,fabric.rows)
                if partial is None or not fabric.place(key,partial):break
                p=partial
            end=now+p['time']
            for comp,duration in [('hbm',float(p['row']['hbm_bytes'])/(costs.chip.hbm_bw_GBps*1024**3)*1e9),('vu',p['vu']),('ici',p['ici'])]:
                if duration:servers[comp]=max(now,servers[comp])+duration;end=max(end,servers[comp])
            running[key]=(p,end-now);serial+=1;heapq.heappush(pending,(end,serial,'operator',key))
            for retirement,part in p['retirements']:
                serial+=1;heapq.heappush(pending,(now+retirement,serial,'array',(key,part)))
            if log is not None:log.append(dict(batch_id=key[0],chain=key[1],operator_id=oid,start_ns=now,finish_ns=end))
            result['materialized_operators']+=1
        future=[pending[0][0]] if pending else []
        if eligible_batch() and batches[idx]['release_ns']>now:future.append(batches[idx]['release_ns'])
        if not future:
            if not states and idx==len(batches):break
            if states and all(all(p==len(c) for p,c in zip(s['pos'],s['chains'])) for s in states.values()):continue
            raise AssertionError(('No arrival/dependency/region event can make progress',idx,list(states)))
        following=min(future);dt=following-now;assert dt>0
        result.update(costs.integrate(dt,list(running.values())))
        if not running:result['idle_ns']+=dt
        if fabric.held:result['array_busy_ns']+=dt
        now=following
    assert not fabric.held and not pending and len(finished)==len(batches)
    result['time_ns']=now-origin
    result['energy_J']=math.fsum(result[k] for k in ENERGIES)
    return result,finished

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
    source=[Path(__file__),a.costs/'manifest.json',a.programs/'verification.json',a.store/'verification.json']
    hashes={str(p.resolve()):sha(p) for p in source}
    save_json(a.out/'scenario.json',dict(jobs=[list(j) for j in jobs],workers=a.workers,memory_gb=a.memory_gb,
      source_sha256=hashes,
      accounting='EDP = whole-chip NoPG energy including original arrival idle gaps * (last completion - first request arrival). Tensor-parallel energy includes all ranks.',
      admission='Source-order adjacent batches and chains only; preserve actual releases and autoregressive dependencies; no arbitrary future-job lookahead.',
      spatial_scope='Conservative native-selected footprint reservation with per-region final-fold retirement. FlashAttention keeps native QK/PV/head dependencies; shared HBM/VU/ICI FIFO service.',
      energy='All native traffic and padded compute/SRAM dynamic work retained; concurrent static energy charged once; fixed-voltage regulator efficiencies recomputed from concurrent mean activity. NoPG idle efficiency uses the native zero-activity point.'))
    completed=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize_run,initargs=tuple(str(getattr(a,k).resolve()) for k in ('costs','store','programs','out'))) as pool:
        for stem in pool.map(run_job,jobs,chunksize=1):
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
