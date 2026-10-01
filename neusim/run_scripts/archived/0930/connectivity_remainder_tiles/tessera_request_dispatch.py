"""Dependency-preserving first-fit request replay at native SRAM-tile boundaries."""
from collections import Counter,OrderedDict
from functools import lru_cache
import heapq,math
from neusim.npusim.backend.tessera_partitioned import FirstFitFabric,packed_lanes,lane_cycles
from neusim.run_scripts.run_tessera_partitioned import ENERGIES

@lru_cache(maxsize=65536)
def tile_plan(occupied,side,grain,geometry,m,nk,nn,variant,frequency):
    """Cache exact first-fit masks; coalesce only simultaneous retirements."""
    h,w=geometry[-2:];probe=FirstFitFabric(side,grain);probe.rows=list(occupied)
    limit=packed_lanes(nk,nn,side**2//(h*w),geometry[3]);boxes=probe.available(h,w,limit)
    if not boxes:return None
    width=len(boxes);groups={};final=list(occupied)
    for lane,(r,c,hh,ww) in enumerate(boxes):
        duration=lane_cycles(m,geometry,variant,grain,(nk*nn+width-1-lane)//width)/frequency
        mask=((1<<(ww//grain))-1)<<(c//grain)
        masks=groups.setdefault(duration,[0]*len(occupied))
        for y in range(r//grain,(r+hh)//grain):
            assert not final[y]&mask;final[y]|=mask;masks[y]|=mask
    return tuple(final),tuple((dt,tuple(mask)) for dt,mask in sorted(groups.items()))

def sections(batch):
    yield [batch['prefix']]
    for _ in range(batch['layers']):
        yield [batch['before']];yield batch['chains'];yield [batch['after']]
    yield [batch['suffix']]

def segments(p,costs):
    """QK/PV/head order is retained; equal SRAM tiles are run-length encoded."""
    phases=p['phases']
    def segment(phase):
        g=phase['geometry'];m,n,k=phase['memory_tile']
        nk=(k+g[0]-1)//g[0];nn=(n+g[1]-1)//g[1]
        width=packed_lanes(nk,nn,costs.side**2//(g[-2]*g[-1]),g[3])
        cycle=lane_cycles(m,g,costs.chip.tessera_variant,costs.grain,(nk*nn+width-1)//width)/costs.chip.freq_GHz
        groups=float(phase['sa_ns'])/cycle
        assert math.isclose(groups,round(groups),rel_tol=1e-12),(groups,phase)
        return dict(g=g,m=m,nk=nk,nn=nn,groups=round(groups))
    if phases and 'phase' in phases[0]:
        assert len(phases)%2==0
        for i in range(0,len(phases),2):
            qk,pv=phases[i:i+2];assert qk['phase']=='QK' and pv['phase']=='PV' and qk['copies']==pv['copies']
            a,b=segment(qk),segment(pv)
            for _ in range(qk['copies']):yield dict(a);yield dict(b)
    elif phases:
        assert len(phases)==1;yield segment(phases[0])

def batch_cost(batch,costs,mode):
    result=costs.serial(batch['prefix']+batch['suffix'])
    body=costs.serial(batch['before']+batch['after']);body.update(chains_run(batch['chains'],costs,mode))
    result.update({k:v*batch['layers'] for k,v in body.items()});return result

def chains_run(chains,costs,mode,trace=None):
    if mode=='serial' or len(chains)==1:return costs.serial(sum(chains,[]))
    batch=dict(batch_id=0,arrival_ns=[0],release_ns=0,dependencies=[],prefix=[],suffix=[],
      before=[],after=[],chains=chains,layers=1)
    result,_=dispatch([batch],costs,mode,trace=trace,compress=False)
    # Energy is reconstructed by the caller from the component columns.
    result.pop('energy_J',None);return result

def replay_trace(batches,costs,mode,log=None):return dispatch(batches,costs,mode,trace=log,compress=True)

def dispatch(batches,costs,mode,trace=None,compress=True):
    """FCFS adjacent ready requests; grow allocations at the next SRAM tile.

    Running FCRs stay pinned until their last fold. Free physical holes can
    begin another request's tile immediately in region_async mode. The same
    selected geometry/tile is retained; only simultaneous lane count adapts.
    Traffic and arithmetic remain exactly the native selected work. Native
    HBM/VU/ICI service overlaps the array; operand readiness is roofline-level.
    """
    # chenyi9: decision start — retain idle time before arrivals, including the first.
    # Source request timestamps are relative to the trace's time-zero origin.
    origin=0.;assert min(min(b['arrival_ns']) for b in batches)>=origin
    now=origin;idx=sequence=0
    # chenyi9: decision end
    fabric=FirstFitFabric(costs.side,costs.grain);states=OrderedDict();live={};pending=[];done={};cohort=set();round_waiting=set()
    servers=dict(hbm=now,vu=now,ici=now);result=Counter();intervals=[];macros=[]
    def push(t,kind,payload):
        nonlocal sequence
        sequence+=1;heapq.heappush(pending,(t,sequence,kind,payload))
    def advance_state(bid):
        state=states[bid]
        while all(p==len(c) for p,c in zip(state['positions'],state['chains'])):
            try:state['chains']=next(state['iterator']);state['positions']=[0]*len(state['chains'])
            except StopIteration:done[bid]=now;del states[bid];return
    def next_batch_ready():return idx<len(batches) and all(d in done for d in batches[idx]['dependencies'])
    def finish_array(key):
        op=live[key];op['array_done']=True
        push(max(now,op['component_end']),'operator',key)
    while idx<len(batches) or states:
        while pending and pending[0][0]<=now:
            _,_,kind,payload=heapq.heappop(pending)
            if kind=='region':
                masks=fabric.regions.pop(payload)
                for i,mask in enumerate(masks):assert fabric.rows[i]&mask==mask;fabric.rows[i]^=mask
            elif kind=='tile':
                key=payload;op=live[key];op['group_active']=False;op['segment']['groups']-=1;cohort.discard(key)
                if op['segment']['groups']==0:
                    try:op['segment']=next(op['segments'])
                    except StopIteration:finish_array(key)
            else:
                key=payload;op=live.pop(key);cohort.discard(key)
                intervals.append((op['start'],now,op['profile']))
                states[key[0]]['positions'][key[1]]+=1
                if trace is not None:trace.append(dict(batch_id=key[0],chain=key[1],start_ns=op['start'],finish_ns=now))
        for bid in list(states):advance_state(bid)
        if compress and not states and not pending and next_batch_ready():
            b=batches[idx];now=max(now,b['release_ns']);own=batch_cost(b,costs,mode)
            nxt=batches[idx+1] if idx+1<len(batches) else None
            safe=(nxt is None or b['batch_id'] in nxt['dependencies'] or nxt['release_ns']>=now+own['time_ns'] or mode=='serial')
            if safe:
                macros.append((now,now+own['time_ns']));result.update(own);now+=own['time_ns'];done[b['batch_id']]=now;idx+=1
                for c in servers:servers[c]=now
                result['compressed_batches']+=1;continue
        while next_batch_ready() and batches[idx]['release_ns']<=now:
            b=batches[idx];bid=b['batch_id'];states[bid]=dict(iterator=iter(sections(b)),chains=[],positions=[]);idx+=1;advance_state(bid)
        ready=[]
        for bid,state in states.items():
            for k,(position,chain) in enumerate(zip(state['positions'],state['chains'])):
                key=(bid,k)
                if position<len(chain) and (key not in live or not live[key]['array_done'] and not live[key]['group_active']):ready.append(key)
        # Already admitted operators must be able to finish their held SRAM
        # tiles before a new operator can claim that memory. Preserve their
        # admission order; unadmitted adjacent requests remain FCFS.
        ready.sort(key=lambda k:(0,live[k]['start'],k) if k in live else (1,now,k))
        if mode=='rounds':
            if not cohort:round_waiting=set(ready)
            ready=[k for k in ready if k in round_waiting]
        if mode=='serial':ready=ready[:1] if not live else [k for k in ready if k in live]
        for key in ready:
            if key not in live:
                state=states[key[0]];oid=state['chains'][key[1]][state['positions'][key[1]]];p=costs.get(oid)
                if sum(op['profile']['peak'] for op in live.values())+p['peak']>costs.capacity:break
                iterator=iter(segments(p,costs));seg=next(iterator,None)
                if seg and tile_plan(tuple(fabric.rows),costs.side,costs.grain,tuple(seg['g']),seg['m'],seg['nk'],seg['nn'],costs.chip.tessera_variant,costs.chip.freq_GHz) is None:break
                end=now+p['time']
                for comp,duration in [('hbm',float(p['row']['hbm_bytes'])/(costs.chip.hbm_bw_GBps*1024**3)*1e9),('vu',p['vu']),('ici',p['ici'])]:
                    if duration:servers[comp]=max(now,servers[comp])+duration;end=max(end,servers[comp])
                live[key]=dict(profile=p,start=now,component_end=end,segments=iterator,segment=seg,group_active=False,array_done=False)
                result['materialized_operators']+=1
                if seg is None:
                    if mode=='rounds':cohort.add(key);round_waiting.discard(key)
                    finish_array(key);continue
            op=live[key];seg=op['segment'];g=seg['g'];h,w=g[-2:]
            plan=tile_plan(tuple(fabric.rows),costs.side,costs.grain,tuple(g),seg['m'],seg['nk'],seg['nn'],costs.chip.tessera_variant,costs.chip.freq_GHz)
            if plan is None:break
            final,retirements=plan;fabric.rows=list(final);finish=now
            for duration,masks in retirements:
                token=(sequence,key);fabric.regions[token]=masks
                end=now+duration;push(end,'region',token);finish=max(finish,end)
            op['group_active']=True;push(finish,'tile',key)
            if mode=='rounds':cohort.add(key);round_waiting.discard(key)
            result['array_tiles']+=1
        future=[pending[0][0]] if pending else []
        if next_batch_ready() and batches[idx]['release_ns']>now:future.append(batches[idx]['release_ns'])
        if not future:
            if idx==len(batches) and not states:break
            raise AssertionError(('No adjacent legal progress',idx,list(states),list(live)))
        following=min(future);assert following>=now
        if fabric.regions:result['array_busy_ns']+=following-now
        now=following
    assert not fabric.regions and not pending and not live and len(done)==len(batches)
    # Integrate unknown-duration operator work after its physical schedule is
    # complete. Piecewise activity sums avoid duplicate whole-chip static power.
    events=[]
    for start,end,p in intervals:
        assert end>start
        rates=[p['work'][c]/(end-start) for c in costs.power if c!='other']
        dynamic=[p['dynamic'][c]/(end-start) for c in costs.power if c!='other']
        events.append((start,1,rates,dynamic));events.append((end,-1,rates,dynamic))
    for start,end in macros:events.append((start,2,[],[]));events.append((end,-2,[],[]))
    events.sort(key=lambda e:e[0]);clock=origin;active=macro=0;work=[0.]*5;dynamic=[0.]*5
    work_count=[0]*5;dynamic_count=[0]*5
    from neusim.run_scripts.replay_tessera_arrivals import efficiency,COMPONENTS
    for when,sign,ww,dd in events:
        dt=when-clock
        if dt and not macro:
            for i,c in enumerate(COMPONENTS):
                activity=work[i] if active else 0.
                eta=efficiency(activity)
                result['static_energy_'+c+'_J']+=costs.power[c]*dt/1e9/eta
                result['dynamic_energy_'+c+'_J']+=max(0.,dynamic[i])*dt/eta
            result['static_energy_other_J']+=costs.power['other']*dt/1e9
            result['dynamic_energy_other_J']+=costs.chip.dynamic_power_other_W*dt/1e9
            if not active:result['idle_ns']+=dt
        if abs(sign)==2:macro+=sign//2
        else:
            active+=sign
            for i in range(5):
                work[i]+=sign*ww[i];dynamic[i]+=sign*dd[i]
                work_count[i]+=sign*int(ww[i]!=0);dynamic_count[i]+=sign*int(dd[i]!=0)
                if not work_count[i]:work[i]=0.
                if not dynamic_count[i]:dynamic[i]=0.
            if not active:work=[0.]*5;dynamic=[0.]*5
        clock=when
    assert not macro and not active and math.isclose(clock,now,abs_tol=1e-6)
    result['time_ns']=now-origin;result['energy_J']=math.fsum(result[k] for k in ENERGIES)
    return result,done
