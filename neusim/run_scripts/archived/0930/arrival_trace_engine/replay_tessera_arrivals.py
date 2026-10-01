"""Native-cost spatial replay with arrival gaps and request dependencies.

The operator lease is conservative: its native-selected physical lane footprint
is held over SA execution; FlashAttention holds the union of its phase footprints.
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
        mapping=json.loads(r['mapping_json']);mask=[0]*self.side
        phases=mapping.get('phase_plans') or ([mapping] if 'geometry' in mapping else [])
        for p in phases:
            if p.get('engine')!='SA':continue
            g=p['geometry'];mt,nt,kt=p['memory_tile'];h,w=g[-2:]
            nk=(kt+g[0]-1)//g[0];nn=(nt+g[1]-1)//g[1]
            lanes=packed_lanes(nk,nn,self.side*self.side//(h*w),g[3])
            fabric=FirstFitFabric(self.side,self.grain)
            for i in range(lanes):assert fabric.place(i,h,w) is not None
            mask=[a|b for a,b in zip(mask,fabric.rows)]
        usedrows=[i for i,r in enumerate(mask) if r]
        if usedrows:
            height=max(usedrows)+1;width=max(r.bit_length() for r in mask)
            mask=mask[:height]
        else:height=width=0;mask=[]
        p=dict(row=r,time=t,work=work,dynamic=dynamic,mask=mask,height=height,width=width,
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

class Fabric:
    """Translate native phase footprints without splitting any physical FCR."""
    def __init__(self,costs):self.costs=costs;self.rows=[0]*costs.side;self.held={}
    def place(self,key,p):
        if not p['mask']:return True
        c=self.costs
        for y in range(0,c.side-p['height']+1,c.grain):
            for x in range(0,c.side-p['width']+1,c.grain):
                mask=[m<<x for m in p['mask']]
                if all(not (self.rows[y+i]&m) for i,m in enumerate(mask)):
                    for i,m in enumerate(mask):self.rows[y+i]|=m
                    self.held[key]=(y,x,mask);return True
        return False
    def release(self,key):
        if key not in self.held:return
        y,x,mask=self.held.pop(key)
        for i,m in enumerate(mask):assert self.rows[y+i]&m==m;self.rows[y+i]^=m

def chains_run(chains,costs,mode,trace=None):
    """Adjacent independent request chains, FCFS within each ready cohort."""
    if mode=='serial':return costs.serial(sum(chains,[]))
    fabric=Fabric(costs);positions=[0]*len(chains);running={};pending=[]
    now=0.;serial=0;result=Counter();servers=dict(hbm=0.,vu=0.,ici=0.)
    cohort=set();done=set()
    while len(done)<len(chains):
        while pending and pending[0][0]<=now:
            _,_,kind,key=heapq.heappop(pending)
            if kind=='array':fabric.release(key)
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
            if not fabric.place(key,p):break
            end=now+p['time']
            # Native bandwidth service is distinct from the one-op HBM latency
            # floor. Repeated fetches retain their complete native byte count.
            hbm_service=float(p['row']['hbm_bytes'])/(costs.chip.hbm_bw_GBps*1024**3)*1e9
            for comp,duration in [('hbm',hbm_service),('vu',p['vu']),('ici',p['ici'])]:
                if duration:
                    servers[comp]=max(now,servers[comp])+duration;end=max(end,servers[comp])
            running[key]=(p,end-now)
            serial+=1;heapq.heappush(pending,(end,serial,'operator',key))
            if p['sa']:
                serial+=1;heapq.heappush(pending,(now+p['sa'],serial,'array',key))
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

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['index']);p.add_argument('--costs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);resource.setrlimit(resource.RLIMIT_AS,(1_000_000_000,1_000_000_000))
    index_costs(a.costs,a.out)
if __name__=='__main__':main()
