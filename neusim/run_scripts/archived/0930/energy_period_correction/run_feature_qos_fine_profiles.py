"""Profile user-selected native task allocation grains without duplicating runs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,csv,json,resource,shutil,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from neusim.run_scripts import run_feature_cnn_qos as base
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,rows,sha,save_json
from neusim.npusim.backend import tessera_joint as joint,tessera_partitioned as part
import numpy as np

# chenyi9: decision start — preserve each architecture's actual fission grain.
UNITS={'Planaria-32':16,'Tessera-8':256}
ORIGINAL_CHIP=base.chip
SHARED_VU=False

def fine_chip(arch='Tessera-8',grain=8,cores=256,objective='e2e_latency'):
 # Grain is a physical parameter, not an algorithm/policy switch.
 grain=int(arch.rsplit('-',1)[1])
 c=ORIGINAL_CHIP('Tessera-8' if arch.startswith('Tessera-') else arch,grain,16,objective)
 share=cores/((c.sa_dim//grain)**2)
 assert 0<share<=1
 c.tessera_parameters['active_pe_budget']=int(c.sa_dim**2*share)
 c.tessera_parameters['selection_bandwidths']=[base.HBM*share]
 c.vmem_size_MB*=share;c.hbm_bw_GBps*=share
 # /root: decision start — mapping and allocation use the same shared-resource fraction.
 if SHARED_VU:c.tessera_parameters['selection_vu_fraction']=share
 # /root: decision end
 return c
# chenyi9: decision end

def init(grains=(8,),shared_vu=False):
 global UNITS,SHARED_VU
 SHARED_VU=shared_vu
 side=ORIGINAL_CHIP().sa_dim
 UNITS={'Planaria-32':(side//32)**2,**{f'Tessera-{g}':(side//g)**2 for g in grains}}
 base.chip=fine_chip

def main():
 p=argparse.ArgumentParser();p.add_argument('--coarse',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=15);p.add_argument('--memory-gb',type=float,default=100)
 p.add_argument('--grains',type=int,nargs='+',default=[8],choices=base.GRAINS)
 p.add_argument('--shared-vu',action='store_true',help='Use the task resource fraction when ranking native VU demand')
 a=p.parse_args()
 assert 1<=a.workers<=64 and 0<a.memory_gb<=200
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers]);cap=int(a.memory_gb*.94*1e9)//(a.workers+1);resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
 assert json.loads((a.coarse/'verification.json').read_text())['status']=='PASS'
 # chenyi9: decision start — never combine old fixed-grid and corrected profiles.
 for r in rows(a.coarse/'operator_costs.csv'):
  if r['architecture']=='Tessera-8' and r.get('packing_policy')!=part.PACKING_POLICY:
   raise ValueError('Coarse Tessera profiles predate Algorithm 2; generate corrected profiles first')
 # chenyi9: decision end
 init(a.grains,a.shared_vu);checks=0
 for arch,total in UNITS.items():
  for core in sorted({1,max(1,total//2),total}):
   for dims in ((1,5,33,67),(32,49,1,9)):
    result=base.profile_job(('matrix','test',dims,arch,core,'e2e_latency'))
    assert result[0]['time_ns']>=result[0]['sa_ns'];checks+=1
 save_json(out/'fine_allocation_checks.json',dict(status='PASS',scalar_native_cases=checks,allocation_units=UNITS))
 unique={}
 for kind,key,payload,arch,core,obj in base.jobs('qos_profiles'):unique[kind,key]=(kind,key,payload)
 sources=[Path(__file__),Path(base.__file__),ROOT/'neusim/configs/chips/ChipConfig.py',*list((ROOT/'neusim/npusim/backend').glob('*.py'))]
 hashes={str(x):sha(x) for x in sources}
 for x in sources:
  dst=out/'source'/x.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(x,dst);assert sha(dst)==sha(x)
 configs={f'{arch}/{c}':fine_chip(arch,int(arch.rsplit('-',1)[1]),c).model_dump() for arch,n in UNITS.items() for c in range(1,n+1)}
 save_json(out/'configs.json',configs)
 source_configs=json.loads((a.coarse/'configs.json').read_text());reused_keys=set()
 reused=0;ran=0;t=time.monotonic()
 with (out/'operator_costs.csv').open('x',newline='') as f:
  writer=None
  for r in rows(a.coarse/'operator_costs.csv'):
   arch=r['architecture']
   if arch not in UNITS:continue
   old=source_configs[f"{arch}/{r['cores']}"];grain=int(arch.rsplit('-',1)[1])
   actual=old['tessera_parameters'].get('active_pe_budget',old['sa_dim']**2)//grain**2
   new=configs[f'{arch}/{actual}']
   # /root: decision start — changed mapping objectives must execute fresh profiles.
   if old!=new and a.shared_vu:continue
   assert old==new, 'Reuse requires the exact same native chip and mapping objective'
   # /root: decision end
   r['cores']=str(actual)
   assert r['objective']=='e2e_latency'
   key=r['kind'],r['key'],arch,actual;assert key not in reused_keys;reused_keys.add(key)
   if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
   writer.writerow(r);reused+=1
  jj=[(*x,arch,core,'e2e_latency') for x in unique.values() for arch,n in UNITS.items()
      for core in range(1,n+1) if (x[0],x[1],arch,core) not in reused_keys]
  with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(a.grains,a.shared_vu)) as pool:
   for rr in base.bounded_map(pool,base.profile_job,jj,block=a.workers*4):
    if writer is None:writer=csv.DictWriter(f,fieldnames=list(rr[0]));writer.writeheader()
    writer.writerows(rr);ran+=1
    if ran%500==0:f.flush();print('fine profiles',ran,'/',len(jj),'elapsed_s',round(time.monotonic()-t,1),flush=True)
 keys=set();count=0
 for r in rows(out/'operator_costs.csv'):
  key=r['kind'],r['key'],r['architecture'],int(r['cores']);assert key not in keys;keys.add(key);count+=1
 assert count==len(unique)*sum(UNITS.values())
 for k in unique:
  for arch,n in UNITS.items():assert all((*k,arch,c) in keys for c in range(1,n+1))
 assert all(sha(Path(x))==h for x,h in hashes.items())
 save_json(out/'verification.json',dict(status='PASS',allocation_units=UNITS,reused_records=reused,new_simulated_records=ran,records=count,coarse_source_sha256=sha(a.coarse/'operator_costs.csv'),source_sha256=hashes,output_sha256=sha(out/'operator_costs.csv'),elapsed_s=time.monotonic()-t))
 print('PASS native-grain QoS profiles',count,flush=True)
if __name__=='__main__':main()
