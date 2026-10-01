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

def fine_chip(arch='Tessera-8',grain=8,cores=256,objective='e2e_latency'):
 c=ORIGINAL_CHIP(arch,grain,16,objective)
 share=cores/UNITS[arch]
 assert 0<share<=1
 c.tessera_parameters['active_pe_budget']=int(c.sa_dim**2*share)
 c.tessera_parameters['selection_bandwidths']=[base.HBM*share]
 c.vmem_size_MB*=share;c.hbm_bw_GBps*=share
 return c
# chenyi9: decision end

def init():base.chip=fine_chip

def main():
 p=argparse.ArgumentParser();p.add_argument('--coarse',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=15);a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:a.workers]);cap=5_000_000_000;resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
 assert json.loads((a.coarse/'verification.json').read_text())['status']=='PASS'
 # chenyi9: decision start — never combine old fixed-grid and corrected profiles.
 for r in rows(a.coarse/'operator_costs.csv'):
  if r['architecture']=='Tessera-8' and r.get('packing_policy')!=part.PACKING_POLICY:
   raise ValueError('Coarse Tessera profiles predate Algorithm 2; generate corrected profiles first')
 # chenyi9: decision end
 init();checks=0
 for core in (1,3,17,64,255,256):
  c=fine_chip(cores=core)
  for dims in ((1,5,33,67),(32,49,1,9)):
   result=base.profile_job(('matrix','test',dims,'Tessera-8',core,'e2e_latency'))
   assert result[0]['time_ns']>=result[0]['sa_ns'];checks+=1
 save_json(out/'fine_allocation_checks.json',dict(status='PASS',scalar_native_cases=checks,allocation_units=UNITS))
 unique={}
 for kind,key,payload,arch,core,obj in base.jobs('qos_profiles'):unique[kind,key]=(kind,key,payload)
 jj=[(*x,'Tessera-8',core,'e2e_latency') for x in unique.values() for core in range(1,257) if core%16]
 sources=[Path(__file__),Path(base.__file__),ROOT/'neusim/configs/chips/ChipConfig.py',*list((ROOT/'neusim/npusim/backend').glob('*.py'))]
 hashes={str(x):sha(x) for x in sources}
 for x in sources:
  dst=out/'source'/x.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(x,dst);assert sha(dst)==sha(x)
 save_json(out/'configs.json',{f'{arch}/{c}':fine_chip(arch,8 if arch=='Tessera-8' else 32,c).model_dump() for arch,n in UNITS.items() for c in range(1,n+1)})
 reused=0;ran=0;t=time.monotonic()
 with (out/'operator_costs.csv').open('x',newline='') as f:
  writer=None
  for r in rows(a.coarse/'operator_costs.csv'):
   if r['architecture']=='Tessera-8':r['cores']=str(int(r['cores'])*16)
   if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
   writer.writerow(r);reused+=1
  with ProcessPoolExecutor(max_workers=a.workers,initializer=init) as pool:
   for rr in base.bounded_map(pool,base.profile_job,jj,block=a.workers*4):
    writer.writerows(rr);ran+=1
    if ran%500==0:f.flush();print('fine profiles',ran,'/',len(jj),'elapsed_s',round(time.monotonic()-t,1),flush=True)
 actual=list(rows(out/'operator_costs.csv'));assert len(actual)==len(unique)*sum(UNITS.values())
 keys={(r['kind'],r['key'],r['architecture'],int(r['cores'])) for r in actual};assert len(keys)==len(actual)
 for k in unique:
  for arch,n in UNITS.items():assert all((*k,arch,c) in keys for c in range(1,n+1))
 # Reused records represent exactly the same PE, SRAM, bandwidth and objective.
 for arch,n in UNITS.items():
  for oldcore in range(1,17):
   old=ORIGINAL_CHIP(arch,8 if arch=='Tessera-8' else 32,oldcore,'e2e_latency')
   new=fine_chip(arch,8 if arch=='Tessera-8' else 32,oldcore*(n//16))
   for attr in ('vmem_size_MB','hbm_bw_GBps','dynamic_power_sa_W','freq_GHz'):assert getattr(old,attr)==getattr(new,attr)
   assert old.tessera_parameters.get('active_pe_budget',128**2)==new.tessera_parameters['active_pe_budget']
 assert all(sha(Path(x))==h for x,h in hashes.items())
 save_json(out/'verification.json',dict(status='PASS',allocation_units=UNITS,reused_records=reused,new_simulated_records=ran,records=len(actual),coarse_source_sha256=sha(a.coarse/'operator_costs.csv'),source_sha256=hashes,output_sha256=sha(out/'operator_costs.csv'),elapsed_s=time.monotonic()-t))
 print('PASS native-grain QoS profiles',len(actual),flush=True)
if __name__=='__main__':main()
