"""Independent exhaustive scalar golden for new array and latency objectives."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,math
from pathlib import Path
from neusim.run_scripts.run_feature_cnn_qos import chip,scalar,one_array
from neusim.run_scripts.run_feature_qos_fine_profiles import fine_chip
from neusim.npusim.backend import tessera_joint as joint,tessera_partitioned as part,tessera_baselines as base
from neusim.npusim.backend.dvfs_power_getter import FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE as reg
from neusim.run_scripts.prepare_feature_cnn_qos import sha,save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);checked=0;winners=0
 for grain in (2,8,32,128):
  c=chip(grain=grain,objective='array_edp')
  for dims in ((1,3,5,7),(2,13,16,14)):
   b,m,n,k=dims;best=math.inf
   for g in part.geometries(m,n,k,c):
    ts=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
    for vv in zip(*ts[:3]):
     t=tuple(map(int,vv));s=scalar(dims,g,t,c);cyc=s['sa_cycles'];useful=math.prod(dims);util=useful/(cyc*c.sa_dim**2)
     eff=next(r.power_efficiency_percent for r in reg if r.activity_factor>=min(1,util))
     energy=(s['charged_sa_macs']*2*c.tessera_parameters['array_energy_pj_per_op']/1e12+c.static_power_sa_W*cyc/c.freq_GHz/1e9)*100/eff
     best=min(best,energy*cyc/c.freq_GHz);checked+=1
   selected=one_array(dims,c);assert math.isclose(best,selected['array_edp_J_ns'],rel_tol=2e-12);winners+=1
 # Check minimum native E2E latency against scalar timing/HBM for both policies.
 for arch in ('Planaria-32','Tessera-8'):
  for cores in (1,3,16) if arch=='Planaria-32' else (1,17,256):
   c=fine_chip(arch,32 if arch=='Planaria-32' else 8,cores)
   dims=(1,3,5,7);best=math.inf
   gs=base.geometries(c) if arch=='Planaria-32' else part.geometries(*dims[1:],c)
   for g in gs:
    ts=joint.factor_tiles(*dims[1:],2,g[3],c.vmem_size_MB*1024**2)
    for vv in zip(*ts[:3]):
     s=scalar(dims,g,tuple(map(int,vv)),c)
     hbm=max(c.hbm_latency_ns,math.ceil(s['hbm_bytes']/(c.hbm_bw_GBps*1024**3/1e9)))
     tm=max(math.ceil(s['sa_cycles']/c.freq_GHz),part.issue_ns(s['reduction_ops'],c),hbm)
     best=min(best,tm);checked+=1
   g,t=joint.select_mapping(*dims,'DT_BFLOAT16',c.model_dump_json());s=scalar(dims,g,t,c)
   tm=max(math.ceil(s['sa_cycles']/c.freq_GHz),part.issue_ns(s['reduction_ops'],c),max(c.hbm_latency_ns,math.ceil(s['hbm_bytes']/(c.hbm_bw_GBps*1024**3/1e9))))
   assert best==tm;winners+=1
 save_json(a.out/'verification.json',dict(status='PASS',scalar_candidates=checked,exhaustive_winners=winners,script_sha256=sha(Path(__file__))))
 print('PASS',checked,'scalar candidates',winners,'exhaustive winners',flush=True)
if __name__=='__main__':main()
