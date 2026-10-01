"""Derive why invocation-weighted array EDP differs between trace workloads."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,rows,save_csv,save_json,sha
BASE=ROOT/'results/tessera/20260929_feature_cnn_qos'

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
 shapes={r['shape_id']:r for r in rows(BASE/'inputs_v1/matrix_shapes.csv')}
 by={(r['shape_id'],r['grain']):r for r in rows(BASE/'granularity_full_v1/operator_costs.csv') if r['grain'] in ('2','8')}
 acc=defaultdict(lambda:defaultdict(float));samples=defaultdict(list)
 for r in rows(BASE/'inputs_v1/matrix_invocations.csv'):
  wid,k,w=r['workload_id'],r['shape_id'],int(r['repeat']);dims=shapes[k];cat='M=1' if dims['M']=='1' else 'M>1';x=acc[wid,cat];x['invocations']+=w
  s,t=by[k,'2'],by[k,'8'];ratio=float(t['array_edp_J_ns'])/float(s['array_edp_J_ns']);x['logratio']+=w*math.log(ratio)
  x['sa2_ns']+=w*int(s['sa_ns']);x['sa8_ns']+=w*int(t['sa_ns']);x['energy2_J']+=w*float(s['array_energy_J']);x['energy8_J']+=w*float(t['array_energy_J'])
  samples[wid].append(dict(shape_id=k,repeat=w,B=dims['B'],M=dims['M'],N=dims['N'],K=dims['K'],array_edp8_over2=ratio,weighted_log_ratio=w*math.log(ratio),grain2_mapping=s['mapping_json'],grain8_mapping=t['mapping_json']))
 out=[]
 for (wid,cat),r in acc.items():
  total=sum(v['invocations'] for (ww,_),v in acc.items() if ww==wid)
  out.append(dict(workload_id=wid,category=cat,invocations=int(r['invocations']),invocation_fraction=r['invocations']/total,geomean_edp8_over2=math.exp(r['logratio']/r['invocations']),array_time8_over2=r['sa8_ns']/r['sa2_ns'],array_energy8_over2=r['energy8_J']/r['energy2_J']))
 save_csv(a.out/'shape_categories.csv',out)
 save_csv(a.out/'largest_weighted_effects.csv',[dict(workload_id=wid,**r) for wid,rr in samples.items() for r in sorted(rr,key=lambda x:x['weighted_log_ratio'],reverse=True)[:3]])
 save_json(a.out/'verification.json',dict(status='PASS',script_sha256=sha(Path(__file__)),sources={str(x):sha(x) for x in [BASE/'inputs_v1/matrix_shapes.csv',BASE/'inputs_v1/matrix_invocations.csv',BASE/'granularity_full_v1/operator_costs.csv']},artifacts={x.name:sha(x) for x in a.out.iterdir() if x.is_file()}))
 print('PASS shape-category diagnostic',flush=True)
if __name__=='__main__':main()
