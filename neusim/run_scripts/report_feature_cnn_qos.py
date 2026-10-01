"""Readback-checked review plots; this script never modifies the paper."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,rows,sha,save_json,save_csv
from neusim.run_scripts.run_feature_cnn_qos import INPUT,GRAINS,ARCHS,chip
BASE=ROOT/'results/tessera/20260929_feature_cnn_qos'
NAMES={'resnet50':'ResNet-50','googlenet':'GoogleNet','mobilenet_v1':'MobileNet-v1','tiny_yolo':'Tiny-YOLO','yolov3':'YOLOv3','ssd_resnet34':'SSD-ResNet-34','ssd_mobilenet_v1':'SSD-MobileNet-v1','gnmt_encoder':'GNMT encoder','gnmt_decoder':'GNMT decoder'}
COLORS={'WS':'#bfc2c7','Planaria-32':'#f0913b','SOSA':'#5cb85c','FlexSA':'#9467bd','SISA':'#8c564b','Tessera-8':'#3b7cb5'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
def gm(v):return math.exp(math.fsum(math.log(x) for x in v)/len(v))
def label(arch):return {'Tessera-8':'Tessera','Planaria-32':'Planaria'}.get(arch,arch)
def savefig(out,name,fig):
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight');fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');plt.close(fig)
def validate_run(run):
 v=json.loads((run/'verification.json').read_text());assert v['status']=='PASS'
 assert sha(run/'operator_costs.csv')==v['output_sha256'];return v

def cnn(out,run):
 validate_run(run);source=list(rows(run/'operator_costs.csv'));layers=list(rows(INPUT/'cnn_layers.csv'))
 by={(r['kind'],r['key'],r['architecture']):r for r in source}
 raw=[];totals={}
 fields=('time_ns','energy_J','sa_ns','useful_macs','charged_sa_macs','hbm_bytes','sram_bytes','hbm_ns','vu_ns')
 for net in NAMES:
  for arch in ARCHS:
   acc={k:[] for k in fields};n=0
   for r in layers:
    if r['network']!=net:continue
    key='x'.join(r[k] for k in (('B','M','N','K') if r['kind']=='matrix' else ('input_bytes','output_bytes','vector_ops')))
    v=by[r['kind'],key,arch];n+=1
    for f in fields:acc[f].append(float(v[f]))
    raw.append(dict(network=net,layer=r['layer'],architecture=arch,source_type=r['source_type'],**{k:v[k] for k in fields}))
   totals[net,arch]=dict(network=net,workload=NAMES[net],architecture=arch,layers=n,**{k:math.fsum(v) for k,v in acc.items()})
   assert totals[net,arch]['useful_macs']==sum(int(r['useful_macs']) for r in layers if r['network']==net)
 for (net,arch),r in totals.items():
  r['e2e_speedup_over_ws']=totals[net,'WS']['time_ns']/r['time_ns'];r['speedup_over_planaria']=totals[net,'Planaria-32']['time_ns']/r['time_ns']
  r['edp_J_s']=r['energy_J']*r['time_ns']/1e9;r['array_utilization']=r['useful_macs']/(128**2*r['sa_ns'])
 save_csv(out/'cnn_layer_costs.csv',raw);save_csv(out/'cnn_totals.csv',list(totals.values()))
 # Independent readback sums from expanded layer table.
 expanded=list(rows(out/'cnn_layer_costs.csv'))
 for r in rows(out/'cnn_totals.csv'):
  for f in ('time_ns','energy_J'):
   gold=math.fsum(float(x[f]) for x in expanded if x['network']==r['network'] and x['architecture']==r['architecture']);assert gold==float(r[f])
 order=sorted(NAMES,key=lambda n:totals[n,'Tessera-8']['e2e_speedup_over_ws']);x=np.arange(len(order)+1);bar=0.16
 fig,ax=plt.subplots(figsize=(12,4));summ=[]
 for j,arch in enumerate(ARCHS[1:]):
  vals=[totals[n,arch]['e2e_speedup_over_ws'] for n in order];avg=gm(vals)
  ax.bar(x+(j-2)*bar,vals+[avg],bar,label=label(arch),color=COLORS[arch],edgecolor='0.3',linewidth=.4)
  summ.append(dict(architecture=arch,dnn_geomean_speedup_ws=avg,cnn_geomean_speedup_ws=gm([totals[n,arch]['e2e_speedup_over_ws'] for n in order if not n.startswith('gnmt')]),dnn_geomean_speedup_planaria=gm([totals[n,arch]['speedup_over_planaria'] for n in order]),min_speedup_ws=min(vals),max_speedup_ws=max(vals)))
 ax.axhline(1,color='0.4',lw=.8);ax.set_yscale('log');ax.set_ylabel('E2E speedup over mono WS');ax.set_xticks(x,[NAMES[n] for n in order]+['Geomean'],rotation=28,ha='right');ax.legend(ncol=5,loc='upper center',bbox_to_anchor=(.5,1.18));ax.grid(axis='y',ls=':',alpha=.3);ax.set_axisbelow(True)
 savefig(out,'fig10_cnn_e2e',fig);save_csv(out/'cnn_summary.csv',summ)
 return dict(source=str(run),source_sha256=sha(run/'operator_costs.csv'),rows=len(totals),layer_rows=len(raw),summary=summ)

def array(out,run):
 validate_run(run);by={};opt={}
 for r in rows(run/'operator_costs.csv'):
  k=int(r['shape_id']);g=int(r['grain']);v=float(r['array_edp_J_ns']);by[k,g]=r;opt[k]=min(opt.get(k,math.inf),v)
 weights=defaultdict(Counter)
 for r in rows(INPUT/'matrix_invocations.csv'):weights[r['workload_id']][int(r['shape_id'])]+=int(r['repeat'])
 # Original Fig. 9 weights layers equally. CNN suite excludes the two GNMT parts.
 for wid,ww in list(weights.items()):
  if wid.startswith('cnn_'):
   weights['dnn_suite'].update(ww)
   if not wid.startswith('cnn_gnmt'):weights['cnn_suite'].update(ww)
 names={r['workload_id']:r['workload'] for r in rows(INPUT/'feature_metrics.csv')};names.update({'cnn_'+k:v for k,v in NAMES.items()});names.update(cnn_suite='CNN suite (7 networks)',dnn_suite='DNN suite (9 networks)')
 data=[];choices=[]
 for wid,ww in weights.items():
  count=sum(ww.values())
  histogram=Counter()
  for k,w in ww.items():histogram[min(GRAINS,key=lambda g:(float(by[k,g]['array_edp_J_ns']),g))]+=w
  for g in GRAINS:
   ll=[(math.log(float(by[k,g]['array_edp_J_ns'])/opt[k]),w) for k,w in ww.items()]
   ratio=math.exp(math.fsum(log*w for log,w in ll)/count)
   ts=sum(int(by[k,g]['sa_ns'])*w for k,w in ww.items());energy=math.fsum(float(by[k,g]['array_energy_J'])*w for k,w in ww.items());macs=sum(int(by[k,g]['useful_macs'])*w for k,w in ww.items())
   data.append(dict(workload_id=wid,workload=names[wid],grain=g,invocations=count,normalized_geomean_array_edp=ratio,overhead_pct=100*(ratio-1),array_ns=ts,array_energy_J=energy,aggregate_array_edp_J_s=energy*ts/1e9,array_utilization=macs/(128**2*ts)))
   choices.append(dict(workload_id=wid,workload=names[wid],grain=g,best_grain_invocations=histogram[g],invocations=count,best_grain_fraction=histogram[g]/count))
 save_csv(out/'granularity_summary.csv',data);save_csv(out/'best_grain_shares.csv',choices)
 save_json(out/'actual_array_configs.json',{str(g):chip(grain=g,objective='array_edp').model_dump() for g in GRAINS})
 lookup={(r['workload_id'],r['grain']):r for r in data}
 # Readback reconstruct one ratio per group with a separate explicit product of logs.
 for wid,ww in weights.items():
  g=8;gold=math.exp(sum(w*(math.log(float(by[k,g]['array_edp_J_ns']))-math.log(opt[k])) for k,w in ww.items())/sum(ww.values()))
  assert math.isclose(lookup[wid,g]['normalized_geomean_array_edp'],gold,rel_tol=2e-12)
 def bars(ax,wid,range_nets=False):
  vals=[lookup[wid,g]['normalized_geomean_array_edp'] for g in GRAINS];yy=np.arange(len(GRAINS))
  ax.barh(yy,vals,color=['#3b7cb5' if g==8 else '#f0913b' for g in GRAINS],height=.65)
  right=vals[:]
  if range_nets:
   lo=[min(lookup['cnn_'+n,g]['normalized_geomean_array_edp'] for n in NAMES if not n.startswith('gnmt')) for g in GRAINS]
   hi=[max(lookup['cnn_'+n,g]['normalized_geomean_array_edp'] for n in NAMES if not n.startswith('gnmt')) for g in GRAINS]
   ax.errorbar(vals,yy,xerr=[np.array(vals)-lo,np.array(hi)-vals],fmt='none',ecolor='0.2',capsize=2);right=hi
  for y,v,xval in zip(yy,vals,right):ax.text(xval*1.025,y,f'{v:.2f}',va='center',fontsize=8)
  ax.axvline(1,color='0.4',ls='--',lw=.7);ax.set_yticks(yy,[str(g) for g in GRAINS]);ax.invert_yaxis();ax.set_title(names[wid],fontsize=10);ax.set_xlabel('Array EDP / per-GEMM optimum',fontsize=9);ax.set_ylabel('Min. grain',fontsize=9);ax.set_xlim(0,max(right)*1.2);ax.grid(axis='x',ls=':',alpha=.3);ax.set_axisbelow(True)
 fig,ax=plt.subplots(figsize=(7,3.8));bars(ax,'cnn_suite',True);savefig(out,'fig9_cnn_array_edp',fig)
 llm=[r['workload_id'] for r in rows(INPUT/'feature_metrics.csv')];fig,axes=plt.subplots(4,3,figsize=(15,14))
 for ax,wid in zip(axes.flat,llm):bars(ax,wid)
 fig.tight_layout();savefig(out,'fig9_llm_array_edp',fig)
 return dict(source=str(run),source_sha256=sha(run/'operator_costs.csv'),shapes=len(opt),records=len(by),groups=len(weights),selected_grain8={names[wid]:lookup[wid,8]['normalized_geomean_array_edp'] for wid in ['cnn_suite','dnn_suite',*llm]},best_grains={names[wid]:min(GRAINS,key=lambda g:lookup[wid,g]['normalized_geomean_array_edp']) for wid in ['cnn_suite',*llm]})

def qos(out,run):
 v=json.loads((run/'verification.json').read_text());assert v['status']=='PASS';assert sha(run/'qps.csv')==v['artifacts']['qps.csv']
 rr=list(rows(run/'qps.csv'));by={(r['workload'],r['qos'],r['architecture']):r for r in rr};summary=[]
 for w in 'ABC':
  for q in 'SMH':
   p,t=(by[w,q,arch] for arch in ('Planaria-32','Tessera-8'));pv,tv=float(p['qps']),float(t['qps'])
   pcap,tcap=p['search_capped']=='True',t['search_capped']=='True'
   ratio=tv/pv if pv and not (pcap and tcap) else ''
   relation='both_capped' if pcap and tcap else ('unattainable' if not pv else ('lower_bound' if tcap else ('upper_bound' if pcap else 'estimated_ratio')))
   summary.append(dict(workload=w,qos=q,planaria_qps=pv,tessera_qps=tv,tessera_over_planaria=ratio,ratio_relation=relation,planaria_capped=p['search_capped'],tessera_capped=t['search_capped'],planaria_sla_attainable=p['sla_attainable'],tessera_sla_attainable=t['sla_attainable']))
 save_csv(out/'qos_comparison.csv',summary)
 fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(summary));width=.35
 for offset,arch,key in [(-width/2,'Planaria-32','planaria'),(width/2,'Tessera-8','tessera')]:
  vals=[r[key+'_qps'] for r in summary];ax.bar(x+offset,vals,width,label=label(arch),color=COLORS[arch],edgecolor='0.3')
  for xx,r in zip(x+offset,summary):
   if r[key+'_capped']=='True':ax.text(xx,r[key+'_qps']*1.05,'≥',ha='center')
   if not r[key+'_qps'] and key=='planaria':ax.text(xx-offset,.04,'SLA\nunmet',transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
 ax.set_yscale('log');ax.set_ylabel('Maximum offered QPS meeting SLA');ax.set_xticks(x,[r['workload']+'-'+r['qos'] for r in summary]);ax.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.16));ax.set_ylim(top=max(r['tessera_qps'] for r in summary)*1.35);ax.grid(axis='y',ls=':',alpha=.3);ax.set_axisbelow(True);savefig(out,'fig12_planaria_policy_qos',fig)
 return dict(source=str(run),source_sha256=sha(run/'qps.csv'),comparison=summary)

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['cnn','array','qos']);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
 result={'cnn':cnn,'array':array,'qos':qos}[a.mode](out,a.run.resolve());save_json(out/'verification.json',dict(status='PASS',**result,script_sha256=sha(Path(__file__)),artifact_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
