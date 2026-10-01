"""Produce read-back checked paper data from completed arrival-aware replays."""
import argparse,json,math,os,shutil
from decimal import Decimal as D
from pathlib import Path
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,PROJECT,rows,sha,save_csv,save_json
ARCHES=('WS','Planaria-32','SOSA','FlexSA','SISA','Tessera-8')
FIELDS=('nonsquare_edp_reduction_pct','nonsquare_array_latency_reduction_pct',
 'nonsquare_extra_array_cycles_removed_pct','interconnect_edp_reduction_pct',
 'interconnect_energy_reduction_pct','skew_array_latency_reduction_pct',
 'async_grain32_edp_reduction_pct','async_grain8_edp_reduction_pct','async_grain8_vs32_edp_reduction_pct')

def write(p,s):
    with p.open('x') as f:f.write(s)
    assert p.read_text()==s

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.run=a.run.resolve();a.replay=a.replay.resolve();a.out=a.out.resolve()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-1:]);a.out.mkdir(parents=True,exist_ok=False)
    audit=json.loads((a.replay/'verification.json').read_text());assert audit['status']=='PASS'
    assert sha(a.replay/'totals.csv')==audit['artifacts']['totals.csv']
    feature=a.run/'feature_inputs_v1';fv=json.loads((feature/'verification.json').read_text());assert fv['status']=='PASS'
    assert sha(feature/'intrinsic_metrics.csv')==fv['artifacts']['intrinsic_metrics.csv']
    manifest=json.loads((a.run/'main_costs/manifest.json').read_text());high=max(manifest['bandwidth_bytes_per_second'])
    configs=json.loads((a.run/'main_costs/configs.json').read_text());cfg=configs['Tessera-8@'+str(high)]
    name_source=PROJECT/'Tessera-HPCA2027-revision/fig/plotting/data/neusim/workload_sources.csv'
    names={r['workload_id']:r['workload'] for r in rows(name_source)}
    values={(r['workload_id'],r['architecture'],int(r['bandwidth_bytes_per_second'])):r for r in rows(a.replay/'totals.csv')}
    native={(r['workload_id'],r['architecture']):r for r in rows(a.run/'main_costs/totals.csv') if int(r['bandwidth_bytes_per_second'])==high}
    intrinsic={r['workload_id']:r for r in rows(feature/'intrinsic_metrics.csv')}
    general=[];features=[]
    def reduction(a,b):return D(100)*(1-D(str(a))/D(str(b)))
    for wid,name in names.items():
        for bandwidth in manifest['bandwidth_bytes_per_second']:
            reference=values[wid,'WS',bandwidth]
            for arch in ARCHES:
                row=values[wid,arch,bandwidth]
                ratio=D(reference['edp_J_s'])/D(row['edp_J_s'])
                independent=(D(reference['system_energy_J'])*D(reference['time_ns']))/(D(row['system_energy_J'])*D(row['time_ns']))
                assert abs(ratio-independent)<=abs(ratio)*D('2e-12')
                general.append(dict(workload_id=wid,workload=name,architecture=arch,
                  display_name='Tessera' if arch=='Tessera-8' else 'Planaria' if arch=='Planaria-32' else arch,
                  bandwidth_bytes_per_second=bandwidth,edp_J_s=row['edp_J_s'],ws_edp_J_s=reference['edp_J_s'],edp_gain_vs_ws=ratio))
        f,s,n,c8,c32,g32=(values[wid,arch,high] for arch in ('Tessera-8','Tessera-8-square','Tessera-8-independent_noskew','Tessera-8-rounds','Tessera-32-rounds','Tessera-32'))
        nf,ns=native[wid,'Tessera-8'],native[wid,'Tessera-8-square'];i=intrinsic[wid]
        ideal=D(nf['useful_macs'])/(D(cfg['sa_dim'])**2*D(str(cfg['freq_GHz'])))
        feature_row=dict(workload_id=wid,workload=name,
          nonsquare_edp_reduction_pct=reduction(f['edp_J_s'],s['edp_J_s']),
          nonsquare_array_latency_reduction_pct=reduction(nf['sa_ns'],ns['sa_ns']),
          nonsquare_extra_array_cycles_removed_pct=D(100)*(D(ns['sa_ns'])-D(nf['sa_ns']))/(D(ns['sa_ns'])-ideal),
          interconnect_edp_reduction_pct=reduction(f['edp_J_s'],n['edp_J_s']),
          interconnect_energy_reduction_pct=reduction(f['energy_J'],n['energy_J']),
          skew_array_latency_reduction_pct=reduction(i['full_sa_cycles'],i['skew_sa_cycles']),
          async_grain32_edp_reduction_pct=reduction(g32['edp_J_s'],c32['edp_J_s']),
          async_grain8_edp_reduction_pct=reduction(f['edp_J_s'],c8['edp_J_s']),
          async_grain8_vs32_edp_reduction_pct=reduction(f['edp_J_s'],g32['edp_J_s']),
          nonsquare_selection_pct=D(100)*D(i['rectangular_calls'])/D(i['matrix_calls']),
          rectangular_operator_invocations=i['rectangular_calls'],matrix_operator_invocations=i['matrix_calls'],
          ideal_sa_ns=ideal,full_sa_ns=nf['sa_ns'],square_sa_ns=ns['sa_ns'],
          full_edp_J_s=f['edp_J_s'],square_edp_J_s=s['edp_J_s'],independent_edp_J_s=n['edp_J_s'],
          grain8_rounds_edp_J_s=c8['edp_J_s'],grain32_rounds_edp_J_s=c32['edp_J_s'],grain32_async_edp_J_s=g32['edp_J_s'],
          full_energy_J=f['energy_J'],independent_energy_J=n['energy_J'],
          array_active_utilization=f['array_active_utilization'],e2e_array_utilization=f['e2e_array_utilization'],idle_ns=f.get('idle_ns','0'))
        features.append(feature_row)
    save_csv(a.out/'general_edp.csv',general);save_csv(a.out/'ablation_contribution.csv',features)
    lines=['% Generated from verified native costs and arrival replay; do not hand-edit.']
    for r in features:lines.append(r['workload']+' & '+' & '.join(f'{r[k]:.2f}\\%' for k in FIELDS)+r' \\')
    body='\n'.join(lines)+'\n';write(a.out/'ablation_rows.tex',body)
    table=r'''\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccccc@{}}
\toprule
\multirow{2}{*}{Workload} & \multicolumn{3}{c}{Rectangle support} & \multicolumn{2}{c}{Interconnection} & Skew & \multicolumn{3}{c}{Cross-round packing} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}\cmidrule(l){8-10}
 & \shortstack{E2E EDP\\reduction} & \shortstack{Array time\\reduction} & \shortstack{Extra cycles\\removed}
 & \shortstack{E2E EDP\\reduction} & \shortstack{Energy\\reduction} & \shortstack{Array time\\reduction}
 & \shortstack{32: off/on\\EDP reduction} & \shortstack{8: off/on\\EDP reduction} & \shortstack{Both on: 32/8\\EDP reduction} \\
\midrule
'''+body+'\\bottomrule\n\\end{tabular*}\n'
    write(a.out/'ablation_table.tex',table)
    for row,line in zip(rows(a.out/'ablation_contribution.csv'),lines[1:]):
        assert [x.split(r'\%')[0] for x in line.split(' & ')[1:]]==[f'{D(row[k]):.2f}' for k in FIELDS]
    def gains(arch):return [float(values[w,arch,high]['edp_J_s'])/float(values[w,'Tessera-8',high]['edp_J_s']) for w in names]
    ws,planaria=gains('WS'),gains('Planaria-32');by={r['workload_id']:r for r in features}
    def gm(v):return math.exp(math.fsum(math.log(x) for x in v)/len(v))
    macros=dict(NeuWSMin=min(ws),NeuWSMax=max(ws),NeuWSGM=gm(ws),NeuPlanariaMax=max(planaria),NeuPlanariaGM=gm(planaria),
      NeuGQAArray=by['gqa_decode']['nonsquare_array_latency_reduction_pct'],NeuMQAArray=by['single_kv_decode']['nonsquare_array_latency_reduction_pct'],
      NeuRectEDPGM=100*(1-gm([float(1-r['nonsquare_edp_reduction_pct']/100) for r in features])),
      NeuHotpotExtra=by['epic_hotpotqa']['nonsquare_extra_array_cycles_removed_pct'],NeuMultiExtra=by['epic_multi_news']['nonsquare_extra_array_cycles_removed_pct'],
      NeuHotpotArray=by['epic_hotpotqa']['nonsquare_array_latency_reduction_pct'],NeuMultiArray=by['epic_multi_news']['nonsquare_array_latency_reduction_pct'])
    for prefix,field in [('NeuRect','nonsquare_selection_pct'),('NeuInter','interconnect_edp_reduction_pct'),('NeuRectEDP','nonsquare_edp_reduction_pct'),('NeuInterEnergy','interconnect_energy_reduction_pct'),('NeuSkewArray','skew_array_latency_reduction_pct')]:
        macros[prefix+'Min']=min(r[field] for r in features);macros[prefix+'Max']=max(r[field] for r in features)
    formatted={k:f'{v:.2f}' for k,v in macros.items()}
    formatted.update(NeuFrequency=f"{cfg['freq_GHz']:g}",NeuSRAM=f"{cfg['vmem_size_MB']:g}",NeuHBMLatency=str(cfg['hbm_latency_ns']),
      NeuHBMFour=f'{high/1e12:g}',NeuArrayPJ=f"{manifest['array_energy_pj_per_op']['Tessera-8']:g}",
      NeuPlanariaPJ=f"{manifest['array_energy_pj_per_op']['Planaria-32']:g}",NeuBandwidthPoints=str(len(manifest['bandwidth_bytes_per_second'])))
    write(a.out/'numbers.tex','% Generated from the verified arrival-preserving replay.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in formatted.items()))
    shutil.copyfile(name_source,a.out/'workload_sources.csv');assert sha(name_source)==sha(a.out/'workload_sources.csv')
    save_json(a.out/'summary.json',dict(hbm4_geomean_edp_gain_ws=gm(ws),hbm4_geomean_edp_gain_planaria=gm(planaria),fields=list(FIELDS),macros=formatted))
    refs=[Path(__file__),a.replay/'verification.json',a.replay/'totals.csv',feature/'verification.json',feature/'intrinsic_metrics.csv',name_source,a.run/'main_costs/manifest.json']
    save_json(a.out/'verification.json',dict(status='PASS',source_sha256={str(p):sha(p) for p in refs},
      artifact_sha256={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},table_fields=list(FIELDS),
      accounting='EDP and total energy use original request arrivals including idle intervals. Array/skew columns retain summed intrinsic native array times. Cross-round controls hold the native EDP-selected map fixed at each grain.'))
if __name__=='__main__':main()
