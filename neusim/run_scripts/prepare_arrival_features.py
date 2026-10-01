"""Derive matched-map skew cycles and rectangle selection from fresh profiles."""
import argparse,json,math,os,sqlite3
from collections import Counter,defaultdict
from pathlib import Path
from neusim.run_scripts.prepare_feature_cnn_qos import rows,sha,save_json,save_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-1:]);a.out.mkdir(parents=True,exist_ok=False)
    source=a.run/'main_costs';manifest=json.loads((source/'manifest.json').read_text());assert manifest['status']=='PASS'
    base=Path(manifest['base'])/'inputs_snapshot';high=max(manifest['bandwidth_bytes_per_second'])
    cfg=json.loads((source/'configs.json').read_text())['Tessera-8@'+str(high)]
    totals={(r['workload_id'],r['architecture']):r for r in rows(source/'totals.csv') if int(r['bandwidth_bytes_per_second'])==high}
    groups={(r['dataset'],r['cohort']):r['workload_id'] for r in totals.values()}
    weights=defaultdict(Counter)
    for r in rows(source/'invocation_weights.csv'):weights[r['kind'],int(r['operator_id'])][groups[r['dataset'],r['cohort']]]+=int(r['repeat'])
    shapes={}
    for r in rows(base/'inputs/workloads.csv'):shapes['shape',int(r['shape_id'])]=(1,*(int(r[k]) for k in ('M','N','K')))
    metadata=json.loads((a.run/'trace_programs_v2/operator_metadata.json').read_text())
    for oid,meta in metadata.items():
        if meta['opcode']=='Einsum':
            aa,bb=meta['inputs'];shapes['graph',int(oid)]=(math.prod(aa[:-2]),aa[-2],bb[-1],aa[-1])
    db=sqlite3.connect('file:'+str(a.run/'cost_store_v1/costs.sqlite')+'?mode=ro',uri=True)
    acc=defaultdict(Counter);operators=[]
    for kind,oid,payload in db.execute('select kind,oid,payload from costs where arch=? and bw=?',('Tessera-8',high)):
        r=json.loads(payload);mapping=json.loads(r['mapping_json']);sa=int(r['sa_ns']);extra=0;rect=False;phase_sa=0
        if int(r['useful_macs']):
            if 'geometry' in mapping:
                phases=[dict(dims=shapes[kind,oid],copies=1,geometry=mapping['geometry'],memory_tile=mapping['memory_tile'],sa_ns=sa)]
            else:
                phases=[dict(x,dims=(1,x['M'],x['N'],x['K'])) for x in mapping['phase_plans']]
            for phase in phases:
                b,m,n,k=phase['dims'];mt,nt,kt=phase['memory_tile'];h,w=phase['geometry'][-2:]
                # chenyi9: decision start -- count all residual SRAM tile executions.
                tiles=b*math.ceil(m/mt)*math.ceil(n/nt)*math.ceil(k/kt)
                # chenyi9: decision end
                extra+=phase['copies']*tiles*(h-1)
                phase_sa+=phase['copies']*phase['sa_ns'];rect|=h!=w
            assert phase_sa==sa
        row=dict(kind=kind,operator_id=oid,full_sa_cycles=sa*cfg['freq_GHz'],restored_skew_cycles=extra,
          skew_sa_cycles=sa*cfg['freq_GHz']+extra,useful_macs=int(r['useful_macs']),rectangular=int(rect))
        operators.append(row)
        for wid,count in weights[kind,oid].items():
            for field in ('full_sa_cycles','restored_skew_cycles','skew_sa_cycles','useful_macs'):acc[wid][field]+=row[field]*count
            if int(r['useful_macs']):acc[wid]['matrix_calls']+=count;acc[wid]['rectangular_calls']+=int(rect)*count
    db.close();save_csv(a.out/'skew_operator_cycles.csv',operators)
    metrics=[]
    for wid,c in acc.items():
        source_total=totals[wid,'Tessera-8']
        assert c['full_sa_cycles']==int(source_total['sa_ns'])*cfg['freq_GHz']
        assert c['useful_macs']==int(source_total['useful_macs'])
        metrics.append(dict(workload_id=wid,**c))
    save_csv(a.out/'intrinsic_metrics.csv',metrics)
    # Independent read-back aggregation uses retained operator rows and weights.
    checked=defaultdict(Counter)
    for r in rows(a.out/'skew_operator_cycles.csv'):
        for wid,count in weights[r['kind'],int(r['operator_id'])].items():
            checked[wid]['full']+=float(r['full_sa_cycles'])*count;checked[wid]['skew']+=float(r['skew_sa_cycles'])*count
    for r in rows(a.out/'intrinsic_metrics.csv'):
        assert float(r['full_sa_cycles'])==checked[r['workload_id']]['full']
        assert float(r['skew_sa_cycles'])==checked[r['workload_id']]['skew']
    refs=[Path(__file__),source/'manifest.json',source/'configs.json',source/'invocation_weights.csv',source/'totals.csv',a.run/'trace_programs_v2/verification.json',a.run/'cost_store_v1/verification.json']
    save_json(a.out/'verification.json',dict(status='PASS',operators=len(operators),workloads=len(metrics),
      accounting='Trace-weighted intrinsic native array-cycle sum. Matched skew adds H-1 per SRAM-tile execution; physical H != W defines rectangular selection.',
      source_sha256={str(p):sha(p) for p in refs},artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print('PASS intrinsic feature derivation',flush=True)
if __name__=='__main__':main()
