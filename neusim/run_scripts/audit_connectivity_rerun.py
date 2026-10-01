"""Independently recompute all weighted totals and compare saved old/new mappings."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sqlite3
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,sha,save_json,save_csv,rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--old',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    manifest=json.loads((a.run/'manifest.json').read_text());assert manifest['status']=='PASS' and not manifest['pilot']
    for name,h in manifest['output_sha256'].items():assert sha(a.run/name)==h,name
    original_weights=list(rows(a.old/'main_costs/invocation_weights.csv'))
    new_weights=list(rows(a.run/'invocation_weights.csv'))
    assert original_weights==new_weights
    configs=json.loads((a.run/'configs.json').read_text());old_configs=json.loads((a.old/'main_costs/configs.json').read_text())
    for name,cfg in configs.items():
        old=json.loads(json.dumps(old_configs[name]));new=json.loads(json.dumps(cfg))
        for c in (old,new):c['tessera_parameters'].pop('selection_bandwidths')
        assert old==new,name
    weight=defaultdict(list)
    for r in new_weights:weight[r['kind'],r['operator_id']].append(((r['dataset'],r['cohort']),int(r['repeat'])))
    db=sqlite3.connect('file:'+str((a.old/'cost_store_v1/costs.sqlite').resolve())+'?mode=ro',uri=True)
    acc=defaultdict(Counter);seen=set();changes=Counter();n=0;counter_checks=0;edp_checks=0;observations=[]
    fields=manifest['fields']
    for r in rows(a.run/'operator_costs.csv'):
        identity=(r['kind'],r['operator_id']);key=(*identity,r['architecture'])
        assert key not in seen;seen.add(key)
        bw=int(r['bandwidth_bytes_per_second'])
        old=json.loads(db.execute('select payload from costs where arch=? and bw=? and kind=? and oid=?',
            (r['architecture'],bw,r['kind'],int(r['operator_id']))).fetchone()[0])
        assert r['useful_macs']==old['useful_macs']
        mapping=json.loads(r['mapping_json']);older=json.loads(old['mapping_json'])
        c=configs[r['architecture']+'@'+str(bw)]
        assert int(r['peak_live_bytes'])<=c['vmem_size_MB']*1024**2
        assert int(r['charged_sa_macs'])>=int(r['useful_macs'])
        assert int(r['padding_sram_read_bytes'])>=0
        if 'geometry' in mapping:
            new_edp=float(r['energy_J'])*int(r['time_ns']);old_edp=float(old['energy_J'])*int(old['time_ns'])
            assert new_edp<=old_edp*(1+2e-12),(key,new_edp,old_edp)
            edp_checks+=1
            changed=any(mapping[k]!=older[k] for k in ('geometry','memory_tile'))
            changes[r['architecture']]+=changed
            if changed and len(observations)<32:
                observations.append(dict(kind=r['kind'],operator_id=r['operator_id'],architecture=r['architecture'],
                    old_edp_J_ns=old_edp,new_edp_J_ns=new_edp,old_tile=older['memory_tile'],new_tile=mapping['memory_tile']))
        for group,repeat in weight[identity]:
            for field in fields:
                value=float(r[field]) if field.endswith('_J') else int(r[field])
                acc[group,r['architecture']][field]+=repeat*value
        n+=1
    assert n==manifest['operators']*len(configs)
    totals=list(rows(a.run/'totals.csv'));assert len(totals)==24
    for r in totals:
        expected=acc[(r['dataset'],r['cohort']),r['architecture']]
        for field in fields:
            if field.endswith('_J'):assert math.isclose(expected[field],float(r[field]),rel_tol=2e-11),(r['workload_id'],field)
            else:assert expected[field]==int(r[field]),(r['workload_id'],field)
            counter_checks+=1
        assert math.isclose(float(r['edp_J_s']),float(r['energy_J'])*int(r['chips'])*int(r['time_ns'])/1e9,rel_tol=2e-12)
    refs=[a.run/'manifest.json',a.run/'totals.csv',a.run/'operator_costs.csv',a.run/'invocation_weights.csv',
          a.old/'main_costs/manifest.json',Path(__file__)]
    save_json(a.out/'verification.json',dict(status='PASS',raw_rows=n,weighted_checks=counter_checks,
        edp_nonregression_checks=edp_checks,changed_mappings=dict(changes),example_changes=observations,
        source_sha256={str(p.resolve()):sha(p) for p in refs}))
    print(json.dumps(dict(status='PASS',raw_rows=n,weighted_checks=counter_checks,
        edp_nonregression_checks=edp_checks,changed_mappings=dict(changes))))


if __name__=='__main__':main()
