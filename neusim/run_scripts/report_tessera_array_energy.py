"""Audit the architecture-energy sweep and publish its mono-WS EDP curves."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import traceback
import numpy as np
from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha
from neusim.run_scripts.report_tessera_ws import MAIN, close, key, label, render, save_csv, save_json
from neusim.run_scripts.run_tessera_joint import FIELDS

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.out.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir();os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    shutil.copy2(__file__,out/Path(__file__).name)
    result=dict(status='running',raw_rows_checked=0,weighted_fields_checked=0,batches_checked=0,hash_checks=0)
    sources={}
    def checked(path):
        sources[str(path)]=sha(path);return path
    try:
        manifest=json.loads(checked(run/'manifest.json').read_text())
        assert manifest['status']=='PASS' and not manifest['pilot']
        expected_coefficients={'WS':4.41,'Tessera-8':4.41,'Planaria-32':4.54,'SOSA':4.54,'SISA':4.54,'FlexSA':4.54}
        assert manifest['array_energy_pj_per_op']==expected_coefficients
        for name,digest in manifest['output_sha256'].items():
            assert sha(checked(run/name))==digest;result['hash_checks']+=1
        for name,digest in manifest['source_sha256'].items():
            source=Path(name)
            snapshot=run/'source'/source.relative_to(ROOT) if source.is_relative_to(ROOT) else source
            actual=snapshot if snapshot.exists() else source
            assert sha(checked(actual))==digest,actual;result['hash_checks']+=1
        base=Path(manifest['base']);inputs=base/'inputs_snapshot'
        meta=json.loads(checked(inputs/'workload_metadata.json').read_text())
        groups=json.loads(checked(inputs/'workload_groups.json').read_text())
        cohorts={(r['dataset'],r['cohort']):r for r in meta['cohorts']}
        tag_groups={r['trace_tag']:k for k,r in cohorts.items() if r['trace_tag']}
        weights=defaultdict(Counter);shapes={};batches=[]
        for r in rows(checked(inputs/'inputs/workloads.csv')):
            group=r['dataset'],r['cohort']
            if cohorts[group]['trace_tag']:continue
            sid=int(r['shape_id']);shape=tuple(int(r[k]) for k in ('M','N','K'))
            assert sid not in shapes or shapes[sid]==shape
            shapes[sid]=shape;weights['shape',sid][group]+=int(r['repeat'])
        for line in checked(base/'speed/trace_graph_batches.jsonl').open():
            r=json.loads(line);batches.append(r)
            for oid,count in r['operators']:weights['graph',oid][tag_groups[r['model']]]+=count
        actual_weights=defaultdict(Counter)
        for r in rows(run/'invocation_weights.csv'):
            actual_weights[r['kind'],int(r['operator_id'])][r['dataset'],r['cohort']]+=int(r['repeat'])
        assert dict(actual_weights)==dict(weights)
        configs=json.loads((run/'configs.json').read_text());config_keys={}
        for name,c in configs.items():
            arch,bw=name.rsplit('@',1);config_keys[arch,int(bw)]=len(config_keys)
            pj=c['dynamic_power_W_per_SA']/(2*c['sa_dim']**2*c['freq_GHz']*1e9)*1e12
            close(pj,expected_coefficients[arch])
            assert c['tessera_parameters']['array_energy_pj_per_op']==expected_coefficients[arch]
            assert c['tessera_parameters']['mapping_policy']=='joint_edp'
        assert set(config_keys)=={(a,b) for a in MAIN for b in manifest['bandwidth_bytes_per_second']}
        group_keys={(g['dataset'],g['cohort']):i for i,g in enumerate(groups)}
        sums=np.zeros((len(groups),len(config_keys),len(FIELDS)),dtype=np.float64)
        integers=defaultdict(Counter);identities=defaultdict(set);macs={}
        for r in rows(run/'operator_costs.csv'):
            identity=r['kind'],int(r['operator_id']);cfg=r['architecture'],int(r['bandwidth_bytes_per_second'])
            assert cfg in config_keys and cfg not in identities[identity]
            identities[identity].add(cfg);useful=int(r['useful_macs'])
            assert identity not in macs or macs[identity]==useful
            macs[identity]=useful
            if identity[0]=='shape':assert useful==math.prod(shapes[identity[1]])
            assert 0<=int(r['peak_live_bytes'])<=configs[f'{cfg[0]}@{cfg[1]}']['vmem_size_MB']*1024**2
            close(sum(float(r[f]) for f in FIELDS if f.startswith(('static_energy_','dynamic_energy_'))),r['energy_J'])
            values=np.array([float(r[f]) for f in FIELDS])
            for group,count in weights[identity].items():
                sums[group_keys[group],config_keys[cfg]]+=count*values
                integers[group,cfg]['time_ns']+=count*int(r['time_ns'])
                integers[group,cfg]['useful_macs']+=count*useful
            result['raw_rows_checked']+=1
        assert len(identities)==manifest['operators']==len(weights)
        assert all(cfgs==set(config_keys) for cfgs in identities.values())
        for r in batches:
            assert sum(macs['graph',oid]*count for oid,count in r['operators'])==r['expected_macs']
            result['batches_checked']+=1
        totals=list(rows(run/'totals.csv'))
        for r in totals:
            group=r['dataset'],r['cohort'];cfg=r['architecture'],int(r['bandwidth_bytes_per_second'])
            for i,f in enumerate(FIELDS):
                close(r[f],sums[group_keys[group],config_keys[cfg],i]);result['weighted_fields_checked']+=1
            for f in ('time_ns','useful_macs'):assert int(r[f])==integers[group,cfg][f]
            close(r['edp_J_s'],float(r['energy_J'])*int(r['chips'])*int(r['time_ns'])*1e-9)
        lookup={key(r):r for r in totals};assert len(lookup)==len(groups)*len(config_keys)
        normalized=[]
        for r in totals:
            wid,arch,bw=key(r);ws=lookup[wid,'WS',bw]
            normalized.append(dict(**r,display_name=label(arch),baseline_architecture='WS',baseline_edp_J_s=ws['edp_J_s'],
                edp_gain_vs_ws=float(ws['edp_J_s'])/float(r['edp_J_s']),array_energy_pj_per_op=expected_coefficients[arch],
                sa_energy_accounting='useful'))
        save_csv(out/'general_edp.csv',normalized)
        for r in rows(out/'general_edp.csv'):
            wid,arch,bw=key(r);ws=lookup[wid,'WS',bw];design=lookup[wid,arch,bw]
            close(r['edp_gain_vs_ws'],float(ws['energy_J'])*int(ws['chips'])*int(ws['time_ns'])/
                  (float(design['energy_J'])*int(design['chips'])*int(design['time_ns'])))
        high=max(manifest['bandwidth_bytes_per_second']);normal={key(r):r for r in normalized}
        headline=[];breakdown=[]
        for g in groups:
            row=dict(workload_id=g['workload_id'],workload=g['name'])
            for arch in MAIN:row[arch+'_edp_gain_vs_ws']=normal[g['workload_id'],arch,high]['edp_gain_vs_ws']
            headline.append(row)
            for arch in MAIN:
                r=normal[g['workload_id'],arch,high]
                energy=dict(workload_id=g['workload_id'],workload=g['name'],architecture=arch,
                            array_energy_pj_per_op=expected_coefficients[arch],system_energy_J=r['system_energy_J'])
                for component in ('sa','vu','sram','hbm','ici','other'):
                    val=float(r[f'dynamic_energy_{component}_J'])+float(r[f'static_energy_{component}_J'])
                    energy[component+'_system_J']=val*int(r['chips'])
                    energy[component+'_pct']=100*val/float(r['energy_J'])
                breakdown.append(energy)
        save_csv(out/'hbm4_headline.csv',headline);save_csv(out/'energy_breakdown.csv',breakdown)
        render_checks=render(out,groups,high,ablation_variants=())
        note=['# E2E EDP with architecture-specific array energy','',
            f'Run: `{run}`. All numbers are derived from independently audited modeled NeuSim totals.','',
            'Planaria, SISA, SOSA and FlexSA use 4.54 pJ/op; Tessera and mono WS use 4.41 pJ/op. '
            'One MAC is two arithmetic operations. These coefficients replace the native dynamic-SA coefficient '
            'before its regulator losses. SRAM, HBM, vector and static energy retain their native models. '
            'SISA/SOSA/FlexSA use the user-requested Planaria proxy; WS/Tessera/Planaria values are stated in '
            '`Tessera-HPCA2027-revision/sec/09_eval.tex:192-204`.','',
            'Every bandwidth point re-selects legal array geometry and finite SRAM tiles by per-GEMM EDP including '
            'HBM energy. FlashAttention retains native outer tiling and jointly maps resident QK/PV phases. '
            'The timing formulas and useful-MAC accounting of the main comparison are unchanged.','',
            'Every sourced workload is E2E. Plot value is `EDP_mono_WS / EDP_design`, so mono WS = 1. '
            'EDP uses summed system energy times summed serial workload time. Tessera denotes Tessera-8. '
            'The ablation contribution table is in the [current README](../../../README.md).','',
            '[Main bandwidth figure](general_edp_bandwidth.png) · [PDF](general_edp_bandwidth.pdf)','',
            '| E2E workload | '+' | '.join(label(a) for a in MAIN)+' |','|---|'+'---:|'*len(MAIN)]
        for r in rows(out/'hbm4_headline.csv'):
            note.append('| '+r['workload']+' | '+' | '.join(f"{float(r[a+'_edp_gain_vs_ws']):.6f}" for a in MAIN)+' |')
        text='\n'.join(note)+'\n'
        with (out/'REPORT.md').open('x') as stream:stream.write(text)
        assert (out/'REPORT.md').read_text()==text
        for p in (Path(__file__),ROOT/'neusim/run_scripts/report_tessera_ws.py'):checked(p)
        result.update(status='PASS',source_sha256=sources,**render_checks)
    except BaseException:
        result['status']='FAIL';(out/'failure.txt').write_text(traceback.format_exc());raise
    finally:
        save_json(out/'verification.json',result)
    save_json(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2),flush=True)

if __name__=='__main__':main()
