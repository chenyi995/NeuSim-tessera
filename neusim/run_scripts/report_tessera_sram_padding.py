"""Audit padded-energy E2E results and derive the current figure and tables."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import getpass
import json
import math
import os
from pathlib import Path
import shutil
import traceback
import numpy as np
from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha
from neusim.run_scripts.report_tessera_ws import MAIN, close, key, label, render, save_csv, save_json
from neusim.run_scripts.run_tessera_joint import FIELDS as CORE_FIELDS

FIELDS=(*CORE_FIELDS,'charged_sa_macs','padding_macs','padding_sram_read_bytes')
ABLATIONS=('Tessera-8-square','Tessera-8-independent_noskew')
INTEGER_FIELDS=('time_ns','useful_macs','sa_ns','charged_sa_macs','padding_macs','padding_sram_read_bytes','hbm_bytes','sram_bytes')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run=a.run.resolve();out=a.out.resolve()
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
        assert manifest['sa_energy_accounting']==manifest['sram_read_accounting']=='padded_tiles'
        expected={arch:(4.41 if arch=='WS' or arch.startswith('Tessera') else 4.54) for arch in (*MAIN,*ABLATIONS)}
        assert manifest['array_energy_pj_per_op']==expected
        high=max(manifest['bandwidth_bytes_per_second'])
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
        actual=defaultdict(Counter)
        for r in rows(run/'invocation_weights.csv'):
            actual[r['kind'],int(r['operator_id'])][r['dataset'],r['cohort']]+=int(r['repeat'])
        assert dict(actual)==dict(weights)
        configs=json.loads((run/'configs.json').read_text());cfg_keys={}
        for name,c in configs.items():
            arch,bw=name.rsplit('@',1);cfg_keys[arch,int(bw)]=len(cfg_keys)
            close(c['dynamic_power_W_per_SA']/(2*c['sa_dim']**2*c['freq_GHz']*1e9)*1e12,expected[arch])
            cp=c['tessera_parameters']
            assert cp['array_energy_pj_per_op']==expected[arch] and cp['mapping_policy']=='joint_edp'
            assert cp['sa_energy_accounting']==cp['sram_read_accounting']=='padded_tiles'
            assert not c['use_vu_for_small_matmul'] and not c['enable_dvfs']
        assert set(cfg_keys)=={(arch,bw) for arch in MAIN for bw in manifest['bandwidth_bytes_per_second']}|{(arch,high) for arch in ABLATIONS}
        group_keys={(g['dataset'],g['cohort']):i for i,g in enumerate(groups)}
        sums=np.zeros((len(groups),len(cfg_keys),len(FIELDS)),dtype=np.float64)
        integers=defaultdict(Counter);identities=defaultdict(set);macs={}
        for r in rows(run/'operator_costs.csv'):
            identity=r['kind'],int(r['operator_id']);cfg=r['architecture'],int(r['bandwidth_bytes_per_second'])
            assert cfg in cfg_keys and cfg not in identities[identity]
            identities[identity].add(cfg);useful=int(r['useful_macs'])
            assert identity not in macs or macs[identity]==useful
            macs[identity]=useful
            if identity[0]=='shape':assert useful==math.prod(shapes[identity[1]])
            c=configs[f'{cfg[0]}@{cfg[1]}']
            assert 0<=int(r['peak_live_bytes'])<=c['vmem_size_MB']*1024**2
            assert int(r['charged_sa_macs'])-useful==int(r['padding_macs'])>=0
            assert 0<=int(r['padding_sram_read_bytes'])<=int(r['sram_bytes'])
            assert int(r['charged_sa_macs'])<=float(r['sa_ns'])*c['num_sa']*c['sa_dim']**2*c['freq_GHz']
            mapping=json.loads(r['mapping_json'])
            if useful:assert mapping['sa_energy_accounting']==mapping['sram_read_accounting']=='padded_tiles'
            close(sum(float(r[f]) for f in FIELDS if f.startswith(('static_energy_','dynamic_energy_'))),r['energy_J'])
            values=np.array([float(r[f]) for f in FIELDS])
            for group,count in weights[identity].items():
                sums[group_keys[group],cfg_keys[cfg]]+=count*values
                for f in INTEGER_FIELDS:integers[group,cfg][f]+=count*int(r[f])
            result['raw_rows_checked']+=1
            if result['raw_rows_checked']%250000==0:print('Audited rows',result['raw_rows_checked'],flush=True)
        assert len(identities)==manifest['operators']==len(weights)
        assert all(cfgs==set(cfg_keys) for cfgs in identities.values())
        for r in batches:
            assert sum(macs['graph',oid]*count for oid,count in r['operators'])==r['expected_macs']
            result['batches_checked']+=1
        totals=list(rows(run/'totals.csv'))
        for r in totals:
            group=r['dataset'],r['cohort'];cfg=r['architecture'],int(r['bandwidth_bytes_per_second'])
            for i,f in enumerate(FIELDS):
                close(r[f],sums[group_keys[group],cfg_keys[cfg],i]);result['weighted_fields_checked']+=1
            for f in INTEGER_FIELDS:assert int(r[f])==integers[group,cfg][f],(group,cfg,f)
            close(r['edp_J_s'],float(r['energy_J'])*int(r['chips'])*int(r['time_ns'])*1e-9)
        lookup={key(r):r for r in totals};assert len(lookup)==len(groups)*len(cfg_keys)
        normalized=[];utils=[]
        for r in totals:
            wid,arch,bw=key(r);c=configs[f'{arch}@{bw}'];ws=lookup[wid,'WS',bw]
            if arch in MAIN:
                normalized.append(dict(**r,display_name=label(arch),baseline_architecture='WS',baseline_edp_J_s=ws['edp_J_s'],
                    edp_gain_vs_ws=float(ws['edp_J_s'])/float(r['edp_J_s']),array_energy_pj_per_op=expected[arch],
                    sa_energy_accounting='padded_tiles',sram_read_accounting='padded_tiles'))
            capacity=c['num_sa']*c['sa_dim']**2*c['freq_GHz']
            utils.append(dict(workload_id=wid,workload=r['workload'],architecture=arch,bandwidth_bytes_per_second=bw,
                useful_sa_macs=int(r['useful_macs']),sa_time_ns=int(r['sa_ns']),e2e_time_ns=int(r['time_ns']),
                physical_PEs_per_chip=c['num_sa']*c['sa_dim']**2,frequency_GHz=c['freq_GHz'],
                array_utilization_pct=100*int(r['useful_macs'])/(capacity*int(r['sa_ns'])),
                e2e_array_utilization_pct=100*int(r['useful_macs'])/(capacity*int(r['time_ns']))))
        save_csv(out/'general_edp.csv',normalized);save_csv(out/'array_utilization.csv',utils)
        for r in rows(out/'general_edp.csv'):
            wid,arch,bw=key(r);ws=lookup[wid,'WS',bw];d=lookup[wid,arch,bw]
            close(r['edp_gain_vs_ws'],float(ws['energy_J'])*int(ws['chips'])*int(ws['time_ns'])/(float(d['energy_J'])*int(d['chips'])*int(d['time_ns'])))
        for r in rows(out/'array_utilization.csv'):
            raw=lookup[key(r)];c=configs[f"{r['architecture']}@{r['bandwidth_bytes_per_second']}"]
            capacity=Decimal(c['num_sa'])*Decimal(c['sa_dim'])**2*Decimal(str(c['freq_GHz']))
            for f,t in (('array_utilization_pct','sa_ns'),('e2e_array_utilization_pct','time_ns')):
                close(r[f],100*Decimal(raw['useful_macs'])/(capacity*Decimal(raw[t])))
            assert 0<float(r['e2e_array_utilization_pct'])<=float(r['array_utilization_pct'])<=100
        normal={key(r):r for r in normalized};util={key(r):r for r in utils}
        headline=[];utilhead=[];ablation=[];breakdown=[]
        for g in groups:
            wid=g['workload_id'];row=dict(workload_id=wid,workload=g['name']);uu=dict(row);aa=dict(row)
            full=lookup[wid,'Tessera-8',high];aa['full_edp_J_s']=full['edp_J_s']
            for variant,field in zip(ABLATIONS,('nonsquare','interconnect')):
                off=lookup[wid,variant,high]
                aa[field+'_off_edp_J_s']=off['edp_J_s']
                aa[field+'_on_normalized_edp']=float(full['edp_J_s'])/float(off['edp_J_s'])
                aa[field+'_edp_reduction_pct']=100*(1-aa[field+'_on_normalized_edp'])
            ablation.append(aa)
            for arch in MAIN:
                row[arch+'_edp_gain_vs_ws']=normal[wid,arch,high]['edp_gain_vs_ws']
                uu[arch+'_array_utilization_pct']=util[wid,arch,high]['array_utilization_pct']
            headline.append(row);utilhead.append(uu)
            for arch in (*MAIN,*ABLATIONS):
                r=lookup[wid,arch,high]
                energy=dict(workload_id=wid,workload=g['name'],architecture=arch,system_energy_J=r['system_energy_J'],
                    array_energy_pj_per_op=expected[arch],padding_sram_read_bytes=r['padding_sram_read_bytes'],padding_macs=r['padding_macs'])
                for component in ('sa','vu','sram','hbm','ici','other'):
                    val=float(r[f'dynamic_energy_{component}_J'])+float(r[f'static_energy_{component}_J'])
                    energy[component+'_system_J']=val*int(r['chips']);energy[component+'_pct']=100*val/float(r['energy_J'])
                breakdown.append(energy)
        for name,data in (('hbm4_headline',headline),('hbm4_array_utilization',utilhead),('ablation_contribution',ablation),('energy_breakdown',breakdown)):
            save_csv(out/(name+'.csv'),data)
        for r in rows(out/'ablation_contribution.csv'):
            full=lookup[r['workload_id'],'Tessera-8',high]
            for variant,field in zip(ABLATIONS,('nonsquare','interconnect')):
                off=lookup[r['workload_id'],variant,high]
                ratio=Decimal(full['energy_J'])*Decimal(full['time_ns'])/(Decimal(off['energy_J'])*Decimal(off['time_ns']))
                close(r[field+'_on_normalized_edp'],ratio);close(r[field+'_edp_reduction_pct'],100*(1-ratio))
        rendering=render(out,groups,high,ablation_variants=())
        note=['# Current E2E EDP evaluation','',
            f'{getpass.getuser()} ruled: padding must consume SRAM read energy; rerun the comparison. All sourced workloads are E2E. Tessera denotes Tessera-8.','',
            f'Run: `{run}`. Status: PASS. Figures and tables are derived from audited NeuSim model results, not measured hardware power.','',
            'All architectures now charge both padded compute and padded SRAM A/B reads for assigned physical tiles. Zero lanes use the same coefficient as valid lanes; unassigned regions and pipeline bubbles are not charged as MACs. '
            'Padding adds dummy operand-port reads. Valid-data SRAM capacity, partial/output storage and HBM transfers retain their existing accounting; this is not a dense padded tensor allocation model. '
            'HBM reloads caused by finite SRAM reuse remain included. Ideal per-PE SRAM bandwidth remains nonblocking.','',
            'Planaria, SISA, SOSA and FlexSA use 4.54 pJ/op; Tessera and mono WS use 4.41 pJ/op. One MAC is two operations. '
            'Source: user ruling and Tessera-HPCA2027-revision/sec/09_eval.tex:192–204; SISA/SOSA/FlexSA use the requested Planaria proxy. '
            'SRAM uses the native dynamic-power/bandwidth coefficient for both reads and writes; native regulator losses and static energy remain included. '
            'HBM energy is NeuSim\'s native controller/PHY model.','',
            'Every bandwidth point jointly searches all capacity-feasible divisor SRAM tiles and architecture-legal array configurations using per-GEMM EDP including padding and HBM energy. '
            'FlashAttention retains native outer tiling and maps resident QK/PV phases with the same objective. Workload EDP is summed system energy multiplied by summed serial E2E time; per-GEMM selection is not a global workload optimum.','',
            'Main figure: `EDP_mono_WS / EDP_design`, mono WS = 1; higher is better.','',
            '[Main figure](general_edp_bandwidth.png) · [PDF](general_edp_bandwidth.pdf) · [Full main data](general_edp.csv)','',
            f'HBM4 ({high/1e12:g} TB/s) EDP gain, derived from this run:','',
            '| E2E workload | '+' | '.join(label(x) for x in MAIN)+' |','|---|'+'---:|'*len(MAIN)]
        for r in rows(out/'hbm4_headline.csv'):
            note.append('| '+r['workload']+' | '+' | '.join(f"{float(r[x+'_edp_gain_vs_ws']):.3f}×" for x in MAIN)+' |')
        note+=['','For each feature separately, disabled EDP = 1. The percentages below are `100 × (1 − EDP_full / EDP_disabled)`. '
            'Both controls retain skew-free execution and the same 4.41 pJ/op, padded compute/read policy and SRAM capacity. '
            'Square-only pads allocated regions to squares; disconnected uses independent 8×8 regions. All three are re-optimized at HBM4.','',
            '| E2E workload | Non-square: EDP reduction | Interconnection: EDP reduction |','|---|---:|---:|']
        for r in rows(out/'ablation_contribution.csv'):
            note.append(f"| {r['workload']} | {float(r['nonsquare_edp_reduction_pct']):.2f}% | {float(r['interconnect_edp_reduction_pct']):.2f}% |")
        note+=['','[Full ablation data](ablation_contribution.csv). Positive percentages mean lower EDP.','',
            '`Array utilization = useful SA MACs / (all physical PEs × frequency_GHz × summed SA time_ns)`. '
            'Padding MACs are excluded from useful work. Invocation-weighted counts and times are summed before division. '
            'Fill/drain and idle physical regions remain in the denominator; tensor-parallel chip factors cancel. '
            'The CSV additionally reports E2E utilization using total workload time.','',
            f'HBM4 ({high/1e12:g} TB/s) useful array utilization, derived from this run:','',
            '| E2E workload | '+' | '.join(label(x) for x in MAIN)+' |','|---|'+'---:|'*len(MAIN)]
        for r in rows(out/'hbm4_array_utilization.csv'):
            note.append('| '+r['workload']+' | '+' | '.join(f"{float(r[x+'_array_utilization_pct']):.2f}%" for x in MAIN)+' |')
        note+=['','[All utilization data](array_utilization.csv) · [HBM4 energy breakdown](energy_breakdown.csv) · [Independent audit](verification.json)','',
            'All current figures and tables use this run. Historical results remain in their run directories, with superseded presentation artifacts archived.','']
        text='\n'.join(note)
        with (out/'REPORT.md').open('x') as f:f.write(text)
        assert (out/'REPORT.md').read_text()==text
        for path in (Path(__file__),ROOT/'neusim/run_scripts/report_tessera_ws.py'):checked(path)
        result.update(status='PASS',source_sha256=sources,**rendering)
    except BaseException:
        result['status']='FAIL';(out/'failure.txt').write_text(traceback.format_exc());raise
    finally:save_json(out/'verification.json',result)
    save_json(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2),flush=True)


if __name__=='__main__':main()
