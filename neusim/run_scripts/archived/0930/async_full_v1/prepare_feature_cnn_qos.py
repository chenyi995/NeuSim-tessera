"""Prepare sourced CNN/LLM inputs and execute matched-mapping array skew ablation.

All output is fresh. External Planaria/FissionSA/paper files are read-only.
Skew uses the existing final-drain-only control documented in
docs/tessera_partitioned.md: each SRAM-tile execution adds H-1 cycles, with
the full design's geometry, reuse, and double-buffer schedule held fixed.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
from neusim.npusim.backend.tessera_partitioned import counts

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parents[1]
OLD = ROOT/'results/tessera/20260929_joint_edp/sram_padding_revision/full_v1'
PLANARIA = PROJECT/'planaria.code'
G2 = PROJECT/'FissionSA/workloads/paper results/G2_planaria_suite'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()

def rows(path):
    with path.open(newline='') as f:
        yield from csv.DictReader(f)

def save_json(path, data):
    with path.open('x') as f:json.dump(data,f,indent=2)
    assert json.loads(path.read_text()) == data

def save_csv(path,data):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    assert list(rows(path)) == [{k:str(v) for k,v in r.items()} for r in data]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    manifest=json.loads((OLD/'manifest.json').read_text());assert manifest['status']=='PASS'
    base=Path(manifest['base']);high=max(manifest['bandwidth_bytes_per_second'])
    cfg=json.loads((OLD/'configs.json').read_text())[f'Tessera-8@{high}']
    totals={(r['workload_id'],r['architecture']):r for r in rows(OLD/'totals.csv') if int(r['bandwidth_bytes_per_second'])==high}
    names={r['workload_id']:r['workload'] for r in rows(PROJECT/'Tessera-HPCA2027-revision/fig/plotting/data/neusim/workload_sources.csv')}
    group={(r['dataset'],r['cohort']):r['workload_id'] for r in totals.values()}
    weight=defaultdict(dict)
    for r in rows(OLD/'invocation_weights.csv'):
        weight[r['kind'],r['operator_id']][group[r['dataset'],r['cohort']]]=int(r['repeat'])
    shapes={}
    inputs=base/'inputs_snapshot/inputs/workloads.csv'
    for r in rows(inputs):
        shapes['shape',r['shape_id']]=(1,*(int(r[k]) for k in ('M','N','K')))
    graph=base/'speed/trace_graph_operators.jsonl'
    with graph.open() as f:
        for line in f:
            r=json.loads(line);op=r['operator']
            if op['opcode']!='Einsum':continue
            aa,bb=([int(x['size']) for x in t['axes']] for t in op['input_tensors'])
            assert aa[-1]==bb[-2]
            b=math.prod(aa[:-2]);m,k=aa[-2:];n=bb[-1]
            assert b*m*n*k==op['stats']['flop_count']//2
            shapes['graph',str(r['operator_id'])]=(b,m,n,k)
    print('Loaded source shape definitions and invocation weights.',flush=True)
    matrix_weight=defaultdict(Counter)
    paired=[];acc=defaultdict(Counter);selected=0
    with (out/'saved_full_hbm4.csv').open('x',newline='') as f:
        writer=None
        for r in rows(OLD/'operator_costs.csv'):
            if r['architecture']!='Tessera-8' or int(r['bandwidth_bytes_per_second'])!=high:continue
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
            writer.writerow(r);selected+=1
            key=r['kind'],r['operator_id'];mapping=json.loads(r['mapping_json'])
            phases=[]
            if int(r['useful_macs']):
                if 'geometry' in mapping:
                    phases=[dict(dims=shapes[key],copies=1,geometry=mapping['geometry'],memory_tile=mapping['memory_tile'])]
                else:
                    phases=[dict(dims=(1,x['M'],x['N'],x['K']),copies=x['copies'],geometry=x['geometry'],memory_tile=x['memory_tile']) for x in mapping['phase_plans']]
            fs=ss=macs=0
            for phase in phases:
                b,m,n,k=phase['dims'];g=tuple(phase['geometry']);tile=tuple(phase['memory_tile']);copies=phase['copies']
                values=[]
                for variant in ('full','skew'):
                    detail=counts(b,m,n,k,2,2,2,g,variant,cfg['tessera_parameters']['grain'],cfg['sa_dim'],
                        cfg['vmem_size_MB']*1024**2,cfg['freq_GHz'],cfg['hbm_bw_GBps'],False,tile,'padded_tiles','padded_tiles')
                    values.append(detail['sa_cycles'])
                # Independent scalar identity for the existing final-drain control.
                tile_executions=b*math.ceil(m/tile[0])*math.ceil(n/tile[1])*math.ceil(k/tile[2])
                assert values[1]-values[0]==tile_executions*(g[-2]-1)
                fs+=copies*values[0];ss+=copies*values[1];macs+=copies*b*m*n*k
                for wid,repeat in weight[key].items():matrix_weight[tuple(phase['dims'])][wid]+=repeat*copies
            assert math.ceil(fs/cfg['freq_GHz'])==int(r['sa_ns']),(key,fs,r['sa_ns'])
            assert macs==int(r['useful_macs'])
            paired.append(dict(kind=key[0],operator_id=key[1],full_sa_cycles=fs,skew_sa_cycles=ss,useful_macs=macs))
            for wid,repeat in weight[key].items():
                acc[wid]['full_sa_cycles']+=repeat*fs;acc[wid]['skew_sa_cycles']+=repeat*ss;acc[wid]['useful_macs']+=repeat*macs
    assert selected==manifest['operators']
    save_csv(out/'skew_operator_cycles.csv',paired)
    feature=[]
    for wid,name in names.items():
        c=acc[wid];f=totals[wid,'Tessera-8'];s=totals[wid,'Tessera-8-independent_noskew']
        assert c['full_sa_cycles']==int(f['sa_ns'])*cfg['freq_GHz'] and c['useful_macs']==int(f['useful_macs'])
        feature.append(dict(workload_id=wid,workload=name,bandwidth_bytes_per_second=high,
            interconnect_full_energy_J=f['energy_J'],interconnect_disabled_energy_J=s['energy_J'],
            interconnect_energy_reduction_pct=100*(1-Decimal(f['energy_J'])/Decimal(s['energy_J'])),
            interconnect_edp_reduction_pct=100*(1-Decimal(f['edp_J_s'])/Decimal(s['edp_J_s'])),
            full_sa_cycles=c['full_sa_cycles'],skew_sa_cycles=c['skew_sa_cycles'],
            skew_array_latency_reduction_pct=100*(1-Decimal(c['full_sa_cycles'])/Decimal(c['skew_sa_cycles']))))
    save_csv(out/'feature_metrics.csv',feature)
    # Source CNN graph: preserve every defined compute/pooling layer. Depthwise
    # channels are distinct batched GEMMs; the old G2 flattened their work only.
    sys.path.insert(0,str(PLANARIA))
    spec=importlib.util.spec_from_file_location('planaria_benchmark_source',PLANARIA/'src/benchmarks/benchmarks.py')
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    from nn_dataflow import ConvLayer,FCLayer,PoolingLayer
    from nn_dataflow.Layer import DWConvLayer,LocalRegionLayer
    network_funcs={'resnet50':'get_resnet_50','googlenet':'get_googlenet','mobilenet_v1':'get_mobilenet_v1',
        'tiny_yolo':'get_yolo','yolov3':'get_yolo_v3','ssd_resnet34':'get_ssd_resnet_34',
        'ssd_mobilenet_v1':'get_ssd_mobilenet_v1','gnmt_encoder':'get_gnmt_encoder','gnmt_decoder':'get_gnmt_decoder'}
    layers=[];cnn_checks=[]
    source_dir=out/'sources';source_dir.mkdir()
    for stem,fn in network_funcs.items():
        src=G2/(stem+'.csv');shutil.copyfile(src,source_dir/src.name);assert sha(src)==sha(source_dir/src.name)
        golden=[]
        for line in src.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith(('#','Layer','layer')):
                golden.append(tuple(int(x.strip()) for x in line.split(',')))
        net=getattr(bench,fn)();matrix_index=0
        for layer_name,layer in net.layer_dict.items():
            if layer_name==net.INPUT_LAYER_KEY:continue
            row=dict(network=stem,workload_id='cnn_'+stem,layer=layer_name,index=len([x for x in layers if x['network']==stem]),source_type=type(layer).__name__)
            if isinstance(layer,ConvLayer):
                m,n,k=layer.hofm*layer.wofm,layer.nofm,layer.nifm*layer.sfil**2
                if isinstance(layer,DWConvLayer):
                    k=layer.sfil**2;b=n;real_n=1
                else:b=1;real_n=n
                assert golden[matrix_index][1:]==(m,n,k),(stem,layer_name,golden[matrix_index],(m,n,k))
                row.update(kind='matrix',B=b,M=m,N=real_n,K=k,useful_macs=b*m*real_n*k,input_bytes=2*b*m*k,output_bytes=2*b*m*real_n,vector_ops=0)
                matrix_weight[b,m,real_n,k]['cnn_'+stem]+=1
                matrix_index+=1
            elif isinstance(layer,LocalRegionLayer):
                # Planaria Layer.get_pool_ops defines one operation per window element.
                work=layer.get_pool_ops() if isinstance(layer,PoolingLayer) else layer.total_ops()
                row.update(kind='vector',B=0,M=0,N=0,K=0,useful_macs=0,input_bytes=2*layer.total_ifmap_size(),output_bytes=2*layer.total_ofmap_size(),vector_ops=work)
            else:raise ValueError(type(layer).__name__)
            layers.append(row)
        assert matrix_index==len(golden)
        cnn_checks.append(dict(network=stem,matrix_layers=matrix_index,all_source_layers=len([x for x in layers if x['network']==stem])))
    save_csv(out/'cnn_layers.csv',layers)
    save_json(out/'cnn_source_checks.json',cnn_checks)
    shape_rows=[];invocations=[]
    for index,(dims,ww) in enumerate(sorted(matrix_weight.items())):
        shape_rows.append(dict(shape_id=index,B=dims[0],M=dims[1],N=dims[2],K=dims[3]))
        invocations.extend(dict(workload_id=wid,shape_id=index,repeat=count) for wid,count in sorted(ww.items()))
    save_csv(out/'matrix_shapes.csv',shape_rows);save_csv(out/'matrix_invocations.csv',invocations)
    back=Counter();shape_by_id={str(r['shape_id']):r for r in shape_rows}
    for r in rows(out/'matrix_invocations.csv'):
        d=shape_by_id[r['shape_id']];back[r['workload_id']]+=int(r['repeat'])*math.prod(d[k] for k in ('B','M','N','K'))
    for wid in names:assert back[wid]==int(totals[wid,'Tessera-8']['useful_macs'])
    for stem in network_funcs:assert back['cnn_'+stem]==sum(x['useful_macs'] for x in layers if x['network']==stem)
    sources=[Path(__file__),OLD/'manifest.json',OLD/'totals.csv',OLD/'configs.json',OLD/'operator_costs.csv',OLD/'invocation_weights.csv',inputs,graph,
        PLANARIA/'src/benchmarks/benchmarks.py',PLANARIA/'nn_dataflow/Layer.py',PLANARIA/'scheduler/scheduler.py',PLANARIA/'scheduler/generator.py',PLANARIA/'scheduler/scheduler.ini',
        PROJECT/'FissionSA/multitenant/run_abc_qos.py',ROOT/'neusim/npusim/backend/tessera_partitioned.py',ROOT/'neusim/npusim/backend/tessera.py',
        *[G2/(stem+'.csv') for stem in network_funcs]]
    hashes={str(p):sha(p) for p in sources}
    for p in sources:
        expect=manifest['output_sha256'].get(p.name) if p.parent==OLD else manifest['source_sha256'].get(str(p))
        if expect:assert hashes[str(p)]==expect,str(p)
    for p in (OLD/'configs.json',OLD/'totals.csv',OLD/'manifest.json'):
        dst=source_dir/p.name;shutil.copyfile(p,dst);assert sha(dst)==sha(p)
    save_json(out/'verification.json',dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),
        skew_array_kernel_run=True,skew_scope='Fixed full mapping; final-drain-only H-1 per SRAM execution; bank release unchanged.',
        operators_checked=selected,workloads=len(names),unique_matrix_shapes=len(shape_rows),cnn_sources=cnn_checks,
        array_only_scope='All matrix work of each source workload; native fused QK/PV outer tiles fixed to the saved full HBM4 mapping; inner geometry/SRAM tiling can be reselected.',
        cnn_scope='Every compute/pooling layer explicitly present in Planaria definitions; serial im2col GEMMs, grouped depthwise, and native vector pooling. No invented unspecified operators.',
        source_sha256=hashes,artifact_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='verification.json'}))
    print(json.dumps(dict(status='PASS',operators=selected,unique_matrix_shapes=len(shape_rows),cnn_layers=len(layers)),indent=2),flush=True)

if __name__=='__main__':main()
