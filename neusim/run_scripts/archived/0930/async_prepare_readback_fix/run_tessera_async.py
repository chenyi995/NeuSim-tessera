"""Arrival-preserving replay of sourced independent request chains.

Native NeuSim determines each operator's mapping, traffic, and component costs.
Replay preserves the source batches, autoregressive dependencies, and external
request arrivals. It does not turn a histogram of GEMMs into independent jobs.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,csv,hashlib,json,math,resource,time
from collections import Counter,defaultdict
from pathlib import Path
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,rows,sha,save_json
from neusim.run_scripts.run_tessera import trace_batches,requests_from_batch
from neusim.run_scripts.run_tessera_speed import operator_identity
from neusim.run_scripts.run_tessera_partitioned import config
from neusim.npusim.frontend.tessera_workloads import native_llm_graph,llm_graph

def raw_identity(r):
    return (r['opcode'],r['config_str'],r['input_tensor_shape_str'],r['output_tensor_shape_str'],
      r['stats']['flop_count'],r['stats']['ici_time_ns'],r['stats']['ici_traffic_inbound_bytes'],
      r['stats']['ici_traffic_outbound_bytes'],json.dumps(r['tessera_spec'],sort_keys=True))

def prepare(costs,out):
    manifest=json.loads((costs/'manifest.json').read_text());assert manifest['status']=='PASS'
    base=Path(manifest['base']);inputs=base/'inputs_snapshot'
    meta=json.loads((inputs/'workload_metadata.json').read_text())
    identities={};operator_meta={};source=[costs/'manifest.json',inputs/'workload_metadata.json']
    op_path=base/'speed/trace_graph_operators.jsonl';source.append(op_path)
    for line in op_path.open():
        item=json.loads(line);r=item['operator'];oid=item['operator_id']
        identities[raw_identity(r)]=oid
        operator_meta[oid]=dict(flops=r['stats']['flop_count'],mxu=r['op_type']=='MXU',opcode=r['opcode'],
            inputs=[[a['size'] for a in t['axes']] for t in r['input_tensors']])
    save_json(out/'operator_metadata.json',operator_meta)
    histogram={}
    path=base/'speed/trace_graph_batches.jsonl';source.append(path)
    for line in path.open():
        r=json.loads(line);histogram[r['model'],r['batch_id']]=dict(r['operators'])
    cfg=config('full',grain=8);summary=[];started=time.monotonic()
    for tag,model in meta['models'].items():
        comp=next((inputs/'trace_sources').rglob(tag+'/batch_composition.csv'))
        mnk=next((inputs/'trace_sources').rglob(tag+'/mnk_trace.csv'));source.extend([comp,mnk])
        composition=defaultdict(dict);arrival={}
        for r in rows(comp):
            bid=int(r['batch_id']);rid=int(r['request_id'])
            assert rid not in composition[bid]
            composition[bid][rid]=r
            t=float(r['arrived_at'])*1e9
            assert rid not in arrival or arrival[rid]==t
            arrival[rid]=t
        last={};count=0
        with (out/(tag+'.jsonl')).open('x') as f:
            for bid,rr in trace_batches(mnk):
                reqs=requests_from_batch(rr)
                ids=[int(r['request_id']) for r in rr if r['op']=='attn_qk']
                assert set(ids)==set(composition[bid]) and len(ids)==len(set(ids))
                for rid,(q,kv) in zip(ids,reqs):
                    r=composition[bid][rid]
                    assert q==int(r['tokens_this_iter'])
                    # KV length is taken from the actual GEMM trace. Source
                    # composition's decode cache bookkeeping can be offset.
                    assert kv>=q
                native=native_llm_graph(model,reqs,cfg,compact_layers=True)
                semantic=llm_graph(model,reqs,2,compact_layers=True)
                assert len(native)==len(semantic)
                refs=[identities[operator_identity(op)] for op in native]
                observed=Counter()
                for oid,op in zip(refs,native):observed[oid]+=op.stats.count
                assert dict(observed)==histogram[tag,bid],(tag,bid)
                # chenyi9: decision start — only independent adjacent requests
                # may overlap; shared projections and successive layers retain
                # their source order. Each request's RoPE/KV/attention is a chain.
                pre=[];body_before=[];chains=[];body_after=[];post=[]
                for oid,sem in zip(refs,semantic):
                    name=sem.name
                    if not name.startswith('layer'):
                        (pre if name=='embedding' else post).append(oid)
                    elif '/rope' in name:
                        chains.append([oid])
                    elif '/kv_append' in name or ('/attention' in name and name.rsplit('attention',1)[-1].isdigit()):
                        chains[-1].append(oid)
                    elif not chains:body_before.append(oid)
                    else:body_after.append(oid)
                assert len(chains)==len(ids) and all(len(c)==3 for c in chains)
                record=dict(batch_id=bid,request_ids=ids,arrival_ns=[arrival[x] for x in ids],
                    release_ns=max(arrival[x] for x in ids),dependencies=sorted({last[x] for x in ids if x in last}),
                    prefix=pre,before=body_before,chains=chains,after=body_after,suffix=post,layers=model['num_layers'])
                # chenyi9: decision end
                f.write(json.dumps(record,separators=(',',':'))+'\n')
                for rid in ids:last[rid]=bid
                count+=1
                if count%1000==0:print('prepare',tag,count,round(time.monotonic()-started,1),flush=True)
        # Read back every batch and reconstruct its independent source histogram.
        check_count=0
        for line in (out/(tag+'.jsonl')).open():
            r=json.loads(line);cc=Counter(r['prefix']+r['suffix'])
            body=r['before']+sum(r['chains'],[])+r['after']
            for oid in body:cc[oid]+=r['layers']
            assert dict(cc)==histogram[tag,r['batch_id']]
            check_count+=1
        assert count==check_count==len(composition)
        summary.append(dict(model=tag,batches=count,requests=len(arrival),source_composition=str(comp),source_mnk=str(mnk)))
    save_json(out/'verification.json',dict(status='PASS',models=summary,
      source_sha256={str(p):sha(p) for p in source},artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},
      accounting='Original request arrivals and idle gaps retained; original batches and request dependencies retained; only adjacent independent request chains may overlap.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare'])
    p.add_argument('--costs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    resource.setrlimit(resource.RLIMIT_AS,(3_000_000_000,3_000_000_000))
    prepare(a.costs.resolve(),a.out.resolve())
if __name__=='__main__':main()
