"""Source-backed exploratory attention workloads; unchanged current NeuSim PPA.

Run from NeuSim: .venv/bin/python results/tessera/20261001_llm_candidate_ppa_v1/experiment.py
All four cohorts are fixed before profiling. All cases within each are retained.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import ast
import csv
import hashlib
import json
import math
import resource
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FISSION = ROOT.parents[1] / 'FissionSA'
sys.path.insert(0, str(ROOT))
from neusim.run_scripts.run_tessera_ppa import configurations
from neusim.run_scripts.run_tessera_partitioned import native_record, ENERGIES
from neusim.npusim.frontend.llm_ops_lib import create_multi_head_flash_attention_op

MODES = ('native_tail', 'joint_edp')
# Source: current PPA endpoint and user's equal-PE/runtimes rulings.
BW = 2_800_000_000_000
WORKERS = 24  # Within chenyi9's 64 CPU / 200 GB authorization.
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path):
    SOURCES[str(path)] = sha(path)
    return path


def read_csv(path):
    with source(path).open() as f:
        return list(csv.DictReader(f))


def save_json(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2)
    assert json.loads(path.read_text()) == obj


def save_csv(path, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with path.open() as f:
        assert list(csv.DictReader(f)) == [{k: str(v) for k,v in r.items()} for r in rows]


def prepare():
    cases = []
    def add(wid, label, q, kv, hq, hkv, d, repeat, path, line, request, evidence):
        assert 0 < q <= kv and hq % hkv == 0 and repeat > 0
        cases.append(dict(workload_id=wid, workload=label, q=q, kv=kv, hq=hq, hkv=hkv, d=d,
                          M=q*(hq//hkv), repeat=repeat, source=str(path), source_line=line,
                          request_id=str(request), evidence=evidence))

    # Falcon architecture is from its pinned official config. KV trajectory is
    # transplanted from the complete recorded Dolly-long decode, not advertised
    # as a Falcon GPU trace. There is no request batching or head duplication.
    cfgpath = FISSION/'Tessera-revision/sources/falcon/config.json'
    cfg = json.loads(source(cfgpath).read_text())
    assert cfg['multi_query'] and not cfg['new_decoder_architecture']
    hq = cfg['num_attention_heads']; d = cfg['hidden_size']//hq
    assert d*hq == cfg['hidden_size']
    path = FISSION/'workloads/workloads-Qwen-models/Qwen2.5-7B-Instruct_bs1_seq268_long/full_trace.csv'
    qwen = read_csv(path); steps = defaultdict(list)
    for i,r in enumerate(qwen,2):
        if r['phase']=='decode' and r['torch_op']=='aten::bmm' and r['op_type']=='qk_scores':
            shape = ast.literal_eval(r['input_shapes'])
            assert tuple(shape[0][-2:]) == (int(r['M']),int(r['K']))
            assert tuple(shape[1][-2:]) == (int(r['K']),int(r['N']))
            steps[int(r['step'])].append((i,r))
    lengths = []
    for step, entries in sorted(steps.items()):
        assert len({int(r['N']) for _,r in entries}) == 1
        i,r = entries[0]; kv=int(r['N']); lengths.append(kv)
        add('falcon_mqa_dolly_decode','Falcon-7B MQA decode',int(r['M']),kv,hq,1,d,
            cfg['num_hidden_layers'],path,i,step,'derived: Falcon config + complete recorded Dolly-long KV trajectory')
    assert len(lengths)==len(steps) and all(b==a+1 for a,b in zip(lengths,lengths[1:]))

    # All source Phi-2 prefill invocations, preserving source chunking and KV.
    metapath=ROOT/'artifacts/tessera-20261001/inputs/snapshot/workload_metadata.json'
    model=json.loads(source(metapath).read_text())['models']['phi2_conv300']
    path=FISSION/'out/tes-old/continuous_batching/traces/phi2_conv300/batch_composition.csv'
    for i,r in enumerate(read_csv(path),2):
        if r['phase'] != 'prefill': continue
        q=int(r['tokens_this_iter']); kv=q+int(r['kv_cache_before'])
        add('phi2_chat_prefill','Phi-2 chat prefill',q,kv,model['num_q_heads'],model['num_kv_heads'],
            model['embedding_dim']//model['num_q_heads'],model['num_layers'],path,i,r['request_id'],
            'source trace: all prefill requests; attention operator boundary')

    path=FISSION/'out/tessera-revision/planaria_workloads/upstream_reuse_800/workloads.csv'
    reuse=read_csv(path)
    pv={(r['cohort'],r['request_id'],r['layerclass']):r for r in reuse if r['op']=='PV'}
    targets={'CacheBlend_wikimqa':('cacheblend_wikimqa_attention','CacheBlend WikiMQA'),
             'EPIC_hotpotqa':('epic_hotpotqa_attention','EPIC HotpotQA')}
    for i,r in enumerate(reuse,2):
        if r['op']!='QK' or r['cohort'] not in targets: continue
        hq,hkv,d=(int(r[k]) for k in ('Hq','Hkv','head_dim'))
        q,kv=int(r['query_tokens']),int(r['context_tokens'])
        counterpart=pv[r['cohort'],r['request_id'],r['layerclass']]
        assert (int(r['M']),int(r['N']),int(r['K']))==(q*(hq//hkv),kv,d)
        assert (int(counterpart['M']),int(counterpart['N']),int(counterpart['K']))==(q*(hq//hkv),d,kv)
        assert r['repeat']==counterpart['repeat'] and int(r['repeat'])%hkv==0
        assert sha(Path(r['model_config']))==r['model_config_sha256']
        source(Path(r['model_config'])); source(Path(r['source']))
        add(*targets[r['cohort']],q,kv,hq,hkv,d,int(r['repeat'])//hkv,path,i,r['request_id'],
            'upstream source-derived reuse plan: all requests and layer classes; attention operator boundary')

    # Independent direct readback of the original fields copied into every case.
    bypath=defaultdict(list)
    for c in cases: bypath[c['source']].append(c)
    for path,cc in bypath.items():
        original=read_csv(Path(path))
        for c in cc:
            r=original[c['source_line']-2]
            if c['workload_id'].startswith('falcon'):
                assert c['kv']==int(r['N']) and c['q']==int(r['M'])
            elif c['workload_id'].startswith('phi'):
                assert c['q']==int(r['tokens_this_iter']) and c['kv']==c['q']+int(r['kv_cache_before'])
            else:
                assert c['q']==int(r['query_tokens']) and c['kv']==int(r['context_tokens'])
                assert c['repeat']*c['hkv']==int(r['repeat'])
    shapes=sorted({tuple(c[k] for k in ('q','kv','hq','hkv','d')) for c in cases})
    ids={s:i for i,s in enumerate(shapes)}
    for c in cases: c['operator_id']=ids[tuple(c[k] for k in ('q','kv','hq','hkv','d'))]
    save_csv(HERE/'cases.csv',cases)
    save_json(HERE/'shapes.json',[list(s) for s in shapes])
    summary=[]
    for wid in dict.fromkeys(c['workload_id'] for c in cases):
        cc=[c for c in cases if c['workload_id']==wid]
        summary.append(dict(workload_id=wid,workload=cc[0]['workload'],source_cases=len(cc),
            requests=len({c['request_id'] for c in cc}),M_min=min(c['M'] for c in cc),M_max=max(c['M'] for c in cc),
            KV_min=min(c['kv'] for c in cc),KV_max=max(c['kv'] for c in cc),
            head_dimensions=sorted({c['d'] for c in cc}),evidence=cc[0]['evidence']))

    # MoE question: verify which dimension varies in the actual routed source.
    moepath=FISSION/'Tessera-revision/traces_checked/olmoe_decode/mnk_trace.csv'
    moe=read_csv(moepath); moe_summary={}
    for op in sorted({r['op'] for r in moe}):
        rr=[r for r in moe if r['op']==op]
        moe_summary[op]=dict(M=sorted({int(r['M']) for r in rr}),KN=sorted({(int(r['K']),int(r['N'])) for r in rr}))
    # Normalize tuples before round-trip validation.
    moe_summary=json.loads(json.dumps(moe_summary))
    save_json(HERE/'source_manifest.json',dict(status='PASS',workloads=summary,moe=moe_summary,
        source_sha256=SOURCES,selection='Four source/model cohorts chosen before PPA profiling; no individual cases filtered by results.',
        accounting='Complete attention-operator E2E invocation sum, including QK/softmax/PV, finite SRAM and HBM; not whole-model latency. All source layer/head multiplicities retained. No causal triangle skipping. No invented arrivals or cross-step overlap.',
        packing='Current native within-round and tail packing retained. Single-operator invocations have no independent request-arrival contract for asynchronous overlap. Dependent decode steps remain serial.'))
    return cases,shapes


CONFIGS={}
def init_worker():
    # 24 workers * 5 GB plus a bounded parent stays below the authorized 200 GB.
    resource.setrlimit(resource.RLIMIT_AS,(5_000_000_000,5_000_000_000))
    for mode in MODES:
        CONFIGS[mode]=configurations(BW,mode)[0]


def job(payload):
    oid,shape=payload
    q,kv,hq,hkv,d=shape
    rows=[]
    for mode in MODES:
        for (arch,bw),chip in CONFIGS[mode].items():
            op=create_multi_head_flash_attention_op([1,q,hq,d],[1,kv,hkv,d],[1,kv,hkv,d])
            r=native_record(op,chip)
            assert r['useful_macs']==2*hq*q*kv*d
            assert r['charged_sa_macs']>=r['useful_macs']
            rows.append(dict(operator_id=oid,tiling_policy=mode,architecture=arch,**r))
    return rows


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:WORKERS])
    begun=time.monotonic(); cases,shapes=prepare()
    configs={}; area={}
    for mode in MODES:
        chips,meta=configurations(BW,mode)
        configs[mode]={name:chip.model_dump() for (name,_),chip in chips.items()}
        for r in meta: area[r['architecture']]=r
    save_json(HERE/'configs.json',configs); save_json(HERE/'area.json',area)
    engine_sources=[ROOT/'neusim/npusim/backend'/name for name in
        ('tessera_partitioned.py','tessera_joint.py','tessera_baselines.py','tessera_native_tail.py')]
    engine_sources += [Path(__file__),ROOT/'neusim/run_scripts/run_tessera_ppa.py',ROOT/'configs/chips/tessera_joules_energy.json']
    engine_hashes={str(p):sha(p) for p in engine_sources}
    rawdir=HERE/'raw'; rawdir.mkdir(exist_ok=False)
    print('START',len(cases),'cases',len(shapes),'unique operators',len(shapes)*len(area)*len(MODES),'costs',flush=True)
    costs={}
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=init_worker) as pool:
        futures=[pool.submit(job,p) for p in enumerate(shapes)]
        for n,f in enumerate(as_completed(futures),1):
            rr=f.result(); oid=rr[0]['operator_id']
            save_json(rawdir/f'{oid:04d}.json',rr)
            for r in rr: costs[r['tiling_policy'],r['architecture'],oid]=r
            if n%10==0 or n==len(shapes): print('PASS',n,'/',len(shapes),round(time.monotonic()-begun,1),'s',flush=True)
    save_csv(HERE/'operator_costs.csv',list(costs.values()))
    rows=[]
    fields=[k for k,v in next(iter(costs.values())).items() if isinstance(v,(int,float)) and k!='operator_id']
    for wid in dict.fromkeys(c['workload_id'] for c in cases):
        cc=[c for c in cases if c['workload_id']==wid]
        for mode in MODES:
            for arch,a in area.items():
                # Peak footprint is not additive; all other numeric native fields are counts/times/energies.
                acc={key:math.fsum(c['repeat']*costs[mode,arch,c['operator_id']][key] for c in cc)
                     for key in fields if key!='peak_live_bytes'}
                acc['peak_live_bytes']=max(costs[mode,arch,c['operator_id']]['peak_live_bytes'] for c in cc)
                useful=sum(c['repeat']*2*c['hq']*c['q']*c['kv']*c['d'] for c in cc)
                assert acc['useful_macs']==useful
                assert math.isclose(acc['energy_J'],sum(acc[k] for k in ENERGIES),rel_tol=2e-12)
                seconds=acc['time_ns']*1e-9; energy=acc['energy_J']; ops=useful*2
                rows.append(dict(workload_id=wid,workload=cc[0]['workload'],tiling_policy=mode,architecture=arch,
                    grain=a['grain'],area_mm2=a['area_mm2'],seconds=seconds,energy_J=energy,
                    performance_density_gops_per_mm2=ops/seconds/a['area_mm2']/1e9,
                    energy_efficiency_gops_per_w=ops/energy/1e9,
                    array_utilization=useful/(128**2*acc['sa_ns']),
                    average_power_W=energy/seconds,
                    **{k:v for k,v in acc.items() if k!='energy_J'}))
    save_csv(HERE/'all_points.csv',rows)
    # Read saved raw costs again to independently reproduce every aggregate's numerator/time/energy.
    raw={}
    for p in rawdir.glob('*.json'):
        for r in json.loads(p.read_text()): raw[r['tiling_policy'],r['architecture'],r['operator_id']]=r
    for r in rows:
        cc=[c for c in cases if c['workload_id']==r['workload_id']]
        rr=[(c['repeat'],raw[r['tiling_policy'],r['architecture'],c['operator_id']]) for c in cc]
        assert math.isclose(r['seconds'],sum(n*x['time_ns'] for n,x in rr)*1e-9,rel_tol=2e-12)
        assert math.isclose(r['energy_J'],sum(n*x['energy_J'] for n,x in rr),rel_tol=2e-12)
    assert all(sha(Path(p))==h for p,h in {**SOURCES,**engine_hashes}.items())
    save_json(HERE/'verification.json',dict(status='PASS',cases=len(cases),unique_operators=len(shapes),points=len(rows),
        workers=WORKERS,worker_memory_cap_bytes=5_000_000_000,elapsed_s=time.monotonic()-begun,
        engine_sha256=engine_hashes,output_sha256={n:sha(HERE/n) for n in ('cases.csv','shapes.json','source_manifest.json','configs.json','area.json','operator_costs.csv','all_points.csv')}))
    print('COMPLETE PASS',len(rows),'points',flush=True)


if __name__=='__main__': main()
