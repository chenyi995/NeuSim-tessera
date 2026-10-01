"""Generate the review README and final cross-artifact audit from source CSVs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import csv,json,math,shutil
from pathlib import Path
from decimal import Decimal as D
from datetime import datetime
from zoneinfo import ZoneInfo
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,PROJECT,rows,sha,save_csv,save_json
from neusim.run_scripts.run_feature_qos_scheduler import make_info,C,UNITS
BASE=ROOT/'results/tessera/20260929_feature_cnn_qos'

def main():
 os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);out=BASE/'final_audit_v2';out.mkdir(exist_ok=False)
 audits={}
 for dirname in ('inputs_v1','verify_joint_v1','verify_extension_v2','verify_selection_v1','cnn_full_v1','granularity_full_v1','qos_profiles_full_v1','qos_native_grain_profiles_v1','qos_full_v1','review_cnn_v1','review_array_v1','review_qos_v3','granularity_explanation_v1'):
  p=BASE/dirname/'verification.json';v=json.loads(p.read_text());assert v['status']=='PASS',dirname
  audits[dirname]=sha(p)
  for name,h in v.get('artifact_sha256',v.get('artifacts',{})).items():assert sha(p.parent/name)==h,(dirname,name)
 paper=PROJECT/'Tessera-HPCA2027-revision';pub=json.loads((paper/'fig/plotting/data/neusim/feature_publication_verification.json').read_text());assert sha(paper/'main.pdf')==pub['pdf_sha256']
 # Validate every published new percentage against its input source.
 feature=list(rows(BASE/'inputs_v1/feature_metrics.csv'));published={r['workload_id']:r for r in rows(paper/'fig/plotting/data/neusim/ablation_contribution.csv')}
 for r in feature:
  for field in ('interconnect_energy_reduction_pct','skew_array_latency_reduction_pct'):assert published[r['workload_id']][field]==r[field]
 # Preserve external source repositories byte-for-byte for all referenced inputs.
 refs=json.loads((BASE/'inputs_v1/verification.json').read_text())['source_sha256'];external={k:v for k,v in refs.items() if '/planaria.code/' in k or '/FissionSA/' in k}
 for name,h in external.items():assert sha(Path(name))==h,name
 profiles=BASE/'qos_native_grain_profiles_v1';info=make_info(profiles);isolated=[];detail=[]
 layers=list(rows(BASE/'inputs_v1/cnn_layers.csv'));types={r['network']+'/'+r['layer']:r['source_type'] for r in layers}
 for arch,nets in info.items():
  for name,allp in nets.items():
   ll,suffix=allp[UNITS[arch]];dw=sum(n*t for lname,n,t,e in ll if types[lname]=='DWConvLayer')
   isolated.append(dict(architecture=arch,network=name,isolated_latency_ms=suffix[0]/1e6,depthwise_latency_ms=dw/1e6,depthwise_fraction=dw/suffix[0],qos_S_ms=C['QOS_BASE_MS'][name],qos_M_ms=C['QOS_BASE_MS'][name]*C['QOS_LEVELS']['M'],qos_H_ms=C['QOS_BASE_MS'][name]*C['QOS_LEVELS']['H']))
   detail.extend(dict(architecture=arch,network=name,layer=lname,source_type=types[lname],tiles=n,time_ns=n*t) for lname,n,t,e in ll)
 save_csv(out/'qos_isolated_networks.csv',isolated);save_csv(out/'qos_isolated_layers.csv',detail)
 # All unachievable B/C-M/H results have an isolated source-model lower bound
 # above the corresponding SLA; no invented QPS or speedup is written.
 qrows=list(rows(BASE/'review_qos_v3/qos_comparison.csv'))
 iso={(r['architecture'],r['network']):r for r in isolated}
 for r in qrows:
  for key,arch in [('planaria','Planaria-32'),('tessera','Tessera-8')]:
   if r[key+'_sla_attainable']=='False':
    assert any(iso[arch,n]['isolated_latency_ms']>iso[arch,n]['qos_'+r['qos']+'_ms'] for n in C['WORKLOADS'][r['workload']])
 # Readback exact arithmetic identity for the copied isolated profile table.
 for r in rows(out/'qos_isolated_networks.csv'):
  ss=math.fsum(float(v['time_ns']) for v in rows(out/'qos_isolated_layers.csv') if v['architecture']==r['architecture'] and v['network']==r['network'])
  assert math.isclose(ss/1e6,float(r['isolated_latency_ms']),rel_tol=2e-15)
 cnn=list(rows(BASE/'review_cnn_v1/cnn_totals.csv'));cs=list(rows(BASE/'review_cnn_v1/cnn_summary.csv'));arr=list(rows(BASE/'review_array_v1/granularity_summary.csv'));name_order=[r['workload_id'] for r in feature]
 text='''# NeuSim feature table and review-only Figs. 9, 10, 12

chenyi9 ruled: add interconnection total energy and fixed-map pure array skew
latency to the paper table; run new CNN E2E, array-only EDP granularity, and QoS
experiments for review. The new Fig. 9/10/12 results have not been installed in
the paper. The feature table has been installed in the verified paper PDF.
chenyi9 ruled: both QoS systems use Planaria's allocation algorithm, preserving
Tessera's 8x8 and Planaria's 32x32 allocation units at equal total PE count.

All reported simulation values are **modeled**, not hardware measurements.
Percentages, ratios, and geometric means below are **derived** from the linked
CSV outputs. Figures retain unfavourable results and search-bound labels.

## Common hardware and workload accounting

The run inherits `inputs_v1/sources/configs.json`: 128x128 total PEs, 1 GHz,
12 MiB finite SRAM, HBM4 2.8 TB/s, and native HBM latency/power, VU/reduction,
and fixed-voltage regulator accounting. The requested per-PE SRAM double-buffer
supply avoids an artificial SRAM bandwidth bottleneck. SRAM capacity, tile reuse,
HBM refetch, and transfer energy remain finite. Padded MACs and padded input
reads consume energy. Dynamic compute coefficients are 4.41 pJ/op for
WS/Tessera and 4.54 pJ/op for Planaria/SOSA/FlexSA/SISA; a MAC is two ops.
These coefficients are the user's prior ruling, with the existing paper values
and proxy scope retained in each saved configuration.

CNN definitions come directly from `planaria.code/src/benchmarks/benchmarks.py`
and `nn_dataflow/Layer.py`. `inputs_v1/cnn_source_checks.json` compares every
compute layer with the existing G2 CSVs. The source graphs define '''+str(sum(r['kind']=='matrix' for r in layers))+''' compute
layers plus '''+str(sum(r['kind']=='vector' for r in layers))+''' pooling/local layers. This run includes every defined layer,
including pooling, and represents depthwise channels as independent batched
GEMMs. NeuSim serializes those batch instances; it does not add a new cross-batch
channel-packing optimization. Standard convolutions use im2col GEMM work;
separate im2col materialization and operators absent from the supplied source
definitions are not invented. E2E here is serial latency of the supplied graph,
not a measured framework inference run. This source/model boundary matters for
MobileNet and for comparison with the previous flattened-depthwise experiment.

## Paper feature table

Source: [feature_metrics.csv](inputs_v1/feature_metrics.csv). Interconnection
energy reduction is `100*(1-E_full/E_disconnected)` and includes total chip and
HBM energy. Both mappings optimize E2E EDP. The skew comparison reruns native
array kernels with the full design's mapping fixed, restoring only the existing
control's final H-1 skew/de-skew drain per SRAM execution. It preserves bank
release timing; it is not a full traditional bank-release timing experiment.
`100*(1-T_full_array/T_skew_array)` is a pure array latency reduction.

| Workload | Interconnection energy reduction | Skew array latency reduction |
|---|---:|---:|
'''
 for r in feature:text+=f"| {r['workload']} | {D(r['interconnect_energy_reduction_pct']):.2f}% | {D(r['skew_array_latency_reduction_pct']):.2f}% |\n"
 text+='''
## Fig. 10: native E2E speed

[Plot](review_cnn_v1/fig10_cnn_e2e.pdf) · [PNG](review_cnn_v1/fig10_cnn_e2e.png) ·
[All architecture totals and utilization](review_cnn_v1/cnn_totals.csv) ·
[Every layer](review_cnn_v1/cnn_layer_costs.csv)

Each architecture selects its own legal geometry and capacity-feasible SRAM
tile by the common **E2E EDP** objective, consistent with the main evaluation.
The plotted metric is subsequently `WS E2E time / architecture E2E time`.
The existing suite contains seven CNNs plus GNMT encoder/decoder. Pooling is
included in E2E; layer energies and times are summed before forming ratios.

| Workload | Tessera / WS E2E speed | Tessera / Planaria E2E speed | Tessera array utilization |
|---|---:|---:|---:|
'''
 for r in cnn:
  if r['architecture']=='Tessera-8':text+=f"| {r['workload']} | {float(r['e2e_speedup_over_ws']):.3f}x | {float(r['speedup_over_planaria']):.3f}x | {100*float(r['array_utilization']):.2f}% |\n"
 text+='\n| Architecture | Nine-network geomean vs WS | Seven-CNN geomean vs WS |\n|---|---:|---:|\n'
 for r in cs:text+=f"| {r['architecture']} | {float(r['dnn_geomean_speedup_ws']):.3f}x | {float(r['cnn_geomean_speedup_ws']):.3f}x |\n"
 text+='''
Utilization is useful MACs divided by total PE count times summed array cycles.
It excludes vector/memory-only elapsed time. A speed below one is preserved.

## Fig. 9: array-only EDP selection

[CNN plot](review_array_v1/fig9_cnn_array_edp.pdf) ·
[All twelve LLM plots](review_array_v1/fig9_llm_array_edp.pdf) ·
[Full numeric results](review_array_v1/granularity_summary.csv) ·
[Best-grain shares](review_array_v1/best_grain_shares.csv)

Every one of '''+f"{json.loads((BASE/'inputs_v1/verification.json').read_text())['unique_matrix_shapes']:,}"+''' distinct batched matrix shapes is evaluated at minimum
grain 2, 4, 8, 16, 32, 64, and 128. All capacity-feasible native divisor SRAM
tiles and legal geometries are searched. The objective is **array energy over
array-active time times array latency**, including padded MAC energy, SA static
energy over that array window, and native regulator loss. SRAM, HBM, VU, and
chip-background energy/time are excluded from this requested array-only metric;
finite SRAM feasibility remains enforced. The exact evaluated configs are in
`review_array_v1/actual_array_configs.json` (the generic `configs.json` in the
raw granularity directory contains inherited E2E presets, not the array sweep
configs; the executed runner and actual config file define this experiment).

The denominator for each GEMM is its minimum array EDP over the seven grains.
Workload bars are invocation-weighted geometric means of those per-GEMM ratios,
matching the old Fig. 9 per-layer accounting. CNN layers have equal invocation
weight. The CSV separately includes total array energy, summed latency, their
product, and utilization; a geometric mean of per-GEMM ratios is not an E2E EDP
ratio. Fused attention QK/PV outer tiles are fixed to the saved HBM4 full-design
plans, so all grains evaluate identical sourced work; inner mappings reoptimize.

| Workload | Best aggregate grain | Grain 8 array EDP / per-GEMM optimum |
|---|---:|---:|
'''
 for wid in ['cnn_suite','dnn_suite',*name_order]:
  rr=[r for r in arr if r['workload_id']==wid];best=min(rr,key=lambda r:float(r['normalized_geomean_array_edp']));r=next(r for r in rr if r['grain']=='8')
  text+=f"| {r['workload']} | {best['grain']} | {float(r['normalized_geomean_array_edp']):.4f}x |\n"
 text+='''
The new result does not reproduce the previous near-flat grain-8 knee. Removing
memory energy/time from the objective exposes small-M array scheduling costs.
`granularity_explanation_v1/shape_categories.csv` and
`largest_weighted_effects.csv` give the sourced shapes, invocation weights, and
both mappings. For M=1 groups, the summed grain-8 array time is about four times
the grain-2 time in this model. The two-request-batch handoff uses `max(M,H)`, so
smaller H lowers the bank service interval on these short streams. The previous
main E2E result still includes memory and vector costs and uses different maps.
The coefficient across grains is held constant; this sweep does not model new
RTL area, clock, or wiring-energy costs for implementing grain 2.

## Fig. 12: same Planaria algorithm, native allocation grains

[Plot](review_qos_v3/fig12_planaria_policy_qos.pdf) ·
[Comparison](review_qos_v3/qos_comparison.csv) ·
[Raw pools](qos_full_v1/pool_results.csv) ·
[Exact scenarios and source hashes](qos_full_v1/scenario.json)

The allocation functions are imported directly from
`planaria.code/scheduler/scheduler.py`: possible core counts, fit checking,
priority/slack scoring, and fit/non-fit allocation. Both sides use these exact
functions. Tessera exposes 256 units of 8x8 PEs; Planaria exposes 16 units of
32x32 PEs. Each tenant receives proportional SRAM capacity and HBM bandwidth.
Every allocation enforces the total PE budget. NeuSim profiles select minimum
native E2E latency, then energy for ties, following the Planaria profiler's
cycle-first policy (`src/optimizer/optimizer.py:429-440`). This differs from the
EDP-selected maps used to report Fig. 10 speed. All '''+f"{json.loads((profiles/'verification.json').read_text())['records']:,}"+''' unique CNN operator /
architecture / allocation records are covered, including pooling.

The event driver fixes bookkeeping for idle queues and simultaneous arrivals /
completions, uses exact rational progress fractions, and checks completion of
each task exactly once. It preserves tile-boundary scheduling and the original
all-task barrier: an early-finishing tenant waits for the latest tile boundary
before reallocation. Native divisor SRAM tiles are equal-work scheduling units;
per-tile latency is the total modeled layer latency divided by tile count.
This is an analytical preemption model without detailed interconnect-placement,
VU-contention, or reconfiguration traces.

A/B/C, S/M/H, SLA thresholds, 50-task pool size, 12 pools per point, and nine
bisection rounds come from `FissionSA/multitenant/run_abc_qos.py`. Arrival generation
is the specified original Planaria generator: **integer-ms Poisson-distributed
intervals**, not exponential interarrival intervals. Identical offered load and
seed yield identical task types, priorities, and arrivals for both architectures.
At sub-ms mean intervals it often generates simultaneous bursts. These finite
pool results must not be interpreted as measured steady-state serving capacity.
The 50,000 QPS endpoint is checked explicitly and reported as a lower bound.
Two capped values do not establish equal throughput or an exact speedup.

| Scenario | Planaria QPS | Tessera QPS | Tessera / Planaria |
|---|---:|---:|---:|
'''
 for r in qrows:
  vals=[]
  for key in ('planaria','tessera'):
   vals.append('SLA unmet' if r[key+'_sla_attainable']=='False' else ('≥ ' if r[key+'_capped']=='True' else '')+f"{float(r[key+'_qps']):,.1f}")
  relation=r['ratio_relation'];ratio='Both capped' if relation=='both_capped' else ('Undefined' if relation=='unattainable' else ('≥ ' if relation=='lower_bound' else '≤ ' if relation=='upper_bound' else '')+f"{float(r['tessera_over_planaria']):.3f}x")
  text+=f"| {r['workload']}-{r['qos']} | {vals[0]} | {vals[1]} | {ratio} |\n"
 text+='''
B/C-M/H fail the SLA even at the low-load feasibility endpoint. In this native
source-graph model, SSD-MobileNet-v1 already exceeds the M-level deadline in
isolation, and both MobileNet variants exceed H. The cost includes explicitly
batched depthwise channels, which NeuSim executes serially. This is a concrete
simulator mapping limitation, not evidence that the hardware cannot satisfy
those deadlines. The isolated audit is:

| Network / architecture | Isolated latency (ms) | Depthwise share | M SLA (ms) | H SLA (ms) |
|---|---:|---:|---:|---:|
'''
 for r in isolated:
  if r['network'] in ('MobileNet-v1','SSD-MobileNet-v1'):text+=f"| {r['network']} / {r['architecture']} | {r['isolated_latency_ms']:.4f} | {100*r['depthwise_fraction']:.2f}% | {r['qos_M_ms']:.4f} | {r['qos_H_ms']:.4f} |\n"
 text+='''
A-M favours Planaria under the shared scheduler; that result is retained. The
current run establishes the comparison, but does not establish a unique causal
attribution between resource allocation, memory reuse, and tile-barrier timing.

## Validation and reproduction

All raw runs are append-only. The `final_audit_v2` manifest checks the current
paper PDF and every published new table value, referenced external source hashes,
output readback, and all result manifests. External Planaria/FissionSA sources
remain unmodified. Original useful-activity regression checks, padded scalar /
vector allocation checks, independent exhaustive array-EDP and latency winners,
and scheduler hand-computed cases all passed. The existing Tessera/native /
partitioned unit-test modules also passed. Run directories with no PASS audit
are intermediate failures and are not result sources.

Reproduction (from the NeuSim root, use fresh output names; run worker-heavy
commands sequentially to remain within the user's 16-CPU / 100-GB budget):

```bash
.venv/bin/python -m neusim.run_scripts.run_feature_cnn_qos granularity --workers 15 --out <fresh-array-run>
.venv/bin/python -m neusim.run_scripts.run_feature_cnn_qos cnn --workers 15 --out <fresh-cnn-run>
.venv/bin/python -m neusim.run_scripts.run_feature_cnn_qos qos_profiles --workers 15 --out <fresh-coarse-profiles>
.venv/bin/python -m neusim.run_scripts.run_feature_qos_fine_profiles --coarse <fresh-coarse-profiles> --workers 15 --out <fresh-native-grain-profiles>
.venv/bin/python -m neusim.run_scripts.run_feature_qos_scheduler --profiles <fresh-native-grain-profiles> --workers 15 --out <fresh-qos-run>
```

The full-grain profile run reuses verified coarse records at identical physical
resources and computes the remaining Tessera allocations. No simulation is
rerun merely to create the report. Figure scripts only consume audited CSVs.
'''
 path=BASE/'README.md'
 archive=out/'previous_README.md';shutil.copyfile(path,archive);assert sha(path)==sha(archive)
 path.write_text(text);assert path.read_text()==text
 save_json(out/'verification.json',dict(status='PASS',audits=audits,paper_pdf_sha256=pub['pdf_sha256'],new_paper_cells_checked=len(feature)*2,external_sources_unchanged=external,readme_sha256=sha(path),source_script_sha256=sha(Path(__file__)),artifact_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
 print('PASS final audit and generated README',flush=True)
if __name__=='__main__':main()
