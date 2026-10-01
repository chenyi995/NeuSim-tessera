# Partition-aware Tessera in native NeuSim

`array_backend="tessera_partitioned"` extends the native GEMM and fused-attention
cost functions. Operators continue through `fill_operators_execution_info`,
native HBM latency/bandwidth, VU issue throughput, component overlap, power
gating, component power and regulator efficiency. The default backend and
`tessera_native_replay` timing-only control retain their previous behavior.

chenyi9 ruled (2026-09-29): provision each PE with a 16-bit input, a 16-bit
weight input and a 32-bit partial-sum output per cycle, with double buffering;
SRAM bandwidth must not constrain this rough evaluation. New partitioned runs
record `sram_bandwidth_model="per_pe_double_buffer"`. This assumption supersedes
the fixed native SRAM bandwidth used by `20260929_partitioned_full`.

## Accounting and references

The earlier workload/configuration source is the read-only
`FissionSA/Tessera-revision/final-three-ablations` archive, specifically
`EXPERIMENT_RATIONALE.md` and its snapshotted `inputs/workloads.csv`. The current
paper suite instead copies `FissionSA/out/tessera-revision/paper_revision/inputs/`
and the current paper's workload group list, with all original repeat weights.
`prepare_tessera_paper.py` copies source files through Python, verifies their
SHA256 digests, and checks workload MAC counts and full-trace coverage. The runner
records hashes of the input, chip configuration, model metadata, full traces,
source files, paper design section and RTL timing audit. All reported results
are analytical simulation estimates; a passing run is not measured silicon.

The physical partition rules and shared memory are from
`Tessera-HPCA2027-revision/sec/04_design.tex`, Strip memory and Mapping.
Operand ping-pong and FP32 partials use the widths specified there. Array timing
is checked against `gemmini/partition/TIMING_AUDIT.md`, including initial
transpose fill, registered seams and the short-M initiation interval.
`tessera.py::bank_cycles` implements the shared weight-bank recurrence;
`test_tessera.py` checks that recurrence against an independent event simulation.
This is an array-boundary calibration, not a full Gemmini command-controller
timing model.

## GEMM schedule and traffic

Each candidate has logical regional dimensions `(kt, nt)`, physical dimensions
`(H, W)`, K parallelism `kp`, and N parallelism `np`. Their product is the number
of physical regions. The selected region remains fixed at edge tiles. Actual
operands and arithmetic exclude padding, while physical launch/drain time
includes the padded region.

The existing native factor-based SRAM tile search receives effective operand
sizes for two buffers and output space for a running accumulator plus `kp`
producer planes. For a memory tile `(Mt, Nt, Kt)`, live shared memory is

```
2 * bytes(A) * Mt * Kt + 2 * bytes(B) * Kt * Nt
  + sizeof(FP32) * (kp + 1) * Mt * Nt.
```

This conservatively reserves complete producer planes. The search function,
factor enumeration, HBM objective and tie-breaking are native NeuSim code.
Only the capacity demand passed to that search changes. The output-stationary
loop accumulates each output tile across K on chip before writing final C.
Consequently HBM partial spills are zero under this schedule; capacity pressure
shrinks tiles and can cause A/B reloads. This does not claim that other loop
orders or private-SRAM organizations have the same traffic.

For each memory tile with actual dimensions `(m, n, k)`:

```
regional K groups = ceil(k / kt)
regional N groups = ceil(n / nt)
rounds = ceil(K groups / kp) * ceil(N groups / np)
array A reads = m * k * N groups
array B reads = k * n
partial writes = m * n * K groups
```

Across all K tiles, `m * n * (total K groups - 1)` scalar additions combine
partials. Each addition reads two FP32 inputs and writes one FP32 result.
The final C accumulator is read for conversion/output. SRAM traffic also
includes HBM input fills. HBM traffic follows the native output-stationary
equation: A reloads per N tile, B reloads per M tile, and C writes once. Partial
read/write counts are converted to VU issue work and SRAM time, not assigned
an external energy-per-access coefficient.

## Per-PE SRAM supply

The default partitioned SRAM model implements chenyi9's ruling above:

```
provisioned_bytes_per_cycle = num_sa * sa_dim**2 * (16 + 16 + 32) / 8
provisioned_bytes_per_ns = provisioned_bytes_per_cycle * freq_GHz
service_ns = ceil(sram_bytes / provisioned_bytes_per_ns)
other_ns = max(SA_ns, VU_ns, HBM_ns, ICI_ns)
visible_SRAM_ns = min(service_ns, other_ns)
```

The widths come from the user's stated assumption, not a characterized SRAM
macro. Ping-pong allows overlap; it does not double the useful port width or
the number of accesses. The visible-time bound explicitly enforces ideal
double buffering, rather than proving that a finite physical buffer can hide
every transfer. Uncapped service demand and the hidden excess are retained as
`sram_service_time_ns` and `sram_overlap_hidden_ns` in operator details and raw
mapping records. The verified configuration uses NoPG and equal component
frequencies. All partitioned variants, including WS and independent arrays,
receive the same bandwidth at equal PE count, regardless of logical cuts.
Vector operators and VU fallback also use this supply, retaining their native
traffic estimate. SRAM capacity, working-set reservations and HBM costs still
apply.

Codex decided: preserve the native dynamic energy cost per transferred byte,
so overlapping more traffic does not erase its energy. Before native regulator
losses, `E_SRAM = sram_bytes * P_native / BW_native`, with bandwidth in bytes per
second and the native power/bandwidth evaluated at the SRAM component frequency.
This is the native power coefficient expressed per byte, not a new fitted
coefficient. Static energy remains native power integrated over execution time.
Setting `sram_bandwidth_model="native"` retains the earlier timing/energy
accounting for controls; the stock NeuSim backend is unchanged.

Native factor search can select a very short K tile to maximize M/N reuse,
especially when M has few factors. Native unpartitioned timing did not model
the resulting array restarts. The added schedule now exposes those restarts
and reductions. This can produce very large penalties, particularly for the
monolithic WS control. The analysis flags short K tiles and provides the exact
mapping. Such results depend on the retained native tiler and do not establish
an equally large hardware advantage under a compute-aware tiler.

## Variants and engine policy

| Variant | Meaning |
|---|---|
| `full` | Dyadic rectangles and square fallback; native EDP selects the mapping |
| `square` | The same logical candidates padded to physical squares |
| `independent_noskew` | Fixed minimum-grain blocks with the same skew-free timing; isolates interconnection/merging |
| `independent` | Fixed minimum-grain traditional WS blocks; bridge to the revision's independent WS baseline |
| `skew` | Full's exact mapping and memory tiles, with traditional final-drain delay only |
| `ws` | One monolithic traditional WS array |

The primary `sa` policy uses the existing `use_vu_for_small_matmul=False` option.
The `native_auto` diagnostic retains NeuSim's existing four-times SA/VU
comparison. Its candidates are also selected by native EDP. A deliberately
slow SA candidate can cause fallback and lower EDP; the original policy is not
changed. Reports include the executed SA/VU MAC fractions. Rejected SA partials
and traffic are not charged to VU execution.

The skew control changes final drain only, following the final revision
ablation. It does not alter bank release. Memory, useful arithmetic and mapping
are identical to full. Native static energy still grows if the longer drain
extends execution. No unmeasured skew-register dynamic power is inserted.

## Fused attention and complete graphs

Fused attention starts from the native FlashAttention tile suggestion and native
HBM traffic equation. The tile is reduced if necessary to reserve ping-pong
Q/K/V buffers, score buffers, FP32 output and one regional-output wave. All
variants share this conservative workspace rule. QK and PV use the partitioned
GEMM scheduler with resident inputs/outputs; scores do not round-trip through
HBM. Grouped query heads share K/V through the M dimension. Softmax uses the
native scalar-operation count, and output combinations between KV tiles have
explicit VU and SRAM costs.

`native_llm_graph` lowers the saved inference graph to native Einsum,
FlashAttention, collective and vector operators. It includes every transformer
layer, norms, rotary embeddings, KV append, residuals, activation, biases where
specified, final norm, LM head and argmax. Layer multiplicity and every request's
KV length are retained. The test compares compact repeated layers against
expanded graphs, and every replayed batch checks useful MACs against the saved
trace plus its LM head.

Native per-operator tensor traffic is retained, including writes and reads at
operator boundaries; there is no custom persistent tensor LRU. The reported
E2E time is the sum of complete model-step service times. It excludes arrival
queueing and CPU/tokenizer scheduling. The operators are analytical costs, not
numerical network execution. Dense attention work follows the saved trace;
no additional causal-triangle skipping is assumed. TP system energy sums rank
energy, while synchronous latency is per rank. Operator-only revision cohorts
and full model traces are separate populations and are never averaged together.
For tensor parallelism, the saved graph's output boundary is rank-local
vocabulary logits and local argmax. A final cross-rank token-selection/gather
is not represented in this graph. Thus the trace totals describe the retained
model-compute boundary, rather than complete request-to-token application time.

## Native energy and limitations

The run uses the existing native component power coefficients, VU issue
throughput, HBM latency, `NoPG`, utilization formulas and regulator losses.
SRAM supply follows the explicit per-PE assumption above; native SRAM bandwidth
is retained as the reference for its energy-per-byte coefficient.
The extension supplies useful SA work and actual VU reduction work to the
existing utilization calculation. Native HBM energy models controller/PHY and
transfer activity; this is a rough memory-energy term, not a DRAM command,
refresh or external cell-array model. SRAM static power is the configured
native whole-component value, not a newly characterized SRAM macro.
ICI denotes inter-chip communication. The interconnection ablation concerns
array partitions; its circuit power is not the native ICI component. NoPG
retains configured ICI static power even when no communication occurs.

The explicit `frequency_policy="chip"` sets each native component frequency
through the existing per-component API. This avoids native `NONE` policy's
hardcoded frequency overriding the requested chip setting. Power coefficients
are still inherited native coefficients, not a frequency/technology calibrated
Tessera implementation. Architectural variants have equal PEs, shared capacity
and component powers; dedicated interconnect/transpose/register area and power
are not separately characterized.

Native execution remains the maximum of SA, VU, SRAM, HBM and communication
times per operator, with visible SRAM time bounded by the other components
under the requested ideal buffering assumption. A full dependency/event
simulation could expose serialization beyond this assumption. The model's reuse counts
are explicit, but the overlap assumption remains NeuSim's analytical roofline.
Absolute joules and large tiler-dependent speedups should be interpreted within
these limits.

## Reproduction and verification

Run from the NeuSim root with a new output directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -B -m neusim.run_scripts.run_tessera_partitioned \
  --archive /data2/chenyi9/fracturableT3/FissionSA/Tessera-revision/final-three-ablations \
  --out results/tessera/NEW_RUN --workers 16
```

`--suite operators` selects archived GEMMs; `--suite traces` selects complete
model traces; the default executes both. `--limit` is a pilot limit and must be
zero for full evaluation. CPU affinity, per-process address space and numerical
library threads enforce the authorized budget. Outputs and source snapshots are
append-only under NeuSim.

Run backend regression after changes to scheduling, memory or energy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python -B -m unittest discover -s neusim/npusim/backend/tests -v
```

The native operator/statistics and graph integration also use:

```bash
.venv/bin/python -B -m unittest \
  neusim.npusim.frontend.tests.test_operator \
  neusim.npusim.frontend.tests.test_frontend_util -v
```

`test_tessera_partitioned.py` compares traffic to independently written loops,
reads timing anchors from the RTL audit, checks capacity, useful MACs, native
energy conservation, fixed-mapping skew, fallback accounting, fused attention
and complete graphs. The result-analysis script independently rebuilds weighted
summaries from raw shape/batch rows and verifies source/output hashes. No
external traffic or energy template is imported into the partitioned model.

The completed evaluation under the earlier fixed native SRAM bandwidth is retained in
[`20260929_partitioned_full`](../results/tessera/archived/0929/20260929_partitioned_full/).
Its [report](../results/tessera/archived/0929/20260929_partitioned_full/analysis/REPORT.md)
contains the E2E, energy, phase, matched-workload and cross-simulator tables.
The [read-back audit](../results/tessera/archived/0929/20260929_partitioned_full/analysis/verification.json)
records the checks; the analysis directory contains the retained aggregation
script, full-precision CSVs and standalone figures. Raw rows and source
snapshots remain at the run root. The earlier pilot and operator precheck are
retained separately from the complete results.
These saved results have not been recomputed under the per-PE SRAM assumption.

## Earlier paper suite with a speed main figure

chenyi9 ruled: use model E2E for saved complete-model traces and single-operator
E2E for cohorts with only operator inputs. The latter includes the operator's
input reads, computation and output writes, with each original invocation
weight applied once. It does not imply a reconstructed application graph.
The primary figures retain both boundaries explicitly and do not combine them
into a single whole-model average.

The speed comparison uses `run_tessera_speed.py`. WS, Planaria-32, SOSA, FlexSA
and SISA array cycles/useful on-chip words are ported from
`FissionSA/fissionsa/modes/combo{0,1,4,10}.py`. Their executable array kernels,
not their standalone DRAM timing or energy systems, define the baseline
schedules. Planaria's legal compositions come from
`Tessera-revision/experiments/planaria_native/run_native_workloads.py::COMPOSITIONS`.
`verify_tessera_baselines.py` freezes these references and compares the ported
cycles and words against the original code over edge and randomized shapes.
The baseline dimensions and equal total PE count match the paper instances.

Every baseline uses the same native SRAM tile search, FP32 partial reductions,
HBM latency/bandwidth, ideal per-PE SRAM supply, power coefficients, operator
graph and SA engine policy as Tessera. Planaria selects its composition using
native NeuSim EDP; it does not import the external Planaria energy selector.
The WS baseline here is the paper's executable array kernel; the earlier
`ws` ablation uses the common weight-bank recurrence and remains a separate
diagnostic. These analytical ports do not claim cycle-accurate full-system
implementations of the baseline chips.

Codex decided: select every design's mapping at HBM4, then freeze mapping and
traffic across the existing paper's bandwidth grid. This isolates bandwidth
sensitivity without conflating it with remapping. The grid and frequency are
read by AST from
`FissionSA/out/tessera-hpca/Batch6-BW-Sweep/scripts/run_batch6.py::BW_GRID,FREQ`;
16-bit words per cycle are converted to bytes per second. The sweep reconstructs
the native maximum of SA, VU, ICI and HBM time from saved component-cost
histograms. It is checked against direct native execution at the reference
point and independently re-aggregated at every bandwidth. It is a pure time
comparison, even though the fixed mapping was selected by native EDP.

The speed runner globally reuses native per-operator costs using the existing
`native_record` cache identity, while retaining every invocation count from
every source batch. `trace_graph_operators.jsonl`, `trace_graph_batches.jsonl`
and `trace_graph_costs.csv` expose the identities, weights and computed costs.
`verify_tessera_graph_reuse.py` compares direct and reused real-model execution,
including process-pool serialization and CSV readback. The final analyzer
independently reweights this inventory and checks each batch, the component
histograms and all bandwidth totals. Reuse changes computation effort only;
it does not introduce cross-operator SRAM residency or change E2E accounting.

The report defines the figures as follows:

* Speed: `T_Planaria32 / T_design`, with both Tessera grains and all baselines.
* Non-square ablation: `EDP_square / EDP_full`, at HBM4, both grains.
* Interconnection ablation: `EDP_independent_noskew / EDP_full`, at HBM4,
  both grains. The disconnected control preserves skew-free timing.

For each boundary, energy and time are summed over the original invocation
weights before computing their product. CSVs also retain the separate time,
energy, SRAM and HBM ratios. Paired restored-skew timing and useful SA
utilization are speed diagnostics. The report checks that Tessera's HBM4
times/MAC counts agree between the independent speed and EDP runs.

`run_all.py` and `run_speed_and_report.py` in
`results/tessera/archived/0929/20260929_paper_sram_per_pe/` retain the sequential commands.
Each stage writes to a fresh directory; status JSON and command logs record
actual completion. The report is usable only after all corresponding
`verification.json` files show PASS. The earlier fixed-SRAM results remain
historical controls.

The server restart interrupted the grain-8 trace stage. `run_recovery.py` in
the same results directory continues that evaluation in `grain8_recovered/`
and records progress in `recovery_status.json`; the earlier status files and
partial results remain unchanged. `resume_tessera_partitioned.py` copies and
hash-checks the completed operator outputs, reuses their exact native cache
identities, and reconstructs the trace graph with all invocation weights.
It compares reconstructed costs with every complete saved batch and preserves
the saved batch values. Integer reductions are exact with a checked overflow
bound; floating-point energy comparisons use the existing analyzer tolerance.
Direct first-batch execution for each model checks the recovery path before
the remaining batches run. The usual independent analyzer still gates use of
the recovered results. The report's `--grain8-run` option selects this run.

The retained `speed_reuse_sources.json` explicitly lists audited EDP runs whose
Tessera costs may seed the speed sweep. The sweep accepts this file through
`--reuse-spec`, or reads it beside the copied inputs. It checks the run and
analysis status, raw-output hashes, metadata, chip configuration and executed
engine code before reusing a cost. A direct native model-batch canary compares
the reused costs before the scan. Baseline costs are evaluated normally.
An engine-code mismatch excludes that run from caching and is recorded in the
speed manifest; its costs are computed directly with the current engine.
Recovered graph costs that did not save mapping geometry retain JSON `null`
in that diagnostic field; their timing and traffic are checked exactly.

## Current paper suite: joint EDP selection

chenyi9 ruled: replace the general speed figure with EDP, jointly select SRAM
tiles and legal array configurations for every baseline, and re-select at each
bandwidth. The non-square and interconnection ablations use HBM4 and the same
joint mapper. The earlier fixed-mapping speed results above remain historical.

`tessera_parameters.mapping_policy = "joint_edp"` enables this policy.
`tessera_joint.py` enumerates every capacity-feasible divisor tile from native
`util.get_factors`, together with each architecture's existing legal geometry.
The search retains the common output-stationary SRAM loop order. SRAM demand
includes ping-pong operand buffers, producer partial planes and a running FP32
output plane. Operand re-fetches caused by M/N tiling are charged to HBM; partial
sums remain resident. This is finite-capacity reuse, with ideal SRAM bandwidth
as requested, and no cross-operator weight cache.

The objective is native total operator energy times native operator time.
It includes static and dynamic energy, HBM activity, SRAM traffic, and the
native regulator-efficiency table. A mapping that reads slightly more HBM can
win when it avoids repeated array startup or regional reduction. Minimizing
HBM traffic alone is therefore not the selection rule. The selected candidates
are executed through the native frontend and power pipeline. Vectorized search
algebra is verified against the scalar counters and this native pipeline.

Planaria searches its legal compositions and SRAM tiles. WS, SOSA, FlexSA and
SISA keep their published architecture schedules but also optimize SRAM tiles.
Tessera searches its grain-specific partitions and SRAM tiles; square-only
and disconnected controls restrict its legal geometries. The skew diagnostic
fixes the full design's selected geometry and SRAM tile. Native FlashAttention
retains its outer Br/Bc selection, with resident QK/PV phases using joint
selection. This is per-GEMM optimization within the stated candidate space,
not global graph EDP optimization or an exhaustive search over loop orders.

The original Planaria optimizer is a different policy. Its
`src/optimizer/optimizer.py::get_stats_fast` already promotes accesses within
finite local SRAM and includes DRAM energy. Its inner tile/order search first
minimizes cycles, breaking ties by energy; `Simulator.get_conv_cycles` then
minimizes EDP over compositions. It does not fetch every SRAM access from DRAM.
The new NeuSim comparison applies the same shared-memory accounting and joint
objective to Planaria and Tessera instead of importing that original selector.

Run `neusim.run_scripts.verify_tessera_joint` before
`neusim.run_scripts.run_tessera_joint`. The runner consumes the checked copied
workloads and native graph inventory from the earlier paper suite, preserves
all original invocation weights, and writes native energy components plus
mapping details for every point. `verify_tessera_joint_integration` checks
real-workload replay and compares joint GEMM EDP against the earlier selector.
`report_tessera_joint` independently reconstructs weights and totals, checks
every full-model batch, and produces:

* `general_edp_bandwidth`: Planaria EDP / architecture EDP.
* `edp_nonsquare`: square-only EDP / full Tessera EDP.
* `edp_interconnect`: disconnected skew-free EDP / full Tessera EDP.

All ratios are derived after accumulating energy and time separately. Model
energy includes the tensor-parallel chip count. The report retains raw time,
energy, average power and component energy tables alongside EDP. Output
directories are append-only; source hashes and independent verification gate
publication. Common native power coefficients provide rough modeling rather
than separate layout-calibrated power estimates for each architecture.
