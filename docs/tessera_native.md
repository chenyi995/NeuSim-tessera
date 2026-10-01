# Tessera SA replay with native NeuSim accounting

The partition-aware implementation is documented in
[tessera_partitioned.md](tessera_partitioned.md). The timing-only replay below
remains the controlled comparison used to diagnose the original discrepancy.

The `tessera_native_replay` backend changes only the SA time returned from the
native GEMM cost function. It enters before the original SA/VU selection rule.
Native SRAM tiling, HBM traffic and latency, component overlap, utilization,
power gating, energy, voltage-regulator losses and frequency policies run as
before. The default backend is unchanged. `test_tessera_native.py` checks exact
native statistics for identical SA timings and exercises fallback and input
validation.

This is a controlled timing-replay adapter, not a complete native Tessera
implementation. Native memory and VU accounting see the same 128x128 host across
the architectural candidates. They do not know the physical partitions, links,
or architecture-specific partial-sum traffic. Importing those costs from another
simulator would change the comparison's accounting boundary.

## Reference and execution

The reference is the read-only `FissionSA/Tessera-revision/final-three-ablations`
archive. `tools/reproduce.py --mode replay` re-evaluates the frozen SA candidate
costs with native Planaria traffic templates, reselects mappings and recomputes
the skew drain ablation. It checks its outputs against the archive. This does
not regenerate every original candidate simulation.

`replay_revision_archive.py` launches that tool with outputs under NeuSim.
`compare_tessera_native.py` then runs the same MNK values through NeuSim, without
expanding `repeat` into a batch dimension. `summarize_tessera_native.py` applies
each saved weight once, sums time and energy separately, and computes workload
EDP from those totals. The result covers the collected GEMMs, not normalization,
softmax, a full model graph, request queuing or application latency.

The source of PE count, frequency, SRAM and HBM parameters is the archive's
`EXPERIMENT_RATIONALE.md` and the snapshotted configuration and candidate files.
NeuSim's `vmem_size_MB=12` means 12 MiB, while Fission's compute candidates use
12,000,000 B. HBM's 2.8e12 bytes/s is converted to the binary GB/s convention
used by native NeuSim. All native power coefficients, 6 VUs, the 500 ns HBM
latency and `NoPG` remain unchanged. Those chip coefficients are not measured
Tessera silicon power. HBM energy in this native model covers controller/PHY
and data transfer; it is not a detailed external DRAM-array/refresh model.

All runs are append-only. The scripts record input hashes, source snapshots,
configuration and output hashes. CPU affinity is limited to the authorized
16 CPUs; each process receives a share of the 100 GB address-space budget.
Numerical libraries use one thread. The only cached native function in the
runner is the pure memory tile search, with unchanged arguments and return
values. Reference packages are imported with bytecode writes disabled.

## Controls and interpretation

| Output prefix | Meaning |
|---|---|
| `fission_` | Reproduced final revision with matched architecture-specific Planaria HBM templates |
| `native_frozen_` | Same revision-selected SA candidates, with native NeuSim surroundings; primary paired comparison |
| `native_` | EDP selection across the same candidate set using native NeuSim costs; a fallback-policy diagnostic |
| `native_forceSA_` | Native configuration `use_vu_for_small_matmul=False`, with native EDP selection |
| `native_1GHz_` | Same candidates as `native_`, component frequencies explicitly configured through the native API |
| `fission_common_` | All architectures use the Planaria full-array HBM template, isolating template dependence |
| `fission_serial_` | Reference transfer/compute serialization control |
| `native_local8_`, `native_local32_` | Previous local model supplies only SA cycles; all surrounding accounting is native |
| `legacy8_`, `legacy32_` | Previous custom model on the same cold isolated GEMMs; not native NeuSim accounting |

Two native behaviors matter for interpretation:

1. The native GEMM dispatcher selects SA when its estimated time is at most
   four times the VU estimate. Otherwise it selects VU. A slower SA candidate
   can therefore trigger faster VU execution, making EDP non-monotonic in SA
   cycles. The comparison retains this policy. Native remapping results must
   identify the selected engine; a VU result does not establish Tessera's array
   benefit. The fixed-mapping and forced-SA controls expose this effect.
2. The native `NONE` frequency policy hardcodes 1.7 GHz for all components,
   including HBM's frequency proxy. `analyze_operator_energy` scales component
   times even when `enable_dvfs=False`. Thus setting only `ChipConfig.freq_GHz=1`
   does not produce a final 1 GHz result. The primary run preserves this behavior;
   the explicit-frequency control uses the existing per-component API without
   changing native power functions.

The reference and legacy ablations also differ. Final revision uses traditional
independent WS32 and a fixed-mapping, final-drain-only skew delta. The previous
custom model uses skew-free independent arrays at the configured grain, and its
skew variant changes both bank release and drain. The previous full-graph results
used a different workload collection and grain. The retained same-input legacy
runs separate these effects from the native wrapper differences.

## Latest complete comparison

Results and generated tables are under
`results/tessera/archived/0929/20260929_native_compare/analysis_with_forceSA/`. The input
replay is `fission_replay/`; final raw native outputs and snapshots are in
`native_full_with_forceSA/`. Earlier pilots and diagnostic runs remain in place.

Example commands, each requiring a **new** output directory:

```bash
.venv/bin/python -B -m neusim.run_scripts.replay_revision_archive \
  --archive /path/to/final-three-ablations --out results/new-comparison --workers 16
.venv/bin/python -B -m neusim.run_scripts.compare_tessera_native \
  --archive /path/to/final-three-ablations \
  --replayed results/new-comparison/fission_replay \
  --out results/new-comparison/native --workers 16
.venv/bin/python -B -m neusim.run_scripts.summarize_tessera_native \
  --archive /path/to/final-three-ablations \
  --replayed results/new-comparison/fission_replay \
  --run results/new-comparison/native --out results/new-comparison/analysis
```
