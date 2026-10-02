# Tessera extension and current experiment

This fork preserves NeuSim's default backend and adds a partition-aware array backend. The published snapshot is the per-workload comparison in [all_workloads.pdf](../artifacts/tessera-20261001/per_workload/figures/all_workloads.pdf). Its inputs, operator costs, arrivals, configurations, energy counters and plotting scripts are included in the [artifact](../artifacts/tessera-20261001/README.md). That snapshot predates the SRAM transfer-boundary correction described below. The corrected local rerun is tracked in [the transfer-tiling run](../results/tessera/20261001_ppa_transfer_tiles_v2/README.md), whose verification records determine completion status.

## Implementation

| Files | Responsibility |
|---|---|
| `neusim/npusim/backend/tessera_partitioned.py` | Finite SRAM, regional geometry, padding, partial sums, array timing and within-round packing |
| `tessera_baselines.py` | Legal baseline geometries and array/traffic counters |
| `tessera_joint.py` | Joint SRAM-tile and array mapping using native E2E EDP |
| `tessera_native_tail.py` | Native factor tiling, whole-array bulk work and fracturable residual packing |
| `power_model.py`, `ChipConfig.py` | Native component energy and explicit calibrated array coefficients |
| `neusim/run_scripts/replay_tessera_arrivals.py`, `tessera_request_dispatch.py` | Recorded arrivals, dependencies, physical regions and shared component service |
| `tools/tessera_artifact.py` | Self-contained input restoration, operator profiling and arrival replay |

Backend filenames without directories in the table are under `neusim/npusim/backend/`. Historical timing-only and custom energy adapters remain in source for provenance; the published configuration selects `tessera_partitioned`. Archived code is not imported or collected as tests.

## Energy units and scope

| Component | Coefficient | Definition |
|---|---:|---|
| WS array | 6.302 pJ/op | One multiply or add; MAC = two ops |
| Tessera array | 6.303 pJ/op | One multiply or add; MAC = two ops |
| Planaria array | 6.485 pJ/op | One multiply or add; MAC = two ops |
| SRAM read or write | 0.255247 pJ/bit | Native aggregate dynamic coefficient, before regulator |
| SRAM 16-bit transfer | 4.083958 pJ/access | Full-width linear byte accounting |
| SRAM 32-bit transfer | 8.167917 pJ/access | Full-width linear byte accounting |
| SRAM static | 24.21353615 W | Charged separately over elapsed time |

Compute uses the period-corrected non-SRAM Joules plateau as an activity-energy proxy, not an isolated dynamic FMA measurement. Native static and regulator terms remain. Independent WS arrays use the WS coefficient. SISA/SOSA/FlexSA use the Planaria proxy. SRAM has no single coefficient per arithmetic op: its traffic depends on reuse, tiling, padding and partial sums. The exploratory N28 SRAM datasheets discussed later have NOT been substituted into these runs.

The compute coefficient is the non-SRAM Joules plateau multiplied by the measured clock period and divided by arithmetic operations per cycle. The raw timelines and extracted source parameters are in `artifacts/tessera-20261001/calibration/`. Full precision remains in `configs/chips/tessera_joules_energy.json`; display rounding does not change the simulation.

SRAM dynamic energy is the native dynamic-power coefficient multiplied by transferred bytes divided by native reference bandwidth. The ideal per-PE SRAM service assumption changes service time, not this per-byte cost. Input fills, array operand reads, padded reads, partial writes and reduction reads/writes all contribute. HBM reloads follow finite-capacity tiling. HBM power remains NeuSim's transfer/controller/PHY model, not a calibrated HBM4 DRAM-core/refresh model. Static energy and regulator losses are separate native terms. The Joules array coefficient includes total non-SRAM plateau power, so retaining native static power makes this a rough activity proxy rather than a clean dynamic-only calibration.

## Mapping and E2E boundary

Normal tiling calls the native factor search with the revision's double-buffer and FP32 partial-output capacity. Connected designs execute complete weight blocks on the full array and pack residual work at the supported grain. Joint EDP tiling evaluates capacity-feasible divisor/dyadic SRAM tiles and legal array mappings against native component delay and total energy. This is a finite per-GEMM search, not a global scheduling optimum. Both policies retain native FlashAttention outer tiling.

The original adapter treated each SRAM transfer tile as an independent array invocation. Native selection could choose a short reduction tile without pricing the resulting repeated padding and startup, producing an artificial normal/EDP gap. New PPA configurations use `sram_tiling_model=transfer_only`: array cycles, physical padding and regional reductions follow the complete GEMM and its selected geometry; the memory tile controls capacity, HBM reloads and SRAM fill traffic. Both normal and EDP scoring use these same counters. A fixed WS geometry therefore has identical array work across transfer choices, while memory-bound E2E time and memory energy can change. Native bulk/tail execution fractures only genuine matrix tails.

This follows NeuSim's analytical separation of compute and overlapped memory service; it does not model DMA command latency or fine-grained operand readiness. An `array_tile` records the compute span and does not imply that the whole operand tensor resides in SRAM. FlashAttention outer tiles and dependency order remain intact. Replay consumes the same compute span, while finite memory reservations and shared HBM/vector service remain enforced. Saved configurations without this flag retain historical execution semantics for reproducing the published snapshot.

Tessera enables within-round packing and asynchronous admission of independent adjacent requests. Arrival idle, batch boundaries, autoregressive dependencies, pinned partials and shared HBM/vector service remain. Other arrays use serial request replay. Workloads without independent request metadata retain the complete sourced operator invocation sum. These are analytical E2E workload estimates, not RTL cycle traces or host scheduler measurements.

## Reproduction

The latest local [area-corrected plots](../results/tessera/20261001_ppa_ws_sram_area_v4/README.md)
reuse the transfer-tiling run's latency, energy and mappings. chenyi9 ruled that independent
WS arrays use the matching Planaria fission tier's SRAM area, with WS FMA and Logic area
scaled to the same PE count. The source is `Tessera-HPCA-2026/fig/plotting/make_area_figs.py`:
`A_FMA[0] + A_LOG[0]` supplies the WS array, and `A_SRAM` supplies the matching Planaria
tier. Both are multiplied by the total-PE ratio. This is an area estimate derived from
the submitted RTL figure, not a new synthesis measurement. Logic includes WS interconnect
and other non-FMA logic; no Planaria interconnect subtraction is inferred. Controllers
remain excluded as in the source figure. Area scaling does not change NeuSim SRAM energy
coefficients or capacity. Saved historical area metadata remains reproducible; the report
accepts an explicit `--area-configurations` override for this correction.

Install the package following the upstream README. From the repository root:

```bash
python tools/tessera_artifact.py verify
python tools/tessera_artifact.py restore --out results/tessera/restored
python tools/tessera_artifact.py profile --restored results/tessera/restored \
  --policy joint_edp --workers 16 --memory-gb 100 --out results/tessera/new-edp-costs
python tools/tessera_artifact.py replay --restored results/tessera/restored \
  --policy joint_edp --costs results/tessera/new-edp-costs --workers 16 --memory-gb 100 \
  --out results/tessera/new-edp-replay
```

Use `native_tail` for the other policy. Add `--transfer-only` to the `profile` command for the repaired accounting; both policies must use the same setting. Omit `--costs` to replay the bundled historical operator profiles. Profiling and replay require fresh output paths. `--limit` is an explicit incomplete smoke test, never a complete workload result. The resource arguments control workers and address-space limits; numerical libraries use one thread. Source-machine absolute paths in frozen manifests are provenance, while the portable adapter uses restored input paths and saved physical configurations.

Run active tests with `python -m pytest`; where pytest-cov is not installed, use `python -m pytest -o addopts=''`. Historical performance sweeps are not automatically launched by tests. Reference scheduler code and sourced CNN test inputs are included under `references/`; the latest PPA workflow does not need sibling repositories.
