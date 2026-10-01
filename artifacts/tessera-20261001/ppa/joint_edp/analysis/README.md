# Equal-PE E2E performance density and energy efficiency

Status: PASS. Results are analytical NeuSim simulations. All configurations contain 128×128 PEs.

## Method

Each marker is the equal-weight geometric mean of the same twelve complete sourced workloads. Performance is useful matrix arithmetic operations divided by E2E elapsed time; one MAC is two operations. Energy efficiency is useful operations per joule. Tensor-parallel chips contribute both energy and area. The denominator is the scaled RTL array-plus-SRAM area, excluding controllers as in the source figure; it is not a measured complete accelerator die area.

All designs use HBM4 (2.8 TB/s), 12 MiB finite live SRAM, 1 GHz, and the same tiling policy for this figure. HBM rereads and padded compute/SRAM accesses are charged. Recorded arrival idle and dependencies are retained. Tessera uses corrected within-round packing and asynchronous reuse of free physical regions for independent adjacent requests. Planaria retains its legal long compositions and conventional schedule. The independent WS banks use all their arrays in parallel.

Area is derived from the matching 32×32 RTL tier multiplied by sixteen: RTL d=2,4,8,16 maps to d=8,16,32,64. The five WS configurations use the monolithic WS area scaled to the same PE count, as requested. The WS curve connects 1×128², 4×64², 16×32², 64×16² and 256×8² independent arrays. Tessera and Planaria each sweep physical fission grains 64, 32, 16 and 8.

Compute uses the period-corrected Joules non-SRAM array proxy per executed arithmetic operation. This proxy includes active array logic and leakage and is not an isolated FMA measurement. The WS coefficient also applies to both independent WS configurations. All energy coefficients exactly match the corrected revision NeuSim evaluation. SRAM and HBM energy follow actual native traffic; NoPG static energy, VU/ICI service and regulator losses are retained. No RTL power ratio multiplies compute, SRAM, HBM, or static energy. RTL is used only for the requested area estimate. Mapping policy: Per-GEMM minimum native E2E EDP over legal array geometries and capacity-feasible divisor/dyadic SRAM tiles. Time breaks EDP ties; FlashAttention retains native Br/Bc and selects resident phases. E2E latency and energy are subsequently measured by native arrival/dependency replay.

## Geometric means (derived; monolithic WS = 1)

| Architecture | Area (mm²/chip) | E2E speed | Performance/area | Energy efficiency |
|---|---:|---:|---:|---:|
| WS | 54.40 | 1.00 | 1.00 | 1.00 |
| WS-independent-64 | 54.40 | 1.35 | 1.35 | 1.21 |
| WS-independent-32 | 54.40 | 1.65 | 1.65 | 1.33 |
| WS-independent-16 | 54.40 | 1.88 | 1.88 | 1.35 |
| WS-independent-8 | 54.40 | 2.01 | 2.01 | 1.26 |
| Planaria-64 | 61.73 | 1.52 | 1.34 | 1.30 |
| Tessera-64 | 58.29 | 1.55 | 1.45 | 1.33 |
| Planaria-32 | 71.04 | 1.80 | 1.38 | 1.43 |
| Tessera-32 | 65.38 | 1.84 | 1.53 | 1.46 |
| Planaria-16 | 86.46 | 1.99 | 1.25 | 1.49 |
| Tessera-16 | 79.54 | 2.02 | 1.38 | 1.52 |
| Planaria-8 | 119.98 | 2.05 | 0.93 | 1.51 |
| Tessera-8 | 107.78 | 2.07 | 1.04 | 1.53 |

## Inputs and validation

- Area only: `/data2/chenyi9/fracturableT3/Tessera-HPCA-2026/fig/plotting/make_area_figs.py`; synthesis provenance: `/data2/chenyi9/fracturableT3/SA-rtl/docs/SYN22_ARRAY_CHIP_DATA.md`.
- Compute calibration: `/data2/chenyi9/fracturableT3/simulators/NeuSim/configs/chips/tessera_joules_energy.json`.
- Source workload names and citations: `/data2/chenyi9/fracturableT3/simulators/NeuSim/results/tessera/20260930_energy_corrected_v1/paper_data_2dp/workload_sources.csv`.
- `../verification/verification.json`: original FissionSA kernels, scalar/vector counters, native energy, and full-grid utilization checks.
- `../main_costs/`: fresh mappings, invocation weights, per-operator records, and configurations.
- `../arrival_replay/`: arrival/dependency-preserving native spatial replay.
- `workload_metrics.csv`: complete per-workload metrics, power components, padding and utilization.
- `geomean.csv`: full-precision plotted coordinates and normalized comparisons.
- Plot coordinates and all geometric means were read back and independently recomputed.
