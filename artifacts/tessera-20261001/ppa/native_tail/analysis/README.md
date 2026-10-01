# Equal-PE E2E performance density and energy efficiency

Status: PASS. Results are analytical NeuSim simulations. All configurations contain 128×128 PEs.

## Method

Each marker is the equal-weight geometric mean of the same twelve complete sourced workloads. Performance is useful matrix arithmetic operations divided by E2E elapsed time; one MAC is two operations. Energy efficiency is useful operations per joule. Tensor-parallel chips contribute both energy and area. The denominator is the scaled RTL array-plus-SRAM area, excluding controllers as in the source figure; it is not a measured complete accelerator die area.

All designs use HBM4 (2.8 TB/s), 12 MiB finite live SRAM, 1 GHz, and the same tiling policy for this figure. HBM rereads and padded compute/SRAM accesses are charged. Recorded arrival idle and dependencies are retained. Tessera uses corrected within-round packing and asynchronous reuse of free physical regions for independent adjacent requests. Planaria retains its legal long compositions and conventional schedule. The independent WS banks use all their arrays in parallel.

Area is derived from the matching 32×32 RTL tier multiplied by sixteen: RTL d=2,4,8,16 maps to d=8,16,32,64. The five WS configurations use the monolithic WS area scaled to the same PE count, as requested. The WS curve connects 1×128², 4×64², 16×32², 64×16² and 256×8² independent arrays. Tessera and Planaria each sweep physical fission grains 64, 32, 16 and 8.

Compute uses the period-corrected Joules non-SRAM array proxy per executed arithmetic operation. This proxy includes active array logic and leakage and is not an isolated FMA measurement. The WS coefficient also applies to both independent WS configurations. All energy coefficients exactly match the corrected revision NeuSim evaluation. SRAM and HBM energy follow actual native traffic; NoPG static energy, VU/ICI service and regulator losses are retained. No RTL power ratio multiplies compute, SRAM, HBM, or static energy. RTL is used only for the requested area estimate. Mapping policy: Native NeuSim SRAM factor tiling with revision double-buffer/FP32-partial capacity accounting. Connected arrays execute full 128x128 weight blocks first; residual K/N strips and corners fracture to the physical grain and pack into available regions. Independent WS retains its fixed subarray size. No energy/EDP candidate ranking. FlashAttention retains native Br/Bc. E2E latency and energy are subsequently measured by native arrival/dependency replay.

## Geometric means (derived; monolithic WS = 1)

| Architecture | Area (mm²/chip) | E2E speed | Performance/area | Energy efficiency |
|---|---:|---:|---:|---:|
| WS | 54.40 | 1.00 | 1.00 | 1.00 |
| WS-independent-64 | 54.40 | 2.93 | 2.93 | 3.02 |
| WS-independent-32 | 54.40 | 4.53 | 4.53 | 4.51 |
| WS-independent-16 | 54.40 | 6.70 | 6.70 | 7.85 |
| WS-independent-8 | 54.40 | 4.74 | 4.74 | 5.54 |
| Planaria-64 | 61.73 | 3.17 | 2.80 | 2.52 |
| Tessera-64 | 58.29 | 3.03 | 2.83 | 2.46 |
| Planaria-32 | 71.04 | 5.94 | 4.55 | 4.43 |
| Tessera-32 | 65.38 | 5.37 | 4.47 | 4.17 |
| Planaria-16 | 86.46 | 9.15 | 5.76 | 6.71 |
| Tessera-16 | 79.54 | 7.89 | 5.40 | 6.18 |
| Planaria-8 | 119.98 | 12.68 | 5.75 | 10.00 |
| Tessera-8 | 107.78 | 10.38 | 5.24 | 8.92 |

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
