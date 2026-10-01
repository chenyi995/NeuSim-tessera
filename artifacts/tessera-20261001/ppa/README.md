# Native bulk/tail versus joint EDP tiling

Status: PASS. Both policies completed all thirteen equal-PE configurations and twelve sourced workloads. Results are analytical NeuSim simulations. All numerical summaries below are derived from the retained CSVs.

- [Native tiling + tail packing](native_tail/analysis/energy_efficiency_vs_performance_area.pdf)
- [Joint EDP tiling](joint_edp/analysis/energy_efficiency_vs_performance_area.pdf)
- [Native per-workload plots and comparisons](native_tail/workloads/README.md)
- [EDP per-workload plots and comparisons](joint_edp/workloads/README.md)

Both figures plot energy efficiency against E2E performance per area. Each point is the equal-weight geometric mean of the twelve workloads. The independent WS line connects 1×128², 4×64², 16×32², 64×16², and 256×8² arrays; Tessera and Planaria each vary the fission grain over 64, 32, 16, and 8. Every configuration has the same total PE count.

chenyi9 ruled: energy uses the corrected revision NeuSim accounting, without RTL power multipliers. The existing calibrated compute coefficients and native SRAM/HBM/static/VU/ICI/regulator terms are retained. Actual padded compute and SRAM reads are charged; HBM rereads follow finite SRAM reuse. RTL is used only for the requested area estimate: the matching 32-scale design is scaled to the equal-PE fabric. Independent WS area uses the equal-PE monolithic WS proxy.

The native policy calls NeuSim find_best_tile_shape_for_matmul through the revision memory_tile wrapper. That wrapper reserves double-buffered operands and FP32 partial sums. Its producer-plane reservation is raised when the selected tile requires concurrent residual reductions, and the same native formula is called again. Within each SRAM tile, complete 128×128 K/N weight blocks execute first on the full connected array. Residual K/N strips and corners fracture to the configured grain. Mixed residual groups fill free physical regions and reuse retired regions. Each occupied grain cell remains reserved until its last planned local use; this is a conservative reservation for admission of another request, not an assumption of zero-cost instantaneous remapping. Independent WS retains its fixed subarray size.

The EDP policy searches the common capacity-feasible divisor/dyadic SRAM tiles and each architecture’s legal geometries for minimum native E2E EDP per GEMM. It preserves the existing finite search rather than claiming a global whole-workload optimum. Both policies preserve native FlashAttention outer Br/Bc tiling and apply their array policy to resident QK/PV phases. Tessera enables within-round packing and asynchronous admission of independent adjacent requests; recorded arrivals, pure idle, dependencies and pinned live SRAM are retained.

## Geometric means, normalized to monolithic WS within each policy

| Policy | Architecture | E2E speed | Energy efficiency | Performance/area |
|---|---|---:|---:|---:|
| native_tail | WS | 1.00× | 1.00× | 1.00× |
| native_tail | WS-independent-64 | 2.93× | 3.02× | 2.93× |
| native_tail | WS-independent-32 | 4.53× | 4.51× | 4.53× |
| native_tail | WS-independent-16 | 6.70× | 7.85× | 6.70× |
| native_tail | WS-independent-8 | 4.74× | 5.54× | 4.74× |
| native_tail | Planaria-64 | 3.17× | 2.52× | 2.80× |
| native_tail | Tessera-64 | 3.03× | 2.46× | 2.83× |
| native_tail | Planaria-32 | 5.94× | 4.43× | 4.55× |
| native_tail | Tessera-32 | 5.37× | 4.17× | 4.47× |
| native_tail | Planaria-16 | 9.15× | 6.71× | 5.76× |
| native_tail | Tessera-16 | 7.89× | 6.18× | 5.40× |
| native_tail | Planaria-8 | 12.68× | 10.00× | 5.75× |
| native_tail | Tessera-8 | 10.38× | 8.92× | 5.24× |
| joint_edp | WS | 1.00× | 1.00× | 1.00× |
| joint_edp | WS-independent-64 | 1.35× | 1.21× | 1.35× |
| joint_edp | WS-independent-32 | 1.65× | 1.33× | 1.65× |
| joint_edp | WS-independent-16 | 1.88× | 1.35× | 1.88× |
| joint_edp | WS-independent-8 | 2.01× | 1.26× | 2.01× |
| joint_edp | Planaria-64 | 1.52× | 1.30× | 1.34× |
| joint_edp | Tessera-64 | 1.55× | 1.33× | 1.45× |
| joint_edp | Planaria-32 | 1.80× | 1.43× | 1.38× |
| joint_edp | Tessera-32 | 1.84× | 1.46× | 1.53× |
| joint_edp | Planaria-16 | 1.99× | 1.49× | 1.25× |
| joint_edp | Tessera-16 | 2.02× | 1.52× | 1.38× |
| joint_edp | Planaria-8 | 2.05× | 1.51× | 0.93× |
| joint_edp | Tessera-8 | 2.07× | 1.53× | 1.04× |

## Validation

- `common_verification/`: original baseline kernels, scalar/vector counters, native energy and revision power coefficients.
- `native_verification_v2/`: literal weight-block counters, native tiler equality, physical non-overlap, bank-event timing, FlashAttention and asynchronous replay.
- Both policy directories retain fresh operator profiles, complete replay checkpoints, configuration files and source hashes.
- This script independently recomputes geometric means and checks identical power, capacity, bandwidth and area inputs between the policies.
- The initial native-plan check is retained as FAIL in `native_verification/`; the missing engine metadata was corrected before either full run.
- The earlier RTL-scaled and energy-only attempts remain separate; neither contributes points to these two figures.
