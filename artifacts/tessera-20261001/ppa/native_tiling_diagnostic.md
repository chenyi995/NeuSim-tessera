# Native tiling diagnostic

This diagnostic reads the first shape in each completed profile. It does not rerun or change mappings. The two policies use identical energy coefficients. Native factor search minimizes HBM traffic; its compute estimate does not vary across tile candidates. Among otherwise tied candidates, the final sort key prefers the smaller SRAM footprint. Consequently it can choose a K tile smaller than the physical array grain. The revision accounting charges every resulting padded array invocation and its SRAM accesses. The joint EDP search instead scores those costs explicitly.

Source: `neusim/npusim/backend/npusim_lib.py`, `find_best_tile_shape_for_matmul`, `MXU_cycles = nc_compute_ns * freq_GHz` and the final candidate sort key; `tessera_partitioned.memory_tile` supplies the revision double-buffer and partial-sum capacity.

| Policy | Architecture | SRAM tile M,N,K | Charged/useful MACs | Operator time (ms) | Energy (J) |
|---|---|---|---:|---:|---:|
| native_tail | WS | [1, 32000, 1] | 128.00 | 262.68 | 31.23 |
| native_tail | Planaria-8 | [1, 32000, 1] | 8.00 | 1.11 | 0.16 |
| native_tail | Tessera-8 | [1, 32000, 1] | 8.00 | 1.14 | 0.16 |
| joint_edp | WS | [1, 256, 128] | 1.00 | 1.54 | 0.19 |
| joint_edp | Planaria-8 | [1, 8000, 256] | 1.00 | 0.09 | 0.02 |
| joint_edp | Tessera-8 | [1, 16000, 128] | 1.00 | 0.09 | 0.02 |

These are operator-level examples. Complete E2E and all workload comparisons remain in the two policy directories.
