# Joint EDP tiling: separate workloads

All metrics are derived from completed analytical NeuSim simulations. The same 128×128 total PEs, finite SRAM, HBM4, revision energy coefficients, padded work, arrivals and dependencies apply to every design. The WS line connects 1×128², 4×64², 16×32², 64×16² and 256×8² independent arrays. WS area is the equal-PE monolithic proxy requested by chenyi9; RTL affects area only. Tessera enables both within-round packing and asynchronous use of free physical regions.

## Tessera grain 32 to grain 8

| Workload | E2E speedup | Energy reduction | Performance/area gain |
|---|---:|---:|---:|
| Llama-2-7B chat | 1.32× | 10.21% | 0.80× |
| Llama-3-8B chat | 1.22× | 7.04% | 0.74× |
| Llama-2-70B chat | 1.04× | 1.04% | 0.63× |
| Phi-2 chat | 1.45× | 15.61% | 0.88× |
| Llama-2-7B code | 1.02× | 0.81% | 0.62× |
| Llama-2-7B arXiv | 1.48× | 14.87% | 0.90× |
| FlashInfer GQA | 1.04× | 3.15% | 0.63× |
| FlashInfer MQA | 1.04× | 2.89% | 0.63× |
| CacheBlend SAMSum | 1.00× | -0.01% | 0.61× |
| CacheBlend WikiMQA | 1.00× | -0.01% | 0.61× |
| EPIC HotpotQA | 1.00× | -0.01% | 0.61× |
| EPIC Multi-News | 1.00× | 0.03% | 0.61× |

## All finer-grain pairs improving both plot axes

All pairs, including regressions, remain in `grain_comparisons.csv`.

| Workload | Coarse → fine | Energy-efficiency gain | Performance/area gain |
|---|---:|---:|---:|
| Llama-2-7B chat | 64 → 32 | 1.18× | 1.28× |
| Llama-2-7B chat | 64 → 16 | 1.28× | 1.29× |
| Llama-2-7B chat | 64 → 8 | 1.32× | 1.02× |
| Llama-2-7B chat | 32 → 16 | 1.09× | 1.01× |
| Llama-3-8B chat | 64 → 32 | 1.14× | 1.22× |
| Llama-3-8B chat | 64 → 16 | 1.21× | 1.18× |
| Phi-2 chat | 64 → 32 | 1.42× | 1.61× |
| Phi-2 chat | 64 → 16 | 1.64× | 1.78× |
| Phi-2 chat | 64 → 8 | 1.68× | 1.41× |
| Phi-2 chat | 32 → 16 | 1.15× | 1.11× |
| Llama-2-7B arXiv | 64 → 32 | 1.23× | 1.34× |
| Llama-2-7B arXiv | 64 → 16 | 1.39× | 1.47× |
| Llama-2-7B arXiv | 64 → 8 | 1.45× | 1.21× |
| Llama-2-7B arXiv | 32 → 16 | 1.13× | 1.09× |
| FlashInfer GQA | 64 → 32 | 1.14× | 1.07× |
| FlashInfer MQA | 64 → 32 | 1.10× | 1.01× |

## Files

- `../analysis/`: geometric-mean plot, full-precision workload metrics and methodology.
- `per_workload.pdf` and twelve workload PNGs: every complete source workload separately.
- `energy_breakdown.csv`: native compute, SRAM, HBM, vector, ICI, static/background and regulator-inclusive energy.
- `connected_vs_independent_ws.csv`: performance, energy and memory traffic at every matched grain.
- All written numbers are copied or computed by this retained script; CSVs and plot coordinates are read back.
