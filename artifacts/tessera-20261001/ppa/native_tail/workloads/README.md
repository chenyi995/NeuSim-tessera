# Native tiling + tail packing: separate workloads

All metrics are derived from completed analytical NeuSim simulations. The same 128×128 total PEs, finite SRAM, HBM4, revision energy coefficients, padded work, arrivals and dependencies apply to every design. The WS line connects 1×128², 4×64², 16×32², 64×16² and 256×8² independent arrays. WS area is the equal-PE monolithic proxy requested by chenyi9; RTL affects area only. Tessera enables both within-round packing and asynchronous use of free physical regions.

## Tessera grain 32 to grain 8

| Workload | E2E speedup | Energy reduction | Performance/area gain |
|---|---:|---:|---:|
| Llama-2-7B chat | 2.71× | 65.25% | 1.65× |
| Llama-3-8B chat | 3.10× | 64.38% | 1.88× |
| Llama-2-70B chat | 2.26× | 59.25% | 1.37× |
| Phi-2 chat | 1.42× | 35.24% | 0.86× |
| Llama-2-7B code | 2.65× | 64.93% | 1.61× |
| Llama-2-7B arXiv | 3.05× | 66.86% | 1.85× |
| FlashInfer GQA | 1.02× | 1.59% | 0.62× |
| FlashInfer MQA | 1.01× | 1.05% | 0.61× |
| CacheBlend SAMSum | 2.04× | 62.35% | 1.24× |
| CacheBlend WikiMQA | 1.40× | 48.09% | 0.85× |
| EPIC HotpotQA | 1.57× | 54.57% | 0.95× |
| EPIC Multi-News | 2.73× | 66.82% | 1.66× |

## All finer-grain pairs improving both plot axes

All pairs, including regressions, remain in `grain_comparisons.csv`.

| Workload | Coarse → fine | Energy-efficiency gain | Performance/area gain |
|---|---:|---:|---:|
| Llama-2-7B chat | 64 → 32 | 2.02× | 2.01× |
| Llama-2-7B chat | 64 → 16 | 3.74× | 3.14× |
| Llama-2-7B chat | 64 → 8 | 5.82× | 3.30× |
| Llama-2-7B chat | 32 → 16 | 1.85× | 1.57× |
| Llama-2-7B chat | 32 → 8 | 2.88× | 1.65× |
| Llama-2-7B chat | 16 → 8 | 1.56× | 1.05× |
| Llama-3-8B chat | 64 → 32 | 2.35× | 2.63× |
| Llama-3-8B chat | 64 → 16 | 4.36× | 4.49× |
| Llama-3-8B chat | 64 → 8 | 6.59× | 4.94× |
| Llama-3-8B chat | 32 → 16 | 1.86× | 1.71× |
| Llama-3-8B chat | 32 → 8 | 2.81× | 1.88× |
| Llama-3-8B chat | 16 → 8 | 1.51× | 1.10× |
| Llama-2-70B chat | 64 → 32 | 1.87× | 1.82× |
| Llama-2-70B chat | 64 → 16 | 3.16× | 2.48× |
| Llama-2-70B chat | 64 → 8 | 4.58× | 2.49× |
| Llama-2-70B chat | 32 → 16 | 1.69× | 1.37× |
| Llama-2-70B chat | 32 → 8 | 2.45× | 1.37× |
| Llama-2-70B chat | 16 → 8 | 1.45× | 1.00× |
| Phi-2 chat | 64 → 32 | 1.41× | 1.51× |
| Phi-2 chat | 64 → 16 | 1.16× | 1.19× |
| Phi-2 chat | 64 → 8 | 2.18× | 1.30× |
| Phi-2 chat | 16 → 8 | 1.89× | 1.09× |
| Llama-2-7B code | 64 → 32 | 1.88× | 1.68× |
| Llama-2-7B code | 64 → 16 | 3.44× | 2.59× |
| Llama-2-7B code | 64 → 8 | 5.37× | 2.69× |
| Llama-2-7B code | 32 → 16 | 1.83× | 1.55× |
| Llama-2-7B code | 32 → 8 | 2.85× | 1.61× |
| Llama-2-7B code | 16 → 8 | 1.56× | 1.04× |
| Llama-2-7B arXiv | 64 → 32 | 2.23× | 2.30× |
| Llama-2-7B arXiv | 64 → 16 | 4.31× | 3.96× |
| Llama-2-7B arXiv | 64 → 8 | 6.73× | 4.26× |
| Llama-2-7B arXiv | 32 → 16 | 1.93× | 1.72× |
| Llama-2-7B arXiv | 32 → 8 | 3.02× | 1.85× |
| Llama-2-7B arXiv | 16 → 8 | 1.56× | 1.08× |
| CacheBlend SAMSum | 64 → 32 | 1.80× | 1.56× |
| CacheBlend SAMSum | 64 → 16 | 3.16× | 2.01× |
| CacheBlend SAMSum | 64 → 8 | 4.77× | 1.92× |
| CacheBlend SAMSum | 32 → 16 | 1.76× | 1.29× |
| CacheBlend SAMSum | 32 → 8 | 2.66× | 1.24× |
| CacheBlend WikiMQA | 64 → 32 | 1.63× | 1.28× |
| CacheBlend WikiMQA | 64 → 16 | 2.33× | 1.29× |
| CacheBlend WikiMQA | 64 → 8 | 3.14× | 1.08× |
| CacheBlend WikiMQA | 32 → 16 | 1.43× | 1.01× |
| EPIC HotpotQA | 64 → 32 | 1.81× | 1.45× |
| EPIC HotpotQA | 64 → 16 | 2.80× | 1.54× |
| EPIC HotpotQA | 64 → 8 | 3.98× | 1.39× |
| EPIC HotpotQA | 32 → 16 | 1.55× | 1.06× |
| EPIC Multi-News | 64 → 32 | 1.93× | 1.73× |
| EPIC Multi-News | 64 → 16 | 3.18× | 2.28× |
| EPIC Multi-News | 64 → 8 | 5.83× | 2.86× |
| EPIC Multi-News | 32 → 16 | 1.64× | 1.32× |
| EPIC Multi-News | 32 → 8 | 3.01× | 1.66× |
| EPIC Multi-News | 16 → 8 | 1.83× | 1.26× |

## Files

- `../analysis/`: geometric-mean plot, full-precision workload metrics and methodology.
- `per_workload.pdf` and twelve workload PNGs: every complete source workload separately.
- `energy_breakdown.csv`: native compute, SRAM, HBM, vector, ICI, static/background and regulator-inclusive energy.
- `connected_vs_independent_ws.csv`: performance, energy and memory traffic at every matched grain.
- All written numbers are copied or computed by this retained script; CSVs and plot coordinates are read back.
