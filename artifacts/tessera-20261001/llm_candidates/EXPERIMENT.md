# Exploratory LLM attention PPA candidates

**Status:** simulated with the unchanged NeuSim `merged_load_v2` / `transfer_only` implementation; source/readback and energy checks PASS. These are attention-operator E2E results, not whole-model inference results. All four selected cohorts are shown.

[All four pairs](figures/all_workloads.pdf) | [Overview](figures/overview.png) | [Raw plotted coordinates](all_points.csv)

## Input scope and provenance

| Workload | Full-envelope M | KV length | Head dimension | Retained source scope |
|---|---:|---:|---:|---|
| Falcon-7B MQA decode | 71–71 | 271–302 | [64] | 32 recorded steps in one source trajectory |
| Phi-2 chat prefill | 1–512 | 1–4069 | [80] | 300 requests; 732 chunk/layer-class cases |
| CacheBlend WikiMQA | 4228–30368 | 5869–7592 | [128] | 200 requests; 600 chunk/layer-class cases |
| EPIC HotpotQA | 732–2944 | 2123–19874 | [128] | 200 requests; 200 chunk/layer-class cases |

M is Q × (query heads / KV heads) for one shared KV group. Query blocking may make the executed SRAM-tile M smaller. Different requests retain different KV matrices and are never concatenated into a fictitious shared-KV GEMM.

Falcon is a **derived architecture scenario**: the pinned official Falcon-7B config supplies the MQA heads, head width and layer count; the complete recorded Qwen Dolly-long decode supplies the KV-length trajectory. It is not an observed Falcon GPU run or a claim that both tokenizers produce identical lengths. In `cases.csv`, Falcon `request_id` stores the source step index; `source_manifest.json` therefore counts step identities, not independent requests. The sequence remains serial.

Phi-2 preserves every prefill chunk of its original chat trace, including small residual chunks and the recorded context lengths. The source trace includes contexts beyond the model metadata maximum; they are retained as source workload scenarios, not validated model-quality settings. CacheBlend and EPIC preserve all source requests, selective/full layer classes and repeat counts. Dense projections, FFNs and serving arrival gaps are outside the explicitly measured attention-operator boundary.

Official model source: https://huggingface.co/tiiuae/falcon-7b/blob/main/config.json . Local source filenames, exact source rows and SHA-256 hashes are in `cases.csv` and `source_manifest.json`. The original QK and PV GEMMs are checked against the derived fused-attention dimensions.

## Comparable-grain outcome

All following numbers are derived from this run’s simulated time and energy. Ratios compare the same minimum array side and equal total PEs. Values above one mean higher energy efficiency; they do not by themselves establish a performance-density win.

| Workload | Tiling | Tessera-8 / independent WS-8 energy efficiency | Planaria-8 / independent WS-8 energy efficiency | Tessera-8 / WS-8 performance density |
|---|---|---:|---:|---:|
| Falcon-7B MQA decode | Normal tiling + tail packing | 0.95× | 0.85× | 0.78× |
| Falcon-7B MQA decode | EDP tiling | 1.00× | 0.99× | 0.96× |
| Phi-2 chat prefill | Normal tiling + tail packing | 0.97× | 0.92× | 0.70× |
| Phi-2 chat prefill | EDP tiling | 1.12× | 1.08× | 0.95× |
| CacheBlend WikiMQA | Normal tiling + tail packing | 1.17× | 1.14× | 0.86× |
| CacheBlend WikiMQA | EDP tiling | 1.19× | 1.16× | 0.94× |
| EPIC HotpotQA | Normal tiling + tail packing | 1.20× | 1.16× | 0.89× |
| EPIC HotpotQA | EDP tiling | 1.21× | 1.18× | 0.95× |

No workload or individual point is removed because it fails the desired ordering. The two RAG cohorts show the more useful energy trend in both mapping policies; Falcon MQA and Phi-2 remain counterexamples to the assumption that moderate M plus irregular K/N is sufficient. The independent WS curve can still extend farther right because the connected designs pay more area and the native bulk/tail policy need not minimize latency.

`comparisons.csv` includes SRAM traffic and padded-compute ratios, separating energy benefit from speed/area benefit. `dominance.csv` checks strict two-coordinate dominance of each of the five independent-WS points by measured Tessera/Planaria points. It does not interpolate or imply that every black point is dominated.

## Unchanged experimental accounting

All families use 128×128 total PEs. Tessera/Planaria minimum side: 64, 32, 16, 8. Independent WS: one 128×128, four 64×64, sixteen 32×32, sixty-four 16×16, or 256 8×8 arrays. The current RTL-derived area proxy and finite-SRAM/HBM energy configuration are copied exactly by `run_tessera_ppa.configurations`. Both padded arithmetic and padded SRAM reads are charged. No transpose-fill charge is reintroduced.

The normal policy retains its full-array bulk and packed tail; the EDP policy retains joint legal-array / SRAM-transfer selection. Current within-round packing is enabled. These operator invocations supply no independent arrival stream for asynchronous request overlap; source decode dependencies remain serial.

The axes use useful MACs × 2: x = useful operations / E2E time / area, y = useful operations / total energy. Straight segments connect the measured grains on linear axes; no fitted semicircle or coordinate transformation is used.

## MoE dimension check

The original OLMoE routing trace is read separately in the retained extraction script. Its gate/up/down matrices vary in routed-token M, while each operator’s K and N stay fixed. This source therefore does not supply irregular K/N merely by using MoE. See `source_manifest.json:moe` for the complete observed sets. The MoE trace is an inspected source, not an additional PPA experiment.

## Reproduction and checks

From the NeuSim root, run `experiment.py` followed by `plot_and_report.py` in a fresh result directory. Output is append-only. This execution used 24 worker CPUs with a 5 GB per-worker address-space cap, within the authorized 64 CPU / 200 GB budget. `verification.json` records the exact engine and output hashes; `plot_verification.json` verifies the plotted coordinate source.
