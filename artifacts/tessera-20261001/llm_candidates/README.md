# Latest LLM candidate PPA artifact

[Labeled overview](figures_labeled/overview.pdf) | [One two-panel page per workload](figures_labeled/all_workloads.pdf) | [Experiment scope and conclusions](EXPERIMENT.md)

This is the requested snapshot of the existing four-candidate attention-operator experiment. All candidates and both tiling policies are retained, including unfavorable results. The plots measure complete attention-operator E2E service, not whole-model inference. Falcon uses a model-config-derived KV trajectory; this is not a measured Falcon GPU trace.

The two RAG cohorts, CacheBlend WikiMQA and EPIC HotpotQA, are closest to the requested curve ordering. They do not establish complete Pareto dominance: independent WS can still extend farther right. Falcon MQA and Phi-2 do not meet the desired ordering in both panels. See the complete comparisons and counterexamples in `EXPERIMENT.md`.

Green circles are Tessera, blue squares are Planaria, and black triangles are independent WS. Each connected-design label specifies the minimum fission subarray; each WS label gives array count and dimensions. Coordinates are copied unchanged from `all_points.csv`.

The full operator-cost CSV is losslessly compressed. Source row identities, case multiplicities, hardware configurations, area estimates, plotting scripts, source hashes and verification logs are included. Redundant per-operator JSON files remain in the original local run; their numeric and mapping fields are present in the CSV. Absolute source paths are provenance, not portable paths.

This sub-artifact has its own `manifest.json`. The parent artifact remains a historical snapshot of the earlier experiment and has a separate manifest.

From the repository root:

```bash
python tools/publish_llm_candidate_artifact.py verify
python tools/publish_llm_candidate_artifact.py restore --out results/tessera/llm-candidates-restored
```

For figure regeneration, restore to a fresh directory with `--plot-inputs-only`, then run `plot_and_report.py` and `plot_labeled.py` from that restored directory. This reads saved costs and does not rerun simulation. The frozen `experiment.py` records the original simulation and source extraction, which uses the original FissionSA paths. `cases.csv`, `shapes.json` and `configs.json` retain the complete native-operator profiling inputs independently of those paths.

Every published file is read back and compared to the original SHA-256; the verification tool also recomputes all aggregate coordinates from saved per-operator costs and repeat counts.
