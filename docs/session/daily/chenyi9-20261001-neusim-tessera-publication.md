# NeuSim Tessera publication

chenyi9 requested publishing the NeuSim changes, workload inputs and results to `chenyi995/NeuSim-tessera`, split into topic commits. The subsequent ruling limits published performance results to the run behind the supplied all_workloads.pdf; earlier runs are archived locally.

Base: `main` at `72056f1e173538a64ff36fa987c1eb1112a670d7`, also the destination main before delivery. Origin now points to the requested fork; the previous origin remains as upstream. Rebase onto origin/main has no divergent upstream changes to replay.

## Commits

| Commit | Change | Files |
|---|---|---:|
| `657895b24416e0369282cb1a782bc9ca6547c2e3` | Add partition-aware Tessera and baseline timing and energy models | 39 |
| `53552615c54f49937cf72c0b7c60c528685a8618` | Preserve reference allocation functions and sourced test inputs | 8 |
| `058feee477e53ab1b48ec18f39e79c4a23df6721` | Add array and scheduler checks with archived tests excluded | 26 |
| `ef6a5f90d25c216637348b413260f0fffc68e4ae` | Add workload profiling mapping and request replay tools | 162 |
| `c2f6144907cbc21eb8facfac7365f7adf7d1766a` | Add lossless artifact export restoration and reproduction commands | 6 |
| `efe04ff1db126579644e50bdec0f024644170569` | Publish the latest two-policy workload inputs and complete result data | 1119 |
| `4424b571c58c666eb8e352852c125fd106318993` | Document Tessera reproduction energy units and the current result set | 25 |

## Validation and data scope

- Active tests: `218 passed, 1 skipped in 3.07s` using `python -m pytest -o addopts='' -q`.
- The first test invocation lacked pytest-cov. The next collected archived tests; collection now excludes them. A source-input path failure after archiving was repaired using hash-checked reference inputs, without weakening the test.
- Artifact: 1119 source files; 200483028 stored bytes. Every file was read/decompressed and SHA256 compared to its source.
- Portable smoke checks: {"native_tail": 26, "joint_edp": 26}; every field matched the original complete operator profiles.
- Plot coordinates independently recomputed for 312 records; published PDF SHA256 `8a8289047e4aba30746b27f12daaa0c00edc1a34e4e05df5f965ec331fb34265` equals the user-specified PDF.
- Archived 30 preceding top-level paths under `results/tessera/archived/1001/previous_results/`; all pre-existing older archives remain.
- Current result sources remain under `results/tessera/20261001_ppa_two_tilings_v1` and `results/tessera/20261001_ppa_workload_pairs_v1`. The published artifact includes complete profiles and reproducible inputs, but omits rebuildable SQLite caches.
- Validation output is local under `results/tessera/20261001_delivery_checks`; original command logs remain under `/tmp/neusim-tessera-*20261001.log`.
- Array coefficients and SRAM units are stated in the artifact and reproduction README. No N28 SRAM candidate was substituted; no performance data was changed.
- The remaining user-requested task is the isolated single-array GEMM/GEMV tiling diagnostic, to be performed after this push. The native tiler may select a short reduction tile that is underpriced by its invariant compute estimate. The published results retain this known modeling/selection interaction.
