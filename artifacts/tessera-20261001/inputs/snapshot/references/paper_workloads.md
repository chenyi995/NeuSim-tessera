# Workload and cycle-level revision (2026-09-29)

The pre-edit paper was committed and pushed to `origin/main` as
`5a5fbc2617ec7c67f455b9e2573da8679ad03d32`. This revision changes the
motivation, LLM performance comparison and three EDP comparisons. Existing
RTL/PPA, DNN and multi-tenant results keep their original data.

## Workloads and sources

All six original model/request combinations remain: Llama-2-7B conversation,
Llama-3-8B conversation, Llama-2-70B TP4 conversation, Phi-2 conversation,
Llama-2-7B code, and Llama-2-7B arXiv at 1 QPS. Their recorded batching,
prefill/decode shapes and invocation counts are preserved. The earlier
eight-cohort revision already included the first and fourth traces; exact
operator/phase/MNK/count checks remove those two duplicates. The union has
**12 cohorts, 989,743 input rows and 30,669 distinct MNKs**.

| Added cohorts | Original source and archived version | Evidence type |
|---|---|---|
| GQA decode; MQA decode | [FlashInfer trace](https://huggingface.co/datasets/flashinfer-ai/flashinfer-trace/tree/da915083d4c7c5e61aa3005e3d17ae488e0fc71c) | Public GPU operator inputs; extracted per-request context lengths and grouped-head QK/PV MNKs, not GPU timings or whole-model traces |
| SAMSum; WikiMQA | [CacheBlend](https://github.com/YaoJiayi/CacheBlend/tree/55ad02675939f783a38d579393527d218a7fd581) | Original requests and recomputation policy, converted to prefill GEMM plans |
| HotpotQA; Multi-News | [EPIC](https://github.com/DerekHJH/epic/tree/3204410f1723ed0f39575b4eba8c17578a1ee1a1), [LongBench data](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672) | Original requests and recomputation policy, converted to prefill GEMM plans |

Reuse plans use [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/tree/63a8b081895390a26e140280378bc85ec8bce07a)
at batch 1 / TP 1. All first 200 requests per dataset are considered:
799 of 800 are valid. Multi-News request 37 has invalid upstream gather
indices; its original input and exclusion reason remain archived.
The original six traces come from the saved Vidur extraction, with the
upstream model/request reference pinned to
[Vidur](https://github.com/microsoft/vidur/tree/abae7f63aa857300f5cdc6f5e0d27860cd24721b).

The plots use the formal operator name MQA for the single-KV-head inputs.
These inputs can arise from a GQA parallel shard; they are not labeled a
complete MQA model trace. Frozen CSV IDs and numerical results are unchanged;
the plotters override only the display name. Requests with private KV
caches remain separate; grouping uses only heads sharing the same KV head.

## Figures and definitions

- **Fig. 1**, immediately beneath Table I: one panel at half column width,
  showing per-cohort WS service-cycle shares. GEMV is `min(M,N)=1`. After excluding GEMVs, small
  means `min(M,N,K)<128`, and non-square means `K != N` for the stationary
  operand. Five disjoint categories avoid double counting. The figure does
  not equate call frequency with performance impact.
- **General speedup figure**: all seven designs use their original array
  cycle models and own per-GEMM configuration rules. WS is the 1x reference;
  Planaria-32, SOSA, FlexSA, SISA and both Tessera grains are plotted.
  Speedup is the ratio of invocation-weighted service-cycle sums. This
  replaces the old six-panel arrival-aware LLM makespan plot; the old values
  are not mixed into the new figure.
- **Three EDP panels**, each with grain-32 and grain-8 bars: independent
  arrays/free, physical-square/free, and fixed free mapping with/without
  skew tail. Values greater than one favor the unrestricted no-skew design.
  Workload EDP is `(sum repeat*energy)*(sum repeat*cycles)`.

The main speedup model uses the original DRAM cadence at 1 GHz and
1400 16-bit words/cycle (2.8 TB/s), with one logical A/B load and final C
store and no SRAM capacity limit. Tessera-128-32/8 independently minimize
their original per-GEMM EDP among 9/25 candidates. Planaria-32 selects its
native EDP-optimal composition and reports cycles through the original
paper's Planaria kernel. All designs exclude arrival idle and cross-GEMM
packing from this figure. It reports modeled GEMM service, not end-to-end
inference latency.

The EDP comparisons preserve the separately confirmed revision model:
16,384 PEs, 12,000,000 B compute-side SRAM, unrestricted SRAM ports, 2800 B
per cycle HBM, and 32 pJ/B for both reads and writes. Existing compute/SRAM
and reduction terms remain: 6 pJ/op, 0.95 pJ/read word, 2.2 pJ/write word.
These analytical coefficients differ from gate-level power measurements.
HBM traffic is taken from the corresponding native Planaria-32
full/square/independent mapping under its 12 MiB configuration. Its padded
transfers and 64-bit off-chip partials are preserved. Each family's flow
is fixed across Tessera candidates and both grains; it is not a native
Planaria-8 scheduler. These are geometry-plus-traffic comparisons, while
the skew experiment keeps mapping, energy and traffic exactly paired.

Physical-square padding charges physical PE occupancy/cycles, not phantom
MACs or SRAM accesses. The skew comparison adds the exposed final skew
tail before recomputing HBM overlap; it does not switch the whole schedule
to conventional WS. See the paper methodology and raw result columns for
the exact timing definitions.

## Raw data and reproduction

Authoritative simulator root:
`/data2/chenyi9/fracturableT3/FissionSA`.
New run root: `out/tessera-revision/paper_revision/`.
Scripts and full commands:
`Tessera-revision/experiments/paper_revision/README.md`.

| Path under the run root | Contents |
|---|---|
| `inputs/` | Original rows with source and repeat weights; unique MNKs; duplicate checks; hashes |
| `native/` | All native composition costs and selected native HBM records |
| `service_base/`, `service_planaria/` | Every main-figure per-MNK result and all Tessera candidate costs |
| `service_source_snapshot/` | Hash-verified simulator and experiment sources for the service comparison |
| `ablation_candidates/` | All 104 revision candidates per MNK, with frozen producer/model sources |
| `ablation_g32/`, `ablation_g8/` | Each grain's 21 flow/timing/architecture results per MNK, native traffic templates, paired skew raw results and workload totals |
| `export/` | Full-precision plotted CSVs, motivation MNK/count/cycle statistics and summary manifests |
| `audit/` | Independent arithmetic and aggregation verification |

The sealed previous eight-cohort archives are preserved under
`Tessera-revision/final-three-ablations{,-grain8}/`. They include original
source datasets and extraction provenance. Numerical checks require the
new run to reproduce the same eight cohorts at both grains.

This paper stores copied figure inputs and SHA-256 records in
`fig/plotting/data/revision/`; numeric text macros in `numbers.tex` are
generated from those results. Plot scripts:

```bash
python3 fig/plotting/make_motivation_fig.py
python3 fig/plotting/make_revision_figs.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The plots use vector PDFs, a shared legend, short labels and grouped
workloads, following the compact panel organization of `ref/cogsys.pdf`.
The MQA naming and half-column Fig. 1 update, unchanged CSV hashes, and
current PDF build are recorded in `fig/plotting/data/revision/presentation_update.json`.
Every numerical conclusion must be read in its stated model; the main
speedup and capacity-aware EDP results use different approved memory
models and are not cross-normalized.
