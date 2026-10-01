# Per-workload normal and EDP tiling figures

Status: PASS. Each sourced workload has two separate figures and a side-by-side comparison. These plots use the completed analytical NeuSim results from `20261001_ppa_two_tilings_v1`; there is no new simulation or mapping selection.

[All workloads, one comparison per PDF page](all_workloads.pdf) · [HTML gallery](index.html) · [Full-precision plotted data](all_points.csv)

The horizontal axis is useful E2E GOPS divided by array-plus-SRAM area; the vertical axis is useful GOPS/W. Each workload is plotted separately in absolute units. Both axes are linear. Axis limits are chosen separately for each panel and are printed on the axes. The plot area is wider than before; neither nonlinear transforms nor fitted curves are used. Straight segments connect the recorded grain points in physical-grain order.

Tessera and Planaria sweep grains 64, 32, 16, 8. Independent WS connects 1×128×128, 4×64×64, 16×32×32, 64×16×16, and 256×8×8 arrays. Every configuration has 128×128 total PEs. Labels next to markers are the fission grain or independent WS subarray side. Tessera packing, recorded arrivals, finite SRAM, padded accesses and revision energy accounting remain as in the completed run. RTL contributes area estimates only.

Normal tiling is the native NeuSim tiler plus full-array bulk execution and fractured tail packing. Its known small-K-tile choices remain unchanged. The other panel uses the completed joint EDP mappings.

| Workload | Normal tiling | EDP tiling | Comparison | Data |
|---|---|---|---|---|
| Llama-2-7B chat | [PNG](llama2_conv/normal.png) · [PDF](llama2_conv/normal.pdf) | [PNG](llama2_conv/edp.png) · [PDF](llama2_conv/edp.pdf) | [PNG](llama2_conv/comparison.png) · [PDF](llama2_conv/comparison.pdf) | [CSV](llama2_conv/data.csv) |
| Llama-3-8B chat | [PNG](llama3_conv/normal.png) · [PDF](llama3_conv/normal.pdf) | [PNG](llama3_conv/edp.png) · [PDF](llama3_conv/edp.pdf) | [PNG](llama3_conv/comparison.png) · [PDF](llama3_conv/comparison.pdf) | [CSV](llama3_conv/data.csv) |
| Llama-2-70B chat | [PNG](llama2_70b_conv/normal.png) · [PDF](llama2_70b_conv/normal.pdf) | [PNG](llama2_70b_conv/edp.png) · [PDF](llama2_70b_conv/edp.pdf) | [PNG](llama2_70b_conv/comparison.png) · [PDF](llama2_70b_conv/comparison.pdf) | [CSV](llama2_70b_conv/data.csv) |
| Phi-2 chat | [PNG](phi2_conv/normal.png) · [PDF](phi2_conv/normal.pdf) | [PNG](phi2_conv/edp.png) · [PDF](phi2_conv/edp.pdf) | [PNG](phi2_conv/comparison.png) · [PDF](phi2_conv/comparison.pdf) | [CSV](phi2_conv/data.csv) |
| Llama-2-7B code | [PNG](llama2_code/normal.png) · [PDF](llama2_code/normal.pdf) | [PNG](llama2_code/edp.png) · [PDF](llama2_code/edp.pdf) | [PNG](llama2_code/comparison.png) · [PDF](llama2_code/comparison.pdf) | [CSV](llama2_code/data.csv) |
| Llama-2-7B arXiv | [PNG](llama2_arxiv/normal.png) · [PDF](llama2_arxiv/normal.pdf) | [PNG](llama2_arxiv/edp.png) · [PDF](llama2_arxiv/edp.pdf) | [PNG](llama2_arxiv/comparison.png) · [PDF](llama2_arxiv/comparison.pdf) | [CSV](llama2_arxiv/data.csv) |
| FlashInfer GQA | [PNG](gqa_decode/normal.png) · [PDF](gqa_decode/normal.pdf) | [PNG](gqa_decode/edp.png) · [PDF](gqa_decode/edp.pdf) | [PNG](gqa_decode/comparison.png) · [PDF](gqa_decode/comparison.pdf) | [CSV](gqa_decode/data.csv) |
| FlashInfer MQA | [PNG](single_kv_decode/normal.png) · [PDF](single_kv_decode/normal.pdf) | [PNG](single_kv_decode/edp.png) · [PDF](single_kv_decode/edp.pdf) | [PNG](single_kv_decode/comparison.png) · [PDF](single_kv_decode/comparison.pdf) | [CSV](single_kv_decode/data.csv) |
| CacheBlend SAMSum | [PNG](cacheblend_samsum/normal.png) · [PDF](cacheblend_samsum/normal.pdf) | [PNG](cacheblend_samsum/edp.png) · [PDF](cacheblend_samsum/edp.pdf) | [PNG](cacheblend_samsum/comparison.png) · [PDF](cacheblend_samsum/comparison.pdf) | [CSV](cacheblend_samsum/data.csv) |
| CacheBlend WikiMQA | [PNG](cacheblend_wikimqa/normal.png) · [PDF](cacheblend_wikimqa/normal.pdf) | [PNG](cacheblend_wikimqa/edp.png) · [PDF](cacheblend_wikimqa/edp.pdf) | [PNG](cacheblend_wikimqa/comparison.png) · [PDF](cacheblend_wikimqa/comparison.pdf) | [CSV](cacheblend_wikimqa/data.csv) |
| EPIC HotpotQA | [PNG](epic_hotpotqa/normal.png) · [PDF](epic_hotpotqa/normal.pdf) | [PNG](epic_hotpotqa/edp.png) · [PDF](epic_hotpotqa/edp.pdf) | [PNG](epic_hotpotqa/comparison.png) · [PDF](epic_hotpotqa/comparison.pdf) | [CSV](epic_hotpotqa/data.csv) |
| EPIC Multi-News | [PNG](epic_multi_news/normal.png) · [PDF](epic_multi_news/normal.pdf) | [PNG](epic_multi_news/edp.png) · [PDF](epic_multi_news/edp.pdf) | [PNG](epic_multi_news/comparison.png) · [PDF](epic_multi_news/comparison.pdf) | [CSV](epic_multi_news/data.csv) |

Validation: source CSV hashes match the completed run; every plotted coordinate is independently recomputed from useful MACs, elapsed time, system energy and total area. Written CSVs are read back and all line coordinates are checked against them. `../plot_workloads.py` reproduces this display.
