# Corrected PPA and labeled LLM candidate publication

chenyi9 requested committing and pushing the current NeuSim code and the latest conclusions associated with `/data2/chenyi9/fracturableT3/simulators/NeuSim/results/tessera/20261001_llm_candidate_ppa_v1/figures_labeled/overview.pdf`. The delivery target is `git@github.com:chenyi995/NeuSim-tessera.git`, branch `main`.

Starting point: `main` at `4c78483587068111da9018bb90da1fcc56b77f95` on 2026-10-01; the remote main matched this commit before the delivery.

## Commits

| Commit | Change |
|---|---|
| `0ca60ecef4f233506f4129fc15416814ab950a5d` | Correct continuous array timing and preserve earlier backend snapshots |
| `6be85aa11ba7936f6106d1e9335135e8b6e21e7b` | Add regression checks for transfer tiling and continuous array timing |
| `792ce7b1fb583ebc82d1af38ecd742c9d93a4978` | Apply corrected timing and independent WS area in PPA workflows with prior scripts archived |
| `d8ea579cf29ef0bc46e2b0e93ef8c96e921e5b7b` | Add lossless publication and verification of the LLM candidate artifact |
| `021316d73fb46d61cc5fbfe4301bf9f55e0f9864` | Preserve LLM candidate extraction and fission label plotting scripts |
| `dc15f5652ae71e4fed496d0d2fca9f64cc891bdd` | Publish all LLM candidate costs and labeled PPA conclusions |
| `d6e8449dd2ddd9016835ee5b79ff1036b4a8c274` | Document corrected PPA accounting and link the latest conclusions while archiving earlier notes |

## Rulings and resulting behavior

chenyi9 ruled that SRAM transfer tiles do not restart array computation, merged Planaria regions load as complete arrays, and Tessera transpose ingress overlaps HBM transfer. The implementation keeps native traffic and finite-capacity accounting, applies continuous weight-bank timing, and uses the same timing mode during selection and request replay.

chenyi9 ruled that independent WS uses matching-grain Planaria SRAM area with WS FMA and logic area. All configurations retain the same total PE count; the area proxy is a derived estimate, not a new synthesis result.

chenyi9 requested plausible LLM candidates under both tiling policies, then requested explicit per-point fission labels. The published snapshot retains every simulated candidate and the unsuccessful cases. Green circles identify Tessera, blue squares Planaria, and black triangles independent WS; point labels give fission size or independent-array count and dimensions.

## Validation and published data

- Active tests: `241 passed, 1 skipped in 4.22s` using `.venv/bin/python -m pytest -o addopts=''`; raw log `/tmp/chenyi9-neusim-llm-publish-pytest.log`.
- Existing candidate simulation: `1564` source cases, `1071` unique operators and `104` plotted points; all original engine source hashes still match the committed implementation. Original run log: `/data2/chenyi9/fracturableT3/simulators/NeuSim/results/tessera/20261001_llm_candidate_ppa_v1/run.log`.
- Lossless publication: `42` source files, `9067545` stored bytes. The complete operator-cost CSV is compressed; per-operator JSON duplicates remain local.
- `tools/publish_llm_candidate_artifact.py export` verified all source and stored hashes, recomputed each aggregate from saved operator costs and repeat counts, and checked label-to-configuration identity.
- `tools/publish_llm_candidate_artifact.py restore --out results/tessera/20261001_llm_candidate_publication_restore_check` passed a complete decompression/readback check.
- Requested overview SHA-256: `f153f3041ad2dbf2725475470ef116664c8754c6394c088292a1d9e69ad72662`. The source PDF, published artifact and committed Git blob are byte-identical.
- The initial whole-diff whitespace check flagged original CSV CRLF endings and a raw log terminal blank line. These frozen bytes are preserved for source-hash equality; editable code/document whitespace and staged English text checks pass.
- Publication copied existing simulation outputs; no new performance experiment or energy/area tuning was performed for this push.

## Conclusions and limits

CacheBlend WikiMQA and EPIC HotpotQA are the closer matches to the requested curve ordering. The curves do not establish complete Pareto dominance. Falcon MQA and Phi-2 do not meet the desired ordering in both panels and remain in the report. This is complete attention-operator E2E accounting, not whole-model inference. Falcon combines a pinned model configuration with an existing recorded KV-length trajectory and is explicitly labeled as a derived scenario.

Native compute, SRAM, HBM, static and regulator accounting are retained. The current coefficients and their proxy limitations are stated in the reproduction document. The original candidate extraction script uses source-machine paths; the saved cases, shapes, configurations and plotting inputs are bundled. Artifact verification, restoration and figure regeneration use the bundled files.

## Preserved history

The delivery includes the prior transfer-tiling, bank-timing, merged-load and WS-area source snapshots under their adjacent `archived/1001/` directories. They document the fixes already made during the preceding work. README/reproduction text was snapshotted under `archived/1001/llm_candidate_publication/` before updating links and scope. Historical artifacts keep their own manifests; the new `llm_candidates` sub-artifact has a separate manifest.

There is no pending approval for this delivery. The untracked user-provided paper PDF remains local; this publication covers simulator code and the requested experiment materials.
