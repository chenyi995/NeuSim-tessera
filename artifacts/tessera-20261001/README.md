# Tessera reproducibility artifact

These are existing analytical simulation outputs, packaged without changing numerical data. The only published performance result is the run behind `per_workload/figures/all_workloads.pdf`. Large CSV/JSONL files use lossless gzip. `manifest.json` records the source, restore location, byte length and SHA256 of every file.

## Current energy coefficients

| Component | Coefficient | Definition |
|---|---:|---|
| WS array | 6.302 pJ/op | One multiply or add; MAC = two ops |
| Tessera array | 6.303 pJ/op | One multiply or add; MAC = two ops |
| Planaria array | 6.485 pJ/op | One multiply or add; MAC = two ops |
| SRAM read or write | 0.255247 pJ/bit | Native aggregate dynamic coefficient, before regulator |
| SRAM 16-bit transfer | 4.083958 pJ/access | Full-width linear byte accounting |
| SRAM 32-bit transfer | 8.167917 pJ/access | Full-width linear byte accounting |
| SRAM static | 24.21353615 W | Charged separately over elapsed time |

Compute uses the period-corrected non-SRAM Joules plateau as an activity-energy proxy, not an isolated dynamic FMA measurement. Native static and regulator terms remain. Independent WS arrays use the WS coefficient. SISA/SOSA/FlexSA use the Planaria proxy. SRAM has no single coefficient per arithmetic op: its traffic depends on reuse, tiling, padding and partial sums. The exploratory N28 SRAM datasheets discussed later have NOT been substituted into these runs.

## Results

- [All workloads, two panels each](per_workload/figures/all_workloads.pdf)
- [Normal-tiling plot gallery](ppa/native_tail/workloads/README.md)
- [EDP-tiling plot gallery](ppa/joint_edp/workloads/README.md)
- [Two-policy summary](ppa/README.md)
- [SRAM and padding audit](energy_audit/README.md)
- [Known native-tiling issue](ppa/native_tiling_diagnostic.md)
- [Workload sources](inputs/workload_sources.csv)

Each policy has thirteen equal-PE configurations and twelve workloads. All have 16384 PEs. Tessera and Planaria vary the minimum grain over 64, 32, 16 and 8. Independent WS connects one 128x128, four 64x64, sixteen 32x32, sixty-four 16x16 and 256 8x8 configurations. Performance and energy efficiency use useful operations. Area includes the scaled array and SRAM proxy.

## Inspect and restore

From the repository root:

```bash
python tools/tessera_artifact.py verify
python tools/tessera_artifact.py restore --out results/tessera/restored-artifact
```

The restoration creates a fresh directory and verifies every extracted file. Original absolute paths inside historical provenance documents identify the original run machine; they are not silently rewritten. The portable profiling command below reads the restored input/configuration copies directly.

```bash
python tools/tessera_artifact.py profile --restored results/tessera/restored-artifact \
  --policy joint_edp --workers 16 --memory-gb 100 --out results/tessera/new-profile
```

This regenerates all operator costs and workload invocation totals with the saved physical configurations. Arrival/dependency replay is a distinct step; isolated operator sums do not replace the plotted E2E totals. Use `replay` after profiling (see `--help`); choose `native_tail` for the other tiling policy. SQLite lookup caches are rebuilt from the included complete operator CSVs; they contain no unique result data.
