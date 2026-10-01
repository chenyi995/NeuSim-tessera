"""Package the existing Tessera runs without changing any input or result."""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/tessera-20261001'
BASE = ROOT / 'results/tessera/archived/0929/20260929_paper_sram_per_pe'
RUN = ROOT / 'results/tessera/20261001_ppa_two_tilings_v1'
OLD = ROOT / 'results/tessera/20260930_energy_corrected_v1'


def digest(path, compressed=False):
    h = hashlib.sha256()
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    # chenyi9: decision start -- publish the modifications, workloads and results.
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    OUT.mkdir(parents=True, exist_ok=False)
    entries = []

    def copy(src, relative, role):
        src = src.resolve()
        compressed = src.stat().st_size > (1 << 20) and src.suffix in ('.csv', '.jsonl', '.json', '.log')
        target = OUT / (str(relative) + ('.gz' if compressed else ''))
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(src)
        if compressed:
            with src.open('rb') as reader, target.open('xb') as raw:
                with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as writer:
                    shutil.copyfileobj(reader, writer, 1 << 20)
        else:
            with src.open('rb') as reader, target.open('xb') as writer:
                shutil.copyfileobj(reader, writer, 1 << 20)
        assert digest(target, compressed) == before == digest(src)
        assert target.stat().st_size < 95_000_000, target
        restore = str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(Path('results/tessera/external_sources') / src.relative_to(ROOT.parents[1]))
        entries.append(dict(file=str(target.relative_to(OUT)), source=str(src), restore=restore,
                            compression='gzip' if compressed else 'none', role=role,
                            bytes=src.stat().st_size, sha256=before,
                            stored_bytes=target.stat().st_size, stored_sha256=digest(target)))

    def tree(source, target, role, reject=lambda p: False):
        for src in sorted(source.rglob('*')):
            if src.is_file() and '__pycache__' not in src.parts and not reject(src):
                copy(src, Path(target) / src.relative_to(source), role)

    tree(BASE / 'inputs_snapshot', 'inputs/snapshot', 'original workload input')
    for name in ('trace_graph_operators.jsonl', 'trace_graph_batches.jsonl', 'manifest.json', 'analysis/verification.json'):
        copy(BASE / 'speed' / name, Path('inputs/graphs') / name, 'native graph input')
    copy(BASE / 'baseline_verification/bandwidth_grid.json', 'inputs/bandwidth_grid.json', 'bandwidth grid')
    tree(OLD / 'trace_programs_v2', 'inputs/arrivals', 'request arrival and dependency program')
    tree(RUN, 'ppa', 'latest two-policy result', lambda p: 'cost_store' in p.parts or 'archived' in p.parts)
    tree(ROOT / 'results/tessera/20261001_ppa_workload_pairs_v1', 'per_workload', 'latest workload plots')
    tree(ROOT / 'results/tessera/20261001_independent_ws8_energy_audit', 'energy_audit', 'energy accounting audit')
    copy(OLD / 'paper_data_2dp/workload_sources.csv', 'inputs/workload_sources.csv', 'workload source identities')
    calibration = json.loads((ROOT / 'configs/chips/tessera_joules_energy.json').read_text())
    for source in calibration['source_sha256']:
        path = Path(source)
        if path.is_relative_to(ROOT):
            continue
        copy(path, Path('calibration') / path.relative_to(ROOT.parents[1] / 'SA-rtl'), 'Joules calibration source')
    area = ROOT.parents[1] / 'Tessera-HPCA-2026/fig/plotting/make_area_figs.py'
    copy(area, 'calibration/make_area_figs.py', 'area source; power arrays are unused')
    records = dict(status='PASS', original_root=str(ROOT), files=entries,
                   uncompressed_bytes=sum(r['bytes'] for r in entries),
                   stored_bytes=sum(r['stored_bytes'] for r in entries),
                   verification='Every stored file was decompressed/read back and SHA256 matched against its source.')
    target = OUT / 'manifest.json'
    target.write_text(json.dumps(records, indent=2) + '\n')
    assert json.loads(target.read_text()) == records
    with (OUT / 'energy_audit/coefficients.csv').open() as f:
        coefficients = list(csv.DictReader(f))
    c = coefficients[0]
    coef = float(c['sram_pj_per_bit'])
    lines = ['# Tessera reproducibility artifact', '',
        'These are existing analytical simulation outputs, packaged without changing numerical data. '
        'The only published performance result is the run behind `per_workload/figures/all_workloads.pdf`. '
        'Large CSV/JSONL files use lossless gzip. `manifest.json` records the source, restore location, '
        'byte length and SHA256 of every file.', '',
        '## Current energy coefficients', '',
        '| Component | Coefficient | Definition |', '|---|---:|---|']
    for name, item in calibration['designs'].items():
        lines.append(f"| {name} array | {item['pj_per_op']:.3f} pJ/op | One multiply or add; MAC = two ops |")
    lines += [f'| SRAM read or write | {coef:.6f} pJ/bit | Native aggregate dynamic coefficient, before regulator |',
              f'| SRAM 16-bit transfer | {16*coef:.6f} pJ/access | Full-width linear byte accounting |',
              f'| SRAM 32-bit transfer | {32*coef:.6f} pJ/access | Full-width linear byte accounting |',
              f'| SRAM static | {float(c["sram_static_W"]):.8f} W | Charged separately over elapsed time |', '',
        'Compute uses the period-corrected non-SRAM Joules plateau as an activity-energy proxy, '
        'not an isolated dynamic FMA measurement. Native static and regulator terms remain. '
        'Independent WS arrays use the WS coefficient. SISA/SOSA/FlexSA use the Planaria proxy. '
        'SRAM has no single coefficient per arithmetic op: its traffic depends on reuse, tiling, padding and partial sums. '
        'The exploratory N28 SRAM datasheets discussed later have NOT been substituted into these runs.', '',
        '## Results', '',
        '- [All workloads, two panels each](per_workload/figures/all_workloads.pdf)',
        '- [Normal-tiling plot gallery](ppa/native_tail/workloads/README.md)',
        '- [EDP-tiling plot gallery](ppa/joint_edp/workloads/README.md)',
        '- [Two-policy summary](ppa/README.md)',
        '- [SRAM and padding audit](energy_audit/README.md)',
        '- [Known native-tiling issue](ppa/native_tiling_diagnostic.md)',
        '- [Workload sources](inputs/workload_sources.csv)', '',
        'Each policy has thirteen equal-PE configurations and twelve workloads. All have 16384 PEs. '
        'Tessera and Planaria vary the minimum grain over 64, 32, 16 and 8. Independent WS connects '
        'one 128x128, four 64x64, sixteen 32x32, sixty-four 16x16 and 256 8x8 configurations. '
        'Performance and energy efficiency use useful operations. Area includes the scaled array and SRAM proxy.', '',
        '## Inspect and restore', '', 'From the repository root:', '', '```bash',
        'python tools/tessera_artifact.py verify',
        'python tools/tessera_artifact.py restore --out results/tessera/restored-artifact',
        '```', '',
        'The restoration creates a fresh directory and verifies every extracted file. Original absolute paths '
        'inside historical provenance documents identify the original run machine; they are not silently rewritten. '
        'The portable profiling command below reads the restored input/configuration copies directly.', '',
        '```bash',
        'python tools/tessera_artifact.py profile --restored results/tessera/restored-artifact \\',
        '  --policy joint_edp --workers 16 --memory-gb 100 --out results/tessera/new-profile',
        '```', '',
        'This regenerates all operator costs and workload invocation totals with the saved physical configurations. '
        'Arrival/dependency replay is a distinct step; isolated operator sums do not replace the plotted E2E totals. '
        'Use `replay` after profiling (see `--help`); choose `native_tail` for the other tiling policy. '
        'SQLite lookup caches are rebuilt from the included complete operator CSVs; they contain no unique result data.', '']
    (OUT / 'README.md').write_text('\n'.join(lines))
    assert (OUT / 'README.md').read_text() == '\n'.join(lines)
    print(json.dumps({k:v for k,v in records.items() if k!='files'}, indent=2), flush=True)
    print('files', len(entries), flush=True)
    # chenyi9: decision end


if __name__ == '__main__':
    main()
