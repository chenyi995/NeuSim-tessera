"""Publish, verify and restore the requested LLM candidate PPA snapshot."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts/tessera-20261001/llm_candidates'
SOURCE = ROOT / 'results/tessera/20261001_llm_candidate_ppa_v1'


def digest(path, compressed=False):
    h = hashlib.sha256()
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def text(path):
    if path.exists():
        return path.read_text()
    with gzip.open(Path(str(path) + '.gz'), 'rt') as f:
        return f.read()


def verify(artifact):
    manifest = json.loads((artifact / 'manifest.json').read_text())
    assert digest(artifact / 'README.md') == manifest['readme_sha256']
    for r in manifest['files']:
        p = artifact / r['file']
        assert digest(p) == r['stored_sha256'], p
        assert digest(p, r['compression'] == 'gzip') == r['sha256'], p
    from io import StringIO
    points = list(csv.DictReader(StringIO(text(artifact / 'all_points.csv'))))
    cases = list(csv.DictReader(StringIO(text(artifact / 'cases.csv'))))
    area = json.loads(text(artifact / 'area.json'))
    cost = {(r['tiling_policy'], r['architecture'], r['operator_id']): r
            for r in csv.DictReader(StringIO(text(artifact / 'operator_costs.csv')))}
    weights = defaultdict(lambda: defaultdict(int))
    for r in cases:
        weights[r['workload_id']][r['operator_id']] += int(r['repeat'])
    for r in points:
        cc = [(n, cost[r['tiling_policy'], r['architecture'], oid])
              for oid, n in weights[r['workload_id']].items()]
        seconds = math.fsum(n * float(v['time_ns']) for n, v in cc) * 1e-9
        energy = math.fsum(n * float(v['energy_J']) for n, v in cc)
        macs = sum(n * int(v['useful_macs']) for n, v in cc)
        assert macs == int(float(r['useful_macs']))
        assert math.isclose(seconds, float(r['seconds']), rel_tol=2e-12)
        assert math.isclose(energy, float(r['energy_J']), rel_tol=2e-12)
        assert float(r['area_mm2']) == area[r['architecture']]['area_mm2']
        assert math.isclose(macs * 2 / seconds / float(r['area_mm2']) / 1e9,
                            float(r['performance_density_gops_per_mm2']), rel_tol=2e-12)
        assert math.isclose(macs * 2 / energy / 1e9,
                            float(r['energy_efficiency_gops_per_w']), rel_tol=2e-12)
    indexed = {(r['workload_id'], r['tiling_policy'], r['architecture']): r for r in points}
    labels = list(csv.DictReader(StringIO(text(artifact / 'figures_labeled/point_labels.csv'))))
    assert len(labels) == len(points)
    for r in labels:
        p = indexed[r['workload_id'], r['tiling_policy'], r['architecture']]
        assert float(r['x']) == float(p['performance_density_gops_per_mm2'])
        assert float(r['y']) == float(p['energy_efficiency_gops_per_w'])
        a = area[r['architecture']]; g = a['grain']
        expected = (f'Fission {g}×{g}' if a['family'] in ('Tessera', 'Planaria')
                    else f"{a['subarrays']} × ({g}×{g})")
        assert r['label'] == expected
    print(f"PASS: {len(manifest['files'])} files, {len(cost)} operator costs, "
          f"{len(points)} aggregate points and {len(labels)} labels.")
    return manifest


def export(source, artifact):
    # chenyi9: decision start -- publish current code and the labeled conclusion figures.
    original = json.loads((source / 'verification.json').read_text())
    assert original['status'] == 'PASS'
    for name, h in original['output_sha256'].items():
        assert digest(source / name) == h
    for name in ('plot_verification.json', 'figures_labeled/verification.json'):
        check = json.loads((source / name).read_text())
        assert check['status'] == 'PASS'
        base = source if name == 'plot_verification.json' else source / 'figures_labeled'
        for filename, h in check['artifacts'].items():
            assert digest(base / filename) == h
    artifact.mkdir(parents=True, exist_ok=False)
    entries = []
    for src in sorted(source.rglob('*')):
        relative = src.relative_to(source)
        # Per-operator JSON duplicates the complete CSV retained losslessly below.
        if not src.is_file() or 'raw' in relative.parts or '__pycache__' in relative.parts:
            continue
        compressed = src.suffix == '.csv' and src.stat().st_size > 1 << 20
        dest_relative = Path('EXPERIMENT.md') if relative == Path('README.md') else relative
        dest = artifact / (str(dest_relative) + ('.gz' if compressed else ''))
        dest.parent.mkdir(parents=True, exist_ok=True)
        h = digest(src)
        with src.open('rb') as reader, dest.open('xb') as out:
            if compressed:
                with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0) as writer:
                    shutil.copyfileobj(reader, writer)
            else:
                shutil.copyfileobj(reader, out)
        assert digest(dest, compressed) == h == digest(src)
        assert dest.stat().st_size < 95_000_000
        entries.append(dict(file=str(dest.relative_to(artifact)), restore=str(relative),
                            source=str(src.resolve()), compression='gzip' if compressed else 'none',
                            sha256=h, stored_sha256=digest(dest), bytes=src.stat().st_size,
                            stored_bytes=dest.stat().st_size))
    report = '\n'.join([
        '# Latest LLM candidate PPA artifact', '',
        '[Labeled overview](figures_labeled/overview.pdf) | '
        '[One two-panel page per workload](figures_labeled/all_workloads.pdf) | '
        '[Experiment scope and conclusions](EXPERIMENT.md)', '',
        'This is the requested snapshot of the existing four-candidate attention-operator experiment. '
        'All candidates and both tiling policies are retained, including unfavorable results. '
        'The plots measure complete attention-operator E2E service, not whole-model inference. '
        'Falcon uses a model-config-derived KV trajectory; this is not a measured Falcon GPU trace.', '',
        'The two RAG cohorts, CacheBlend WikiMQA and EPIC HotpotQA, are closest to the requested '
        'curve ordering. They do not establish complete Pareto dominance: independent WS can still '
        'extend farther right. Falcon MQA and Phi-2 do not meet the desired ordering in both panels. '
        'See the complete comparisons and counterexamples in `EXPERIMENT.md`.', '',
        'Green circles are Tessera, blue squares are Planaria, and black triangles are independent WS. '
        'Each connected-design label specifies the minimum fission subarray; each WS label gives '
        'array count and dimensions. Coordinates are copied unchanged from `all_points.csv`.', '',
        'The full operator-cost CSV is losslessly compressed. Source row identities, case multiplicities, '
        'hardware configurations, area estimates, plotting scripts, source hashes and verification logs '
        'are included. Redundant per-operator JSON files remain in the original local run; their numeric '
        'and mapping fields are present in the CSV. Absolute source paths are provenance, not portable paths.', '',
        'This sub-artifact has its own `manifest.json`. The parent artifact remains a historical snapshot '
        'of the earlier experiment and has a separate manifest.', '',
        'From the repository root:', '', '```bash',
        'python tools/publish_llm_candidate_artifact.py verify',
        'python tools/publish_llm_candidate_artifact.py restore --out results/tessera/llm-candidates-restored',
        '```', '',
        'For figure regeneration, restore to a fresh directory with `--plot-inputs-only`, then run '
        '`plot_and_report.py` and `plot_labeled.py` from that restored directory. This reads saved costs '
        'and does not rerun simulation. The frozen `experiment.py` records the original simulation and '
        'source extraction, which uses the original FissionSA paths. `cases.csv`, `shapes.json` and '
        '`configs.json` retain the complete native-operator profiling inputs independently of those paths.', '',
        'Every published file is read back and compared to the original SHA-256; the verification tool '
        'also recomputes all aggregate coordinates from saved per-operator costs and repeat counts.', ''])
    with (artifact / 'README.md').open('x') as f:
        f.write(report)
    assert (artifact / 'README.md').read_text() == report
    manifest = dict(status='PASS', source=str(source.resolve()), files=entries,
                    readme_sha256=digest(artifact / 'README.md'))
    with (artifact / 'manifest.json').open('x') as f:
        json.dump(manifest, f, indent=2)
    assert json.loads((artifact / 'manifest.json').read_text()) == manifest
    verify(artifact)
    # chenyi9: decision end


def restore(artifact, out, plot_inputs_only=False):
    manifest = verify(artifact)
    out.mkdir(parents=True, exist_ok=False)
    for r in manifest['files']:
        rel = Path(r['restore'])
        if plot_inputs_only and (rel.parts[0].startswith('figures') or rel.name in
                                ('README.md', 'comparisons.csv', 'dominance.csv', 'plot_verification.json')):
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = artifact / r['file']
        with (gzip.open(src, 'rb') if r['compression'] == 'gzip' else src.open('rb')) as reader:
            with dest.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
        assert digest(dest) == r['sha256']
    print('PASS: restored files match their original uncompressed SHA-256.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('export', 'verify', 'restore'))
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--artifact', type=Path, default=ARTIFACT)
    p.add_argument('--out', type=Path)
    p.add_argument('--plot-inputs-only', action='store_true')
    a = p.parse_args()
    if a.action == 'export':
        export(a.source, a.artifact)
    elif a.action == 'verify':
        verify(a.artifact)
    else:
        if a.out is None:
            p.error('--out is required for restore')
        restore(a.artifact, a.out, a.plot_inputs_only)


if __name__ == '__main__':
    main()
