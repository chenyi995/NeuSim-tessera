"""Finalize the requested latest-only publication and retain local archives."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/tessera-20261001'
ARCHIVE = ROOT / 'results/tessera/archived/1001/publication_sources'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    path = ART / 'manifest.json'
    original = json.loads(path.read_text())
    backup = ARCHIVE / 'original_artifact_manifest.json'
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    assert sha(backup) == sha(path)
    kept = []
    extracted = []
    for row in original['files']:
        p = ART / row['file']
        historical = row['file'].startswith('inputs/snapshot/paper_reference/')
        reference_code = row['file'].startswith('calibration/') and p.suffix != '.csv'
        if historical or reference_code:
            target = ARCHIVE / row['file']
            target.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                assert not target.exists()
                p.rename(target)
            assert sha(target) == row['stored_sha256']
            if reference_code:
                text = target.read_text()
                fields = {}
                if p.name == 'make_area_figs.py':
                    for node in ast.parse(text).body:
                        if isinstance(node, ast.Assign):
                            key = getattr(node.targets[0], 'id', '')
                            if key in ('A_FMA','A_LOG','A_SRAM'):
                                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
                                    fields[key] = ast.literal_eval(node.value.left) * ast.literal_eval(node.value.right)
                                else:
                                    fields[key] = ast.literal_eval(node.value)
                else:
                    import re
                    for key, pattern in [('clock_period_ns',r'CLK_PERIOD_NS=([\d.]+)'),
                                         ('array_side',r'localparam int D = (\d+)'),
                                         ('half_period_ns',r'always #([\d.]+)'),
                                         ('plateau_threshold',r'thr = ([\d.]+) \* peak')]:
                        match = re.search(pattern, text)
                        if match:
                            fields[key] = float(match[1])
                extracted.append(dict(source=row['source'],sha256=row['sha256'],fields=fields))
        else:
            kept.append(row)
    (ART / 'calibration/source_parameters.json').write_text(json.dumps(extracted,indent=2)+'\n')
    assert json.loads((ART / 'calibration/source_parameters.json').read_text()) == extracted
    original.update(files=kept,uncompressed_bytes=sum(r['bytes'] for r in kept),
                    stored_bytes=sum(r['stored_bytes'] for r in kept))
    path.write_text(json.dumps(original,indent=2)+'\n')
    assert json.loads(path.read_text()) == original
    # chenyi9: archive every preceding performance experiment locally.
    runs = ROOT / 'results/tessera'
    keep = {'archived','20261001_ppa_two_tilings_v1','20261001_ppa_workload_pairs_v1',
            '20261001_artifact_restore_check','20261001_delivery_checks'}
    archive = runs / 'archived/1001/previous_results'
    archive.mkdir(parents=True, exist_ok=False)
    moved = []
    for src in sorted(runs.iterdir()):
        if src.name in keep:
            continue
        target = archive / src.name
        src.rename(target)
        assert target.exists() and not src.exists()
        moved.append(dict(source=str(src), archived=str(target)))
    record = dict(status='PASS',decision='chenyi9 requested publishing only the all_workloads.pdf run and archiving previous results.',moved=moved)
    (archive / 'archive_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    assert json.loads((archive/'archive_manifest.json').read_text()) == record
    print(json.dumps(dict(status='PASS',published_files=len(kept),archived_paths=len(moved)),indent=2))


if __name__ == '__main__':
    main()
