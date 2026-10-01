"""Preserve the superseded energy-only experiment before native-tail support."""
import hashlib,json,shutil
from pathlib import Path

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[2]
records=[]
for relative in ('neusim/run_scripts/run_tessera_ppa.py','neusim/run_scripts/run_tessera_joint.py',
    'neusim/run_scripts/report_tessera_ppa.py','neusim/run_scripts/run_tessera_partitioned.py',
    'neusim/npusim/backend/tessera_partitioned.py','neusim/run_scripts/tessera_request_dispatch.py'):
    src=ROOT/relative;dst=src.parent/'archived/1001/ppa_before_native_tail'/src.name
    dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists()
    shutil.copyfile(src,dst);assert src.read_bytes()==dst.read_bytes()
    records.append(dict(source=str(src),snapshot=str(dst),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
(RUN/'source_snapshots.json').write_text(json.dumps(records,indent=2)+'\n')
assert json.loads((RUN/'source_snapshots.json').read_text())==records
with (RUN.parent/'20261001_ppa_min_energy_v1/interruption.json').open('x') as f:
    json.dump(dict(status='INTERRUPTED',reason='chenyi9 replaced energy-only mapping with native bulk/tail and joint EDP plots.',
        replacement_run=str(RUN),stopped_session=10746),f,indent=2)
