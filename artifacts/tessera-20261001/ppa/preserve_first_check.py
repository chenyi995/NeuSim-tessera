"""Retain the initial native-tail source and failed verification for diagnosis."""
import hashlib,json,shutil
from pathlib import Path
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[2];records=[]
for src in (ROOT/'neusim/npusim/backend/tessera_native_tail.py',RUN/'verify_native_tail.py'):
    dst=src.parent/'archived/1001/native_tail_first_check'/src.name
    dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists()
    shutil.copyfile(src,dst);assert src.read_bytes()==dst.read_bytes()
    records.append(dict(source=str(src),snapshot=str(dst),sha256=hashlib.sha256(src.read_bytes()).hexdigest()))
(RUN/'first_check_snapshots.json').write_text(json.dumps(records,indent=2)+'\n')
assert json.loads((RUN/'first_check_snapshots.json').read_text())==records
