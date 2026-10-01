"""Preserve the original audit and generate an indexed read-only query version."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / 'inspect_tiling.py').read_text()
query = 'where arch=? and kind=? and oid=?'
arguments = "(arch, w['kind'], int(w['operator_id']))"
assert source.count(query) == source.count(arguments) == 2
updated = source.replace(query, 'where arch=? and bw=? and kind=? and oid=?').replace(
    arguments, "(arch, int(target['bandwidth_bytes_per_second']), w['kind'], int(w['operator_id']))")
out = HERE / 'inspect_tiling_indexed.py'
with out.open('x') as handle:
    handle.write(updated)
assert out.read_text() == updated
