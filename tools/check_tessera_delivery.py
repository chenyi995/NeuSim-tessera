"""Read back packaged results and compare portable smoke costs to the full run."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/tessera-20261001'
CHECK=ROOT/'results/tessera/20261001_delivery_checks'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        yield from csv.DictReader(f)


def main():
    counts={}
    for policy,name in (('native_tail','profile_native'),('joint_edp','profile_edp')):
        current=list(rows(CHECK/name/'operator_costs.csv'))
        key=lambda r:(r['kind'],r['operator_id'],r['architecture'],r['bandwidth_bytes_per_second'])
        want={key(r):r for r in current}
        found={}
        for row in rows(ART/'ppa'/policy/'main_costs/operator_costs.csv.gz'):
            identity=key(row)
            if identity in want:
                found[identity]=row
        assert set(found)==set(want)
        for identity,row in want.items():
            for field,value in row.items():
                if field=='mapping_json':
                    assert json.loads(value)==json.loads(found[identity][field]),(identity,field)
                else:
                    assert value==found[identity][field],(identity,field,value,found[identity][field])
        counts[policy]=len(want)
        totals=list(rows(ART/'ppa'/policy/'analysis/workload_metrics.csv'))
        assert len(totals)==156
        for r in totals:
            ops=2*int(r['useful_macs'])*int(r['chips'])
            assert math.isclose(float(r['performance_density_gops_per_mm2']),ops/float(r['seconds'])/float(r['total_area_mm2'])/1e9,rel_tol=2e-12)
            assert math.isclose(float(r['energy_efficiency_gops_per_w']),ops/float(r['system_energy_J'])/1e9,rel_tol=2e-12)
    original=ROOT/'results/tessera/20261001_ppa_workload_pairs_v1/figures/all_workloads.pdf'
    published=ART/'per_workload/figures/all_workloads.pdf'
    assert sha(original)==sha(published)
    refs=json.loads((ROOT/'references/manifest.json').read_text())
    for r in refs:
        assert sha(ROOT/r['file'])==r['sha256']
    result=dict(status='PASS',portable_smoke_records=counts,plotted_records=312,
                original_pdf_sha256=sha(original),published_pdf_sha256=sha(published),
                evidence='Both policy smoke profiles match every field of the corresponding complete original operator profiles; all plotted coordinates recomputed.')
    with (CHECK/'readback_verification.json').open('x') as f:json.dump(result,f,indent=2)
    assert json.loads((CHECK/'readback_verification.json').read_text())==result
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
