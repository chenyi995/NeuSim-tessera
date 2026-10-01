"""Locate existing GEMM mappings responsible for the normal-tiling SRAM gap."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '20261001_ppa_two_tilings_v1'


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    folder = SOURCE / 'native_tail'
    totals = rows(folder / 'main_costs/totals.csv')
    target = next(r for r in totals if r['workload_id'] == 'llama2_conv')
    weights = [r for r in rows(folder / 'main_costs/invocation_weights.csv')
               if (r['dataset'], r['cohort']) == (target['dataset'], target['cohort'])]
    dbs = {mode: sqlite3.connect('file:' + str(SOURCE / mode / 'cost_store/costs.sqlite') + '?mode=ro', uri=True)
           for mode in ('native_tail', 'joint_edp')}
    summed = {arch: Counter() for arch in ('WS-independent-8', 'Tessera-8')}
    details = []
    for w in weights:
        records = {}
        count = int(w['repeat'])
        for arch in summed:
            r = json.loads(dbs['native_tail'].execute(
                'select payload from costs where arch=? and kind=? and oid=?',
                (arch, w['kind'], int(w['operator_id']))).fetchone()[0])
            records[arch] = r
            for field in ('sram_bytes', 'charged_sa_macs', 'useful_macs', 'hbm_bytes'):
                summed[arch][field] += count * int(r[field])
        w8, t8 = (records[arch] for arch in summed)
        delta = count * (int(t8['sram_bytes']) - int(w8['sram_bytes']))
        if delta > 0:
            entry = dict(operator_id=w['operator_id'], kind=w['kind'], invocations=count,
                         tessera_extra_sram_bytes=delta,
                         tessera_extra_charged_macs=count * (int(t8['charged_sa_macs']) - int(w8['charged_sa_macs'])),
                         mappings={})
            for mode, db in dbs.items():
                for arch in summed:
                    r = records[arch] if mode == 'native_tail' else json.loads(db.execute(
                        'select payload from costs where arch=? and kind=? and oid=?',
                        (arch, w['kind'], int(w['operator_id']))).fetchone()[0])
                    entry['mappings'][mode + '/' + arch] = dict(
                        mapping=json.loads(r['mapping_json']),
                        sram_bytes=int(r['sram_bytes']), charged_sa_macs=int(r['charged_sa_macs']),
                        useful_macs=int(r['useful_macs']),
                        charged_to_useful_macs=int(r['charged_sa_macs']) / int(r['useful_macs']) if int(r['useful_macs']) else 0)
            details.append(entry)
    for arch, values in summed.items():
        original = next(r for r in totals if r['workload_id'] == 'llama2_conv' and r['architecture'] == arch)
        for field, value in values.items():
            assert value == int(original[field]), (arch, field)
    for db in dbs.values():
        db.close()
    details.sort(key=lambda r: r['tessera_extra_sram_bytes'], reverse=True)
    gap = summed['Tessera-8']['sram_bytes'] - summed['WS-independent-8']['sram_bytes']
    top = details[:5]
    top_fraction = sum(r['tessera_extra_sram_bytes'] for r in top) / gap
    result = dict(status='PASS', workload_id='llama2_conv', checked_weighted_totals=summed,
                  net_sram_gap_bytes=gap, top_five_gap_fraction=top_fraction,
                  largest_contributors=top)
    out = HERE / 'tiling_contributors.json'
    with out.open('x') as handle:
        json.dump(result, handle, indent=2)
    assert json.loads(out.read_text()) == result
    lines = ['# Normal-tiling SRAM contributors: Llama-2-7B chat', '',
             'Source: completed per-operator profiles and invocation weights. Weighted SRAM, HBM, useful MAC and '
             'charged MAC totals were reconstructed independently and matched the recorded workload totals exactly.', '',
             f'The five largest positive operator contributions account for {100 * top_fraction:.2f}% of the net Tessera-8 minus independent-WS8 SRAM gap.', '',
             '| Operator | Policy / architecture | M,N,K | SRAM tile M,N,K | Charged/useful MACs |',
             '|---|---|---|---|---:|']
    for entry in top:
        for key, r in entry['mappings'].items():
            mapping = r['mapping']
            phase = mapping.get('phase_plans', [mapping])[0]
            shape = [mapping.get(k, phase.get(k)) for k in ('M', 'N', 'K')]
            tile = mapping.get('memory_tile', phase.get('memory_tile'))
            lines.append(f"| {entry['operator_id']} | {key} | {shape} | {tile} | {r['charged_to_useful_macs']:.2f} |")
    lines += ['', 'The native rule ranks HBM traffic and uses SRAM footprint as a final tie-breaker. '
              'The two architectures reserve different partial-sum workspaces, so their capacity-feasible native '
              'tiles differ. This can make the connected design select a much smaller K tile and pay more padding '
              'than the fixed small arrays. That is a mapping consequence, not a reduction in the disconnected '
              'arrays\' intrinsic communication requirements.', '']
    document = '\n'.join(lines)
    (HERE / 'tiling_contributors.md').write_text(document)
    assert (HERE / 'tiling_contributors.md').read_text() == document
    print(document)


if __name__ == '__main__':
    main()
