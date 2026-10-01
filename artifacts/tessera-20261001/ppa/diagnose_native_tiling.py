"""Explain the first source GEMM using retained profiles, without rerunning it."""
import csv
import hashlib
import json
import math
from pathlib import Path

RUN = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = {mode: RUN / mode / 'main_costs/operator_costs.csv'
             for mode in ('native_tail', 'joint_edp')}
    first = {}
    for mode, path in paths.items():
        with path.open() as handle:
            reader = csv.DictReader(handle)
            seed = next(reader)
            oid = seed['operator_id']
            selected = [seed]
            for row in reader:
                if row['operator_id'] != oid or row['kind'] != seed['kind']:
                    break
                selected.append(row)
        assert len(selected) == 13
        first[mode] = {row['architecture']: row for row in selected}
    assert {r['operator_id'] for data in first.values() for r in data.values()} == {oid}
    evidence = []
    for mode, data in first.items():
        for arch in ('WS', 'Planaria-8', 'Tessera-8'):
            row = data[arch]
            mapping = json.loads(row['mapping_json'])
            evidence.append(dict(policy=mode, architecture=arch, kind=row['kind'],
                operator_id=oid, mapping=mapping,
                time_ns=row['time_ns'], energy_J=row['energy_J'],
                static_J=row['static_J'], dynamic_J=row['dynamic_J'],
                hbm_bytes=row['hbm_bytes'], sram_bytes=row['sram_bytes'],
                sa_ns=row['sa_ns'], vu_ns=row['vu_ns'],
                charged_to_useful_macs=int(row['charged_sa_macs']) / int(row['useful_macs'])))
    # Codex: decision start — report the first recorded shape, not a favorable example.
    lines = ['# Native tiling diagnostic', '',
        'This diagnostic reads the first shape in each completed profile. It does not rerun or change mappings. '
        'The two policies use identical energy coefficients. Native factor search minimizes HBM traffic; '
        'its compute estimate does not vary across tile candidates. Among otherwise tied candidates, '
        'the final sort key prefers the smaller SRAM footprint. Consequently it can choose a K tile '
        'smaller than the physical array grain. The revision accounting charges every resulting padded '
        'array invocation and its SRAM accesses. The joint EDP search instead scores those costs explicitly.', '',
        'Source: `neusim/npusim/backend/npusim_lib.py`, `find_best_tile_shape_for_matmul`, '
        '`MXU_cycles = nc_compute_ns * freq_GHz` and the final candidate sort key; '
        '`tessera_partitioned.memory_tile` supplies the revision double-buffer and partial-sum capacity.', '',
        '| Policy | Architecture | SRAM tile M,N,K | Charged/useful MACs | Operator time (ms) | Energy (J) |',
        '|---|---|---|---:|---:|---:|']
    for row in evidence:
        mapping = row['mapping']
        tile = mapping.get('memory_tile')
        if tile is None:
            tile = mapping['phase_plans'][0]['memory_tile']
        lines.append(f"| {row['policy']} | {row['architecture']} | {tile} | "
                     f"{row['charged_to_useful_macs']:.2f} | {float(row['time_ns']) / 1e6:.2f} | "
                     f"{float(row['energy_J']):.2f} |")
    lines += ['', 'These are operator-level examples. Complete E2E and all workload comparisons remain in the two policy directories.', '']
    # Codex: decision end
    target = RUN / 'native_tiling_diagnostic.json'
    result = dict(status='PASS', evidence=evidence,
                  source_sha256={str(p): sha(p) for p in [Path(__file__), *paths.values()]})
    with target.open('x') as handle:
        json.dump(result, handle, indent=2)
    assert json.loads(target.read_text()) == result
    # Independently reread the selected CSV entries and check copied fields/ratios.
    for mode, path in paths.items():
        with path.open() as handle:
            source = {}
            for row in csv.DictReader(handle):
                if row['operator_id'] != oid:
                    break
                source[row['architecture']] = row
        for row in evidence:
            if row['policy'] != mode:
                continue
            original = source[row['architecture']]
            for field in ('time_ns', 'energy_J', 'static_J', 'hbm_bytes', 'sram_bytes'):
                assert row[field] == original[field]
            assert math.isclose(row['charged_to_useful_macs'] * int(original['useful_macs']),
                                int(original['charged_sa_macs']), rel_tol=1e-13)
    doc = RUN / 'native_tiling_diagnostic.md'
    with doc.open('x') as handle:
        handle.write('\n'.join(lines))
    assert doc.read_text() == '\n'.join(lines)
    print(doc.read_text())


if __name__ == '__main__':
    main()
