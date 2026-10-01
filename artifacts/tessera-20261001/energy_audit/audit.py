"""Audit existing WS8 SRAM counters and energy; do not run new simulations."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent / '20261001_ppa_two_tilings_v1'
sys.path.insert(0, str(ROOT))


def rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, records):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    assert rows(path) == [{k: str(v) for k, v in r.items()} for r in records]


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-1:])
    from neusim.configs.chips.ChipConfig import ChipConfig
    from neusim.npusim.backend.dvfs_power_getter import FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
    from neusim.run_scripts.replay_tessera_arrivals import efficiency
    arches = ('WS', 'WS-independent-8', 'Tessera-8')
    evidence = []
    comparisons = []
    operator_checks = []
    samples = []
    coefficients = []
    inputs = [Path(__file__)]
    for mode in ('native_tail', 'joint_edp'):
        folder = SOURCE / mode
        source = folder / 'arrival_replay/totals.csv'
        check_path = folder / 'arrival_replay/verification.json'
        check = json.loads(check_path.read_text())
        assert check['status'] == 'PASS' and sha(source) == check['artifacts']['totals.csv']
        config_path = folder / 'main_costs/configs.json'
        configs = json.loads(config_path.read_text())
        metrics_path = folder / 'analysis/workload_metrics.csv'
        metrics = {(r['workload_id'], r['architecture']): r for r in rows(metrics_path)}
        lookup = {(r['workload_id'], r['architecture']): r for r in rows(source)}
        ids = list(dict.fromkeys(k[0] for k in metrics))
        inputs += [source, check_path, config_path, metrics_path,
                   folder / 'cost_store/verification.json']
        for arch in arches:
            config = ChipConfig.model_validate(configs[arch + '@2800000000000'])
            coeff = config.dynamic_power_vmem_W / config.vmem_bw_GBps * 1000
            coefficients.append(dict(policy=mode, architecture=arch,
                sram_dynamic_W=config.dynamic_power_vmem_W,
                sram_reference_bytes_per_ns=config.vmem_bw_GBps,
                sram_pj_per_byte=coeff, sram_pj_per_bit=coeff / 8,
                sram_static_W=config.static_power_vmem_W,
                whole_chip_static_W=config.static_power_W,
                compute_pj_per_op=config.tessera_parameters['array_energy_pj_per_op']))
            for wid in ids:
                r = lookup[wid, arch]
                m = metrics[wid, arch]
                chips = int(r['chips'])
                components = {k: float(v) * chips for k, v in r.items()
                              if k.startswith(('dynamic_energy_', 'static_energy_'))}
                total = float(r['system_energy_J'])
                assert math.isclose(math.fsum(components.values()), total, rel_tol=2e-12)
                static = sum(v for k, v in components.items() if k.startswith('static_'))
                evidence.append(dict(policy=mode, workload_id=wid, workload=m['workload'], architecture=arch,
                    seconds=r['seconds'], energy_kJ=total / 1000,
                    sa_dynamic_kJ=components['dynamic_energy_sa_J'] / 1000,
                    sram_dynamic_kJ=components['dynamic_energy_sram_J'] / 1000,
                    hbm_dynamic_kJ=components['dynamic_energy_hbm_J'] / 1000,
                    other_dynamic_kJ=sum(v for k, v in components.items()
                        if k.startswith('dynamic_') and k not in ('dynamic_energy_sa_J', 'dynamic_energy_sram_J', 'dynamic_energy_hbm_J')) / 1000,
                    all_static_kJ=static / 1000,
                    sram_static_kJ=components['static_energy_sram_J'] / 1000,
                    sram_dynamic_pct=100 * components['dynamic_energy_sram_J'] / total,
                    all_static_pct=100 * static / total,
                    sram_TB=float(r['sram_bytes']) * chips / 1e12,
                    hbm_TB=float(r['hbm_bytes']) * chips / 1e12,
                    charged_to_useful_macs=float(r['charged_sa_macs']) / float(r['useful_macs']),
                    efficiency_GOPS_W=m['energy_efficiency_gops_per_w']))
        for wid in ids:
            w, t = (lookup[wid, arch] for arch in ('WS-independent-8', 'Tessera-8'))
            comparisons.append(dict(policy=mode, workload_id=wid, workload=metrics[wid, 'WS']['workload'],
                ws8_over_t8_sram_traffic=float(w['sram_bytes']) / float(t['sram_bytes']),
                ws8_over_t8_sram_dynamic=float(w['dynamic_energy_sram_J']) / float(t['dynamic_energy_sram_J']),
                ws8_over_t8_hbm_traffic=float(w['hbm_bytes']) / float(t['hbm_bytes']),
                ws8_over_t8_total_energy=float(w['system_energy_J']) / float(t['system_energy_J']),
                ws8_speedup_over_t8=float(t['seconds']) / float(w['seconds']),
                tessera_energy_reduction_pct=100 * (1 - float(t['system_energy_J']) / float(w['system_energy_J']))))
        # Inspect every already-profiled independent-WS8 record. Energy is derived
        # directly from saved bytes and the unchanged native coefficient, not from
        # calling the simulator's energy function again.
        config = ChipConfig.model_validate(configs['WS-independent-8@2800000000000'])
        database = sqlite3.connect('file:' + str(folder / 'cost_store/costs.sqlite') + '?mode=ro', uri=True)
        checked = 0
        ratios = []
        for oid, payload in database.execute('select oid,payload from costs where arch=?', ('WS-independent-8',)):
            r = json.loads(payload)
            mapping = json.loads(r['mapping_json'])
            if mapping.get('sram_bandwidth_model') != 'per_pe_double_buffer':
                continue
            traffic = float(r['sram_bytes'])
            active = min(1., float(r['sram_ns']) / float(r['time_ns']))
            eta = next(p.power_efficiency_percent / 100 for p in FIXED_VOLTAGE_REGULATOR_OVERHEAD_TABLE
                       if p.activity_factor >= active)
            expected = config.dynamic_power_vmem_W * traffic / config.vmem_bw_GBps / 1e9 / eta
            assert math.isclose(expected, float(r['dynamic_energy_sram_J']), rel_tol=2e-12, abs_tol=1e-20), (mode, oid)
            assert efficiency(active) == eta
            if traffic:
                ratios.append(float(r['dynamic_energy_sram_J']) / traffic * 1e12)
            checked += 1
        database.close()
        assert checked > 0
        operator_checks.append(dict(policy=mode, checked_ws8_operators=checked,
                                    charged_sram_pj_per_byte_min=min(ratios),
                                    charged_sram_pj_per_byte_max=max(ratios)))
        # Preserve the first recorded GEMM as a concrete mapping example.
        profile = folder / 'main_costs/operator_costs.csv'
        with profile.open() as handle:
            reader = csv.DictReader(handle)
            first = next(reader)
            oid = first['operator_id']
            selected = [first]
            for r in reader:
                if r['operator_id'] != oid:
                    break
                selected.append(r)
        for r in selected:
            if r['architecture'] in arches:
                mapping = json.loads(r['mapping_json'])
                samples.append(dict(policy=mode, operator_id=oid, architecture=r['architecture'],
                    mapping=mapping, seconds=float(r['time_ns']) / 1e9,
                    energy_J=r['energy_J'], sram_bytes=r['sram_bytes'],
                    array_a_read_bytes=r['array_a_read_bytes'], array_b_read_bytes=r['array_b_read_bytes'],
                    partial_write_bytes=r['partial_write_bytes'], reduction_read_bytes=r['reduction_read_bytes'],
                    reduction_write_bytes=r['reduction_write_bytes'], reduction_ops=r['reduction_ops']))
    save(HERE / 'energy_breakdown.csv', evidence)
    save(HERE / 'ws8_vs_tessera8.csv', comparisons)
    save(HERE / 'coefficients.csv', coefficients)
    lines = ['# Independent WS8 energy audit', '',
        'This is a read-only analysis of completed simulations. No simulator code, coefficient, mapping or run result was changed.', '',
        '## Native SRAM accounting', '',
        'Both reads and writes contribute transferred bytes, including padded operand reads, partial writes and reduction reads/writes. '
        'The native model uses one aggregate dynamic coefficient for SRAM transfers; it does not distinguish read from write energy. '
        'The ideal per-PE bandwidth affects service time. Energy retains the native reference bytes/ns coefficient.', '',
        '| Architecture | SRAM pJ/byte | SRAM pJ/bit | SRAM static W | Chip static W | Compute pJ/op |',
        '|---|---:|---:|---:|---:|---:|']
    for r in coefficients[:3]:
        lines.append(f"| {r['architecture']} | {r['sram_pj_per_byte']:.2f} | {r['sram_pj_per_bit']:.2f} | "
                     f"{r['sram_static_W']:.2f} | {r['whole_chip_static_W']:.2f} | {r['compute_pj_per_op']:.3f} |")
    lines += ['', 'These coefficients precede regulator losses. Whole-chip static power is shared across architectures '
              'under the revision model and accumulates over execution and recorded idle time. The array-plus-SRAM area '
              'estimate affects the horizontal plot axis, not energy efficiency.', '',
              '## Llama-2-7B chat: complete E2E energy', '',
              '| Tiling | Architecture | Delay (s) | SRAM traffic (TB) | Compute dynamic (kJ) | SRAM dynamic (kJ) | HBM dynamic (kJ) | All static (kJ) | Total (kJ) |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in evidence:
        if r['workload_id'] == 'llama2_conv':
            lines.append(f"| {r['policy']} | {r['architecture']} | {float(r['seconds']):.2f} | {r['sram_TB']:.2f} | "
                         f"{r['sa_dynamic_kJ']:.2f} | {r['sram_dynamic_kJ']:.2f} | {r['hbm_dynamic_kJ']:.2f} | "
                         f"{r['all_static_kJ']:.2f} | {r['energy_kJ']:.2f} |")
    lines += ['', 'All-static already includes SRAM static energy. Total also includes VU/ICI/other dynamic energy, retained in the CSV.', '',
              '## Independent WS8 versus Tessera-8 for every workload', '',
              '| Tiling | Workload | WS8/T8 SRAM traffic | WS8/T8 total energy | WS8 speedup |',
              '|---|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['policy']} | {r['workload']} | {r['ws8_over_t8_sram_traffic']:.2f}× | "
                     f"{r['ws8_over_t8_total_energy']:.2f}× | {r['ws8_speedup_over_t8']:.2f}× |")
    lines += ['', '## Scope of the conclusion', '',
        'The figures compare total E2E energy efficiency, rather than SRAM efficiency or SRAM power alone. '
        'The native small-K-tile issue identified in the preceding diagnostic remains present. '
        'The normal policy restricts connected-array fragmentation to residual strips/corners, while independent WS '
        'always uses its fixed small arrays. The EDP policy searches array geometries and SRAM tiles jointly.', '',
        'The SRAM coefficient is NeuSim\'s aggregate reference power divided by reference bandwidth. It is not a '
        'per-bank macro characterization for each architecture, and it does not add architecture-specific SRAM '
        'bank/periphery or distribution-network power. The requested ideal SRAM bandwidth remains in force. '
        'This audit establishes consistency with that model, not post-layout physical power accuracy.', '',
        'Source code: `power_model.analyze_dynamic_energy`, `ChipConfig.vmem_bw_GBps`, '
        '`tessera_baselines.counts`, and `replay_tessera_arrivals.Costs.integrate`.', '',
        'All source numbers are read and transformed by this retained script. Per-operator SRAM charges were '
        'independently reconstructed from recorded traffic and regulator efficiency, and output CSVs were read back.', '']
    document = '\n'.join(lines)
    (HERE / 'README.md').write_text(document)
    assert (HERE / 'README.md').read_text() == document
    inputs += [ROOT / 'neusim/npusim/backend/power_model.py', ROOT / 'neusim/configs/chips/ChipConfig.py',
               ROOT / 'neusim/npusim/backend/tessera_baselines.py', ROOT / 'neusim/run_scripts/replay_tessera_arrivals.py']
    result = dict(status='PASS', operator_checks=operator_checks, first_gemm_samples=samples,
                  source_sha256={str(p): sha(p) for p in inputs},
                  artifacts={p.name: sha(p) for p in HERE.iterdir() if p.is_file() and p.name != 'audit.log'})
    with (HERE / 'verification.json').open('x') as handle:
        json.dump(result, handle, indent=2)
    assert json.loads((HERE / 'verification.json').read_text()) == result
    print(json.dumps(operator_checks, indent=2))
    print(document)


if __name__ == '__main__':
    main()
