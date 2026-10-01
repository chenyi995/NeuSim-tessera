#!/usr/bin/env python3
"""Workload A/B/C × QoS S/M/H — Planaria (combo 0 + spatial scheduling) vs combo 6.1 busy-chain fused FCFS.

Scenarios follow the Planaria paper (MICRO 2020 §VI, Table I):
  Workload-A (heavier): RESNET-50, GoogleNet, YOLOv3, SSD-ResNet-34, GNMT
  Workload-B (lighter): MobileNet-v1, SSD-MobileNet-v1, Tiny-YOLO
      (the paper also lists EfficientNet-B0; absent from both the planaria.code
      benchmarks and the G2 suite, so it is omitted)
  Workload-C (mixed):  all 8 networks above
  GNMT = the gnmt_encoder + gnmt_decoder CSVs concatenated into one task.

Three QoS levels: S = 1× MLPerf server QoS, M = 1/4×, H = 1/16× (paper §VI-A).
Baselines (ms): RESNET-50/GoogleNet 15, YOLOv3/SSD-ResNet-34 100
           (= planaria.code scheduler.ini), MobileNet-v1/SSD-MobileNet-v1/
           Tiny-YOLO 10, GNMT 250 (MLPerf v0.5 server).

Metrics (paper §VI-A / Fig 12–15 methodology, with combo 6.1 in place of PREMA):
  * Throughput: max QPS meeting the SLA (bisection over the Poisson mean
    inter-arrival time λ).
    SLA criterion per MLPerf: classification/detection tasks 99%, translation
    (GNMT) 97% completed within QoS
    (aggregated across S_SEARCH task pools).
  * SLA satisfaction rate: at the fixed operating point λ = 0.5 ms
    (multitenant.ini default), the fraction of task pools meeting the above
    criterion (S_RATE pools).
  * Fairness: min_{i,j} PP_i/PP_j (PREMA definition,
    PP_i = (T_iso/T_multi) / (priority_i/Σpriority)), take min per pool
    then average.
  * Energy: mean total energy per pool at the operating point (planaria 6.20 /
    tessera 5.93 pJ/op, same methodology as run_c6_vs_planaria.py).

Both sides share the same WorkloadGenerator task pool (50 tasks, integer-ms
Poisson intervals, uniform type/priority); exec = finish − arrival
(including queueing).

Usage:
  python3 run_abc_qos.py profile [--workers 64]   # fill 8-network × c=1..16 perf tables
  python3 run_abc_qos.py smoke                    # single-pool runtime / isolated-time sanity
  python3 run_abc_qos.py run [--workers 96]       # full 3×3×2 sweep
  python3 run_abc_qos.py plot                     # paper-style 4-panel figure

Outputs (six_models):
  out/"paper result"/128x128/six_models/multitenant_abc_qos.csv        summary
  out/"paper result"/128x128/six_models/multitenant_abc_qos_pools.csv  per-pool detail
  out/"paper result"/128x128/six_models/multitenant_abc_qos.png        figure
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from generator import WorkloadGenerator
from scheduler import scheduler
from profile_nn_tables import (read_gemm_csv, build_ab_table, layer_entry,
                               G2_DIR, OUT_DIR)
from run_c6_vs_planaria import (c61_chain_finishes, combo6_net_cost as _c6_cost_single,
                                E_OP_C6_PJ, E_SRAM_READ_PJ, E_SRAM_WRITE_PJ,
                                C6_D0, D_BIG)

from fissionsa.workload import GEMMLayer
from fissionsa.modes.combo6 import (simulate_layer_t3_connected_grid,
                                    _should_swap, _t3_partition)
from fissionsa.modes import combo6_4, combo6_edp

_NSTRIPS = D_BIG // C6_D0   # 128 / 8 = 16


def edp_chain_finishes(chain, net_tiles):
    """Busy-chain fused-pack with the combo6_edp mapping (FCFS overlap, EDP-opt
    tiling). Same per-task finish accounting as c61_chain_finishes, but the tiles
    are combo6_edp's and packing keeps FCFS order (combo6_edp._fcfs_round_pack_overlap).
    A task, once its tiles enter the chain, is packed to completion (non-preemptive);
    tasks still fuse across the busy chain."""
    layers = []
    for idx, name in chain:
        for M, tiles in net_tiles[name]:
            layers.append((M, [dict(t, task=idx) for t in tiles]))
    rounds = combo6_edp._fcfs_round_pack_overlap(layers, _NSTRIPS, C6_D0, D_BIG)
    prefix = max(t['sub_H'] for (t, _, _) in rounds[0])
    last_round, round_finish = {}, []
    for r_idx, r in enumerate(rounds):
        prefix += max(max(t['M'], t['sub_H']) for (t, _, _) in r)
        round_finish.append(prefix + max(t['sub_W'] for (t, _, _) in r) - 1)
        for t, _, _ in r:
            last_round[t['task']] = r_idx
    return {idx: round_finish[last_round[idx]] for idx, _ in chain}


def chain_finishes(chain, net_tiles):
    """Dispatch: combo6_edp mapping when TESSERA_MODE=edp, else combo 6.1."""
    if _MODE == 'edp':
        return edp_chain_finishes(chain, net_tiles)
    return c61_chain_finishes(chain, net_tiles)

SIX_DIR = REPO / 'out' / 'paper result' / '128x128' / 'six_models'
TABLES_ABC = OUT_DIR / 'nn_tables_abc.json'
# MT_SUFFIX lets a Tessera-d0 variant (e.g. 128×32, TESSERA_D0=32) write to
# its own files instead of clobbering the default d0=8 results.
_SUF = os.environ.get('MT_SUFFIX', '')
# TESSERA_MODE=c64 switches the Tessera layer model + tile front end to
# combo 6.4 (swap-decision routed N-strip, never swaps); default = combo 6.
_MODE = os.environ.get('TESSERA_MODE', '')
OUT_SUMMARY = SIX_DIR / f'multitenant_abc_qos{_SUF}.csv'
OUT_POOLS = SIX_DIR / f'multitenant_abc_qos{_SUF}_pools.csv'
OUT_PNG = SIX_DIR / f'multitenant_abc_qos{_SUF}.png'
OUT_PNG_TP = SIX_DIR / f'multitenant_abc_qos{_SUF}_throughput.png'

FREQ = 700e6          # Hz, planaria cmx-16 convention
NUM_CORES = 16
NUM_TASKS = 50
OP_LAMBDA_MS = 0.5    # fixed operating point (multitenant.ini default)

NET_CSVS = {
    'RESNET-50': ['resnet50'],
    'GoogleNet': ['googlenet'],
    'YOLOv3': ['yolov3'],
    'SSD-ResNet-34': ['ssd_resnet34'],
    'GNMT': ['gnmt_encoder', 'gnmt_decoder'],
    'MobileNet-v1': ['mobilenet_v1'],
    'SSD-MobileNet-v1': ['ssd_mobilenet_v1'],
    'Tiny-YOLO': ['tiny_yolo'],
}

WORKLOADS = {
    'A': ['RESNET-50', 'GoogleNet', 'YOLOv3', 'SSD-ResNet-34', 'GNMT'],
    'B': ['MobileNet-v1', 'SSD-MobileNet-v1', 'Tiny-YOLO'],
    'C': list(NET_CSVS.keys()),
}

QOS_BASE_MS = {
    'RESNET-50': 15.0, 'GoogleNet': 15.0,
    'YOLOv3': 100.0, 'SSD-ResNet-34': 100.0,
    'MobileNet-v1': 10.0, 'SSD-MobileNet-v1': 10.0, 'Tiny-YOLO': 10.0,
    'GNMT': 250.0,
}
QOS_LEVELS = {'S': 1.0, 'M': 0.25, 'H': 1.0 / 16}

SLA_TARGET = {name: (0.97 if name == 'GNMT' else 0.99) for name in NET_CSVS}

# c61edf = combo 6.1 + EDF: still whole-array serial + busy-chain fusion, but on
# each arrival the unfinished tasks on the chain are reordered by absolute
# deadline (arrival + QoS) before repacking; the already-finished prefix is
# frozen (non-preemptive). Energy is identical to c61 (words are independent
# of packing order).
SYSTEMS = ['planaria', 'c61', 'c61edf']
S_SEARCH = 12         # number of pools per λ point in the throughput bisection
S_RATE = 24           # number of pools for operating-point SLA rate / fairness / energy
BISECT_ROUNDS = 9
LAM_LO_MS, LAM_HI_MS = 0.02, 50.0   # λ search range (mean inter-arrival time)


def net_layers(name):
    """Concatenate (lid, M, N, K) from all CSVs of this task type, renumbering layer ids."""
    rows = []
    for csv_name in NET_CSVS[name]:
        for _, M, N, K in read_gemm_csv(G2_DIR / f'{csv_name}.csv'):
            rows.append((len(rows), M, N, K))
    return rows


# ---------------------------------------------------------------- profile

def cmd_profile(workers):
    shapes, seen = [], set()
    for name in NET_CSVS:
        for _, M, N, K in net_layers(name):
            for c in range(1, NUM_CORES + 1):
                if (c, M, N, K) not in seen:
                    seen.add((c, M, N, K))
                    shapes.append((c, M, N, K))
    ab = build_ab_table(shapes, workers)

    tables = {}
    for c in range(NUM_CORES, 0, -1):
        tables[f'{c} Cores'] = {}
        for name in NET_CSVS:
            entries = []
            for lid, M, N, K in net_layers(name):
                a, b, nt = ab[f'{c},{M},{N},{K}']
                tiles, cyc_per_tile, e_uj = layer_entry(M, N, K, a, b, nt)
                entries.append([f'L{lid}', tiles, cyc_per_tile, e_uj])
            tables[f'{c} Cores'][name] = entries
    json.dump(tables, TABLES_ABC.open('w'), indent=0)
    print(f'wrote {TABLES_ABC}')


# ---------------------------------------------------------------- worker side

_W = {}   # per-process heavy state


def _worker_init():
    raw = json.load(TABLES_ABC.open())
    nn_info = {ck: {n: [tuple(e) for e in ents] for n, ents in nets.items()}
               for ck, nets in raw.items()}

    net_tiles, net_cost, iso = {}, {}, {'planaria': {}, 'c61': {}}
    for name in NET_CSVS:
        tiles_list, cycles, energy_pj = [], 0, 0.0
        sim = (combo6_4.simulate_layer_t3_connected_grid if _MODE == 'c64'
               else simulate_layer_t3_connected_grid)
        for lid, M, N, K in net_layers(name):
            if _MODE == 'edp':
                # combo6_edp mapping: per-layer EDP-optimal (kt,nt) @ HBM4
                kt, nt = combo6_edp.best_tiling(M, N, K, 1400, C6_D0)
                comp, inw, outw = combo6_edp._cycles_words(M, N, K, kt, nt, C6_D0)
                cycles += comp
                energy_pj += (2 * M * N * K * E_OP_C6_PJ
                              + inw / 8 * E_SRAM_READ_PJ + outw / 8 * E_SRAM_WRITE_PJ)
                tiles_list.append((M, combo6_edp.layer_tiles(M, N, K, 1400, C6_D0)))
                continue
            r = sim(
                GEMMLayer(layer_id=lid, m=M, n=N, k=K), d=32, m=4, d0=C6_D0, dram_bw=0)
            cycles += r.compute_cycles
            energy_pj += (2 * M * N * K * E_OP_C6_PJ
                          + r.input_words / 8 * E_SRAM_READ_PJ
                          + r.output_words / 8 * E_SRAM_WRITE_PJ)
            if _MODE == 'c64':
                tiles_list.append((M, combo6_4.layer_tiles(M, N, K, D_BIG, C6_D0)))
            else:
                if _should_swap(M, N, K, D_BIG, C6_D0):
                    M, N, K = N, M, K
                A, B, C, Dc = _t3_partition(M, N, K, D_BIG, C6_D0)
                tiles_list.append((M, list(A) + list(B) + list(C) + list(Dc)))
        net_tiles[name] = tiles_list
        net_cost[name] = (cycles, energy_pj / 1e6)
        iso['planaria'][name] = sum(e[1] * e[2] for e in
                                    nn_info[f'{NUM_CORES} Cores'][name])
        iso['c61'][name] = chain_finishes([(0, name)], net_tiles)[0]

    _W.update(nn_info=nn_info, net_tiles=net_tiles, net_cost=net_cost, iso=iso)


def _pool_metrics(records, iso, qos_dict):
    """records = [(name, prio, exec_ms, energy_uj)] -> pool-level metrics."""
    per_net_ok, per_net_n = {}, {}
    prios = [r[1] for r in records]
    sum_prio = float(sum(prios))
    pp = []
    energy = 0.0
    for name, prio, exec_ms, e_uj in records:
        per_net_n[name] = per_net_n.get(name, 0) + 1
        ok = exec_ms <= qos_dict[name]
        per_net_ok[name] = per_net_ok.get(name, 0) + ok
        iso_ms = iso[name] / FREQ * 1000
        pp.append((iso_ms / exec_ms) / (prio / sum_prio) if exec_ms > 0 else 1.0)
        energy += e_uj

    cd_ok = sum(v for n, v in per_net_ok.items() if n != 'GNMT')
    cd_n = sum(v for n, v in per_net_n.items() if n != 'GNMT')
    tr_ok = per_net_ok.get('GNMT', 0)
    tr_n = per_net_n.get('GNMT', 0)
    sla_met = ((cd_n == 0 or cd_ok / cd_n >= 0.99)
               and (tr_n == 0 or tr_ok / tr_n >= 0.97))
    fairness = min(pp) / max(pp) if pp else 1.0
    return dict(sla_met=sla_met, fairness=fairness, energy_uj=energy,
                cd_ok=cd_ok, cd_n=cd_n, tr_ok=tr_ok, tr_n=tr_n)


def run_pool(job):
    """job = (wk, qk, system, lam_ms, seed) -> pool-level metrics dict."""
    wk, qk, system, lam_ms, seed = job
    if not _W:
        _worker_init()
    np.random.seed(seed)
    random.seed(seed)

    gen = WorkloadGenerator(distribution='poisson',
                            args={'max_cycles': 0, 'arrival_time': lam_ms,
                                  'frequency': FREQ, 'N': NUM_TASKS})
    for name in WORKLOADS[wk]:
        gen.add_task_type(name)
    workload = gen.generate()   # [(arrival_cycle, type, priority)]
    qos_dict = {n: QOS_BASE_MS[n] * QOS_LEVELS[qk] for n in WORKLOADS[wk]}

    ms = lambda cyc: cyc / FREQ * 1000
    records = []
    if system == 'planaria':
        done = scheduler(workload, _W['nn_info'], qos_dict, FREQ, NUM_CORES,
                         verbose=False)
        keys = [t + str(i) for i, (_, t, _) in enumerate(workload)]
        for task in done:
            idx = keys.index(task.queue_key)
            records.append((task.task_name, workload[idx][2],
                            ms(task.current_time - task.start_time), task.energy))
    else:
        edf = system == 'c61edf'
        net_tiles, net_cost = _W['net_tiles'], _W['net_cost']
        deadline = {i: w[0] + qos_dict[w[1]] * 1e-3 * FREQ
                    for i, w in enumerate(workload)}
        c61, chain, chain_start, chain_rel = {}, [], 0, {}
        for idx, (arrival, name, _prio) in enumerate(workload):
            if chain and arrival <= chain_start + max(chain_rel.values()):
                if edf:
                    frozen = [t for t in chain
                              if chain_start + chain_rel[t[0]] <= arrival]
                    pending = [t for t in chain
                               if chain_start + chain_rel[t[0]] > arrival]
                    pending.append((idx, name))
                    pending.sort(key=lambda t: (deadline[t[0]], t[0]))
                    chain = frozen + pending
                else:
                    chain.append((idx, name))
            else:
                for i2, f in chain_rel.items():
                    c61[i2] = chain_start + f
                chain = [(idx, name)]
                chain_start = arrival
            chain_rel = chain_finishes(chain, net_tiles)
        for i2, f in chain_rel.items():
            c61[i2] = chain_start + f
        for idx, (arrival, name, prio) in enumerate(workload):
            records.append((name, prio, ms(c61[idx] - arrival),
                            net_cost[name][1]))

    iso_key = 'planaria' if system == 'planaria' else 'c61'
    out = _pool_metrics(records, _W['iso'][iso_key], qos_dict)
    out.update(wk=wk, qk=qk, system=system, lam_ms=lam_ms, seed=seed)
    return out


# ---------------------------------------------------------------- run

def _agg_feasible(results):
    """Throughput criterion: aggregated across pools, classification/detection ≥99% and translation ≥97%."""
    cd_ok = sum(r['cd_ok'] for r in results)
    cd_n = sum(r['cd_n'] for r in results)
    tr_ok = sum(r['tr_ok'] for r in results)
    tr_n = sum(r['tr_n'] for r in results)
    return ((cd_n == 0 or cd_ok / cd_n >= 0.99)
            and (tr_n == 0 or tr_ok / tr_n >= 0.97))


def cmd_run(workers, run_systems):
    SIX_DIR.mkdir(parents=True, exist_ok=True)
    configs = [(wk, qk, s) for wk in WORKLOADS for qk in QOS_LEVELS
               for s in run_systems]
    pool_rows = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as ex:

        def batch(jobs):
            return list(ex.map(run_pool, jobs))

        # ---- phase T: throughput bisection
        print(f'[T] feasibility at λ={LAM_HI_MS} ms', flush=True)
        state = {}   # cfg -> dict(lo, hi) / final qps value
        jobs = [(wk, qk, s, LAM_HI_MS, 5000 + i)
                for (wk, qk, s) in configs for i in range(S_SEARCH)]
        res = batch(jobs)
        pool_rows += res
        for cfg in configs:
            sub = [r for r in res if (r['wk'], r['qk'], r['system']) == cfg]
            if _agg_feasible(sub):
                state[cfg] = dict(lo=LAM_LO_MS, hi=LAM_HI_MS)
            else:
                state[cfg] = dict(qps=0.0)   # paper convention: not attainable
                print(f'  {cfg}: SLA not attainable even at '
                      f'{1000/LAM_HI_MS:.0f} QPS', flush=True)

        for rnd in range(BISECT_ROUNDS):
            live = [c for c in configs if 'qps' not in state[c]]
            if not live:
                break
            jobs = []
            for cfg in live:
                mid = math.sqrt(state[cfg]['lo'] * state[cfg]['hi'])
                state[cfg]['mid'] = mid
                jobs += [cfg + (mid, 6000 + rnd * 100 + i) for i in range(S_SEARCH)]
            res = batch(jobs)
            pool_rows += res
            for cfg in live:
                sub = [r for r in res if (r['wk'], r['qk'], r['system']) == cfg]
                if _agg_feasible(sub):
                    state[cfg]['hi'] = state[cfg]['mid']
                else:
                    state[cfg]['lo'] = state[cfg]['mid']
            print(f'[T] round {rnd + 1}/{BISECT_ROUNDS} done '
                  f'({time.time() - t0:.0f}s)', flush=True)

        for cfg in configs:
            if 'qps' not in state[cfg]:
                state[cfg]['qps'] = 1000.0 / state[cfg]['hi']

        # ---- phase R: operating point λ = 0.5 ms
        print(f'[R] operating point λ={OP_LAMBDA_MS} ms', flush=True)
        jobs = [(wk, qk, s, OP_LAMBDA_MS, 9000 + i)
                for (wk, qk, s) in configs for i in range(S_RATE)]
        res = batch(jobs)
        pool_rows += res

    # ---- summary (systems not run are merged from the existing CSV)
    old = {}
    if OUT_SUMMARY.exists():
        for r in csv.DictReader(OUT_SUMMARY.open()):
            old[(r['workload'], r['qos'])] = r
    rows = []
    for wk in WORKLOADS:
        for qk in QOS_LEVELS:
            row = dict(workload=wk, qos=qk)
            for s in SYSTEMS:
                if s in run_systems:
                    sub = [r for r in pool_rows
                           if (r['wk'], r['qk'], r['system']) == (wk, qk, s)
                           and r['lam_ms'] == OP_LAMBDA_MS]
                    row[f'{s}_qps'] = state[(wk, qk, s)]['qps']
                    row[f'{s}_sla_rate'] = sum(r['sla_met'] for r in sub) / len(sub)
                    row[f'{s}_fairness'] = sum(r['fairness'] for r in sub) / len(sub)
                    row[f'{s}_energy_uj'] = sum(r['energy_uj'] for r in sub) / len(sub)
                else:
                    # System not run: merge only if the existing CSV already has its data; otherwise skip the whole column
                    o = old.get((wk, qk))
                    if o is None or f'{s}_qps' not in o:
                        continue
                    for col in ('qps', 'sla_rate', 'fairness', 'energy_uj'):
                        row[f'{s}_{col}'] = float(o[f'{s}_{col}'])
            p = row['planaria_qps']
            for s in SYSTEMS[1:]:
                if f'{s}_qps' not in row:
                    continue
                row[f'qps_ratio_{s}_over_planaria'] = \
                    (row[f'{s}_qps'] / p) if p > 0 else float('inf')
            rows.append(row)

    cols = list(rows[0].keys())
    with OUT_SUMMARY.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in row.items()})
    print(f'wrote {OUT_SUMMARY}')

    pool_cols = ['wk', 'qk', 'system', 'lam_ms', 'seed', 'sla_met', 'fairness',
                 'energy_uj', 'cd_ok', 'cd_n', 'tr_ok', 'tr_n']
    new_file = not OUT_POOLS.exists()
    with OUT_POOLS.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=pool_cols, extrasaction='ignore')
        if new_file:
            w.writeheader()
        for r in sorted(pool_rows, key=lambda r: (r['wk'], r['qk'],
                                                  r['system'], r['lam_ms'])):
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v)
                        for k, v in r.items() if k in pool_cols})
    print(f'wrote {OUT_POOLS} ({len(pool_rows)} pool runs, '
          f'{time.time() - t0:.0f}s total)')

    for row in rows:
        print(row)


# ---------------------------------------------------------------- smoke

def cmd_smoke():
    _worker_init()
    print('isolated exec (ms):')
    for name in NET_CSVS:
        p = _W['iso']['planaria'][name] / FREQ * 1000
        c = _W['iso']['c61'][name] / FREQ * 1000
        print(f'  {name:18s} planaria {p:8.3f}  c61 {c:8.3f}  '
              f'QoS-H {QOS_BASE_MS[name] / 16:.3f}')
    for system in SYSTEMS:
        for lam in (0.5, 0.1):
            t0 = time.time()
            r = run_pool(('C', 'H', system, lam, 5000))
            print(f'C/H {system:8s} λ={lam}: {time.time() - t0:6.1f}s  '
                  f'sla_met={r["sla_met"]} cd {r["cd_ok"]}/{r["cd_n"]} '
                  f'tr {r["tr_ok"]}/{r["tr_n"]} fairness={r["fairness"]:.3f}')


# ---------------------------------------------------------------- plot

def cmd_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(OUT_SUMMARY.open()))
    order = [(wk, qk) for wk in 'ABC' for qk in 'SMH']
    rows = {(r['workload'], r['qos']): r for r in rows}
    x = np.arange(len(order))
    labels = [f'QoS-{qk}' for _, qk in order]
    c_p, c_t, c_e = '#a8cbe4', '#1f4e79', '#e8a33d'   # planaria / c6.1 / c6.1+EDF
    LBL = {'planaria': 'Planaria (combo 0)', 'c61': 'Tessera (combo 6.1 FCFS)',
           'c61edf': 'Tessera (combo 6.1 EDF)'}

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))

    def group_decor(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        for i, wk in enumerate('ABC'):
            ax.text(1 + 3 * i, -0.32, f'Workload-{wk}',
                    transform=ax.get_xaxis_transform(),
                    ha='center', fontsize=8)
            if i:
                ax.axvline(3 * i - 0.5, color='gray', lw=0.6, ls=':')

    ax = axes[0][0]
    v1 = [float(rows[o]['qps_ratio_c61_over_planaria']) for o in order]
    v2 = [float(rows[o]['qps_ratio_c61edf_over_planaria']) for o in order]
    qps_floor = 0.98 * 1000.0 / LAM_LO_MS   # planaria hit the search floor -> ratio is an upper bound
    capped = [float(rows[o]['planaria_qps']) >= qps_floor for o in order]
    ax.bar(x - 0.17, v1, 0.34, color=c_t, label=LBL['c61'])
    ax.bar(x + 0.17, v2, 0.34, color=c_e, label=LBL['c61edf'])
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_ylabel('Throughput / Planaria')
    ax.set_title('Throughput (max QPS meeting SLA) vs Planaria')

    def qps_mark(o, v, sysname):   # '≤' applies only when this system is not capped but planaria is
        sys_cap = float(rows[o][f'{sysname}_qps']) >= qps_floor
        p_cap = float(rows[o]['planaria_qps']) >= qps_floor
        return ('*' if sys_cap else '≤') if p_cap else ''

    for xi, o, va, vb in zip(x, order, v1, v2):
        ax.text(xi - 0.17, va, qps_mark(o, va, 'c61') + f'{va:.2f}',
                ha='center', va='bottom', fontsize=6)
        ax.text(xi + 0.17, vb, qps_mark(o, vb, 'c61edf') + f'{vb:.2f}',
                ha='center', va='bottom', fontsize=6)
    ax.set_ylim(0, max(v1 + v2) * 1.35)
    if any(capped):
        ax.text(0.02, 0.97, f'≤: Planaria hit λ search floor '
                f'({1000 / LAM_LO_MS / 1000:.0f}k QPS), ratio is an upper bound\n'
                f'*: both hit the floor — tie within tested range',
                transform=ax.transAxes, fontsize=6.5, va='top')
    ax.legend(fontsize=7)
    group_decor(ax)

    def three_bars(ax, col, scale=1.0):
        for s, c, dx in (('planaria', c_p, -0.28), ('c61', c_t, 0.0),
                         ('c61edf', c_e, 0.28)):
            ax.bar(x + dx, [scale * float(rows[o][f'{s}_{col}']) for o in order],
                   0.26, color=c, label=LBL[s])
        ax.legend(fontsize=7)

    ax = axes[0][1]
    three_bars(ax, 'sla_rate', 100.0)
    ax.set_ylabel('SLA satisfaction rate (%)')
    ax.set_title(f'SLA satisfaction rate @ λ={OP_LAMBDA_MS} ms')
    ax.set_ylim(0, 118)
    group_decor(ax)

    ax = axes[1][0]
    three_bars(ax, 'fairness')
    ax.set_ylabel('Fairness (min PPi/PPj)')
    ax.set_title(f'Fairness @ λ={OP_LAMBDA_MS} ms')
    group_decor(ax)

    ax = axes[1][1]
    vals = [float(rows[o]['planaria_energy_uj'])
            / float(rows[o]['c61_energy_uj']) for o in order]
    ax.bar(x, vals, 0.6, color=c_t)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_ylabel('Energy reduction / Planaria')
    ax.set_title('Energy reduction, combo 6.1 vs Planaria (EDF identical)')
    group_decor(ax)

    fig.suptitle('Workload A/B/C × QoS S/M/H — Planaria (combo 0 + spatial scheduler) '
                 'vs Tessera (combo 6.1 FCFS / EDF fused packing)', fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(OUT_PNG, dpi=180)
    print(f'wrote {OUT_PNG}')

    # Standalone throughput figure: Tessera = combo 6.1 EDF, single series of ratio bars
    fig2, ax = plt.subplots(figsize=(6.2, 3.6))
    vals = [float(rows[o]['qps_ratio_c61edf_over_planaria']) for o in order]
    ax.bar(x, vals, 0.6, color=c_t)
    ax.axhline(1.0, color='k', lw=0.8, ls='--')
    ax.set_ylabel('Throughput / Planaria')
    ax.set_title('Throughput (max QPS meeting SLA), Tessera vs Planaria')
    for xi, v in zip(x, vals):
        ax.text(xi, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_ylim(0, max(vals) * 1.18)
    group_decor(ax)
    fig2.tight_layout()
    fig2.savefig(OUT_PNG_TP, dpi=180)
    print(f'wrote {OUT_PNG_TP}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['profile', 'smoke', 'run', 'plot'])
    p.add_argument('--workers', type=int, default=64)
    p.add_argument('--systems', default=','.join(SYSTEMS),
                   help='comma-separated; run only these systems, merge the rest from the existing CSV')
    a = p.parse_args()
    if a.cmd == 'profile':
        cmd_profile(a.workers)
    elif a.cmd == 'smoke':
        cmd_smoke()
    elif a.cmd == 'run':
        cmd_run(a.workers, a.systems.split(','))
    else:
        cmd_plot()
