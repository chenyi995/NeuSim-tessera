"""Independent progress/resource goldens for the repaired QoS execution model."""
import json
import math
import unittest
from collections import Counter

from neusim.run_scripts.run_feature_qos_scheduler import (
    MatchedCosts, MatchedTask, run_matched, POLICY, FREQ,
)


# Codex: decision start — validate the requested QoS repair with exact work ledgers.
class MatchedQoSTests(unittest.TestCase):
    def fixture(self, arch, shape=(2, 9, 17, 11), capacity=1 << 20, resident=False):
        """Synthetic validation inputs; never reported as workload performance."""
        grain = 32 if arch == 'Planaria-32' else 8
        units = (64 // grain) ** 2
        profiles = {}
        for c in range(1, units + 1):
            high = c >= units // 2
            g = (grain, grain, False, 1, grain, grain)
            mt, nt, kt = (4, 6, 11) if high else (2, 3, 5)
            row = dict(kind='matrix', key='x'.join(map(str, shape)),
                       mapping_json=json.dumps(dict(geometry=g, memory_tile=(mt, nt, kt))),
                       hbm_bytes=0 if resident else 1, vu_ns=1, reduction_ops=1024)
            profiles[c] = [row]
        chip = dict(side=64, grain=grain, capacity_bytes=capacity, frequency_Hz=FREQ,
                    hbm_bytes_per_cycle=16, hbm_latency_cycles=0)
        return MatchedCosts({'test': profiles}, chip, arch)

    def test_unfinished_work_is_covered_once_after_remapping(self):
        for arch in ('Planaria-32', 'Tessera-8'):
            model = self.fixture(arch)
            for seed in (7, 19, 41):
                trace = []
                def policy(queue, possible, total):
                    return {key: 1 if t.groups == 0 else total for key, t in queue.items()}
                result, _, _ = run_matched([(0, 'test', 1)], model, {'test': 100}, seed,
                                          asynchronous=arch == 'Tessera-8', trace=trace,
                                          allocation_policy=policy)
                tiles = [x for x in trace if x['event'] == 'tile_start']
                ledger = Counter()
                hbm = 0
                for t in tiles:
                    for m in range(t['m'], t['m'] + t['h']):
                        for n in range(t['n'], t['n'] + t['w']):
                            for k in range(t['k0'], t['k0'] + t['k']):
                                ledger[t['b'], m, n, k] += 1
                    hbm += 2 * (t['h'] * t['k'] + t['k'] * t['w'])
                    if t['k0'] + t['k'] == 11:
                        hbm += 2 * t['h'] * t['w']
                expected = {(b, m, n, k) for b in range(2) for m in range(9)
                            for n in range(17) for k in range(11)}
                self.assertEqual(set(ledger), expected)
                self.assertEqual(set(ledger.values()), {1})
                self.assertEqual(result[0]['useful_macs'], len(expected))
                self.assertEqual(sum(x['bytes'] for x in trace if x['event'] == 'hbm_service'), hbm)
                # A live C tile retains its old K mapping; the next C tile reselects.
                self.assertEqual([t['k'] for t in tiles[:3]], [5, 5, 1])
                self.assertEqual([t['memory_tile'] for t in tiles[:3]], [[2, 3, 5]] * 3)
                self.assertEqual(tiles[3]['memory_tile'], [4, 6, 11])

    def test_estimated_remaining_time_matches_fixed_share_execution(self):
        for arch in ('Planaria-32', 'Tessera-8'):
            for resident in (False, True):
                model = self.fixture(arch, resident=resident)
                for share in (1, model.units // 2, model.units):
                    for seed in (7, 19):
                        task = MatchedTask(0, 'test', 0, 0, 1, 100, model)
                        expected = task.get_remaining_estimated_time(share)
                        result, _, _ = run_matched([(0, 'test', 1)], model, {'test': 100}, seed,
                                                  allocation_policy=lambda q, p, n: {key: share for key in q})
                        self.assertTrue(math.isclose(result[0]['finish_cycle'], expected, rel_tol=2e-12),
                                        (arch, share, resident, result[0]['finish_cycle'], expected))

    def test_profile_switch_changes_live_bandwidth_and_estimator(self):
        model = self.fixture('Tessera-8')
        t = MatchedTask(0, 'test', 0, 0, 1, 100, model)
        pinned = model.profiles['test'][1][0]
        t.rectangles = []
        t.partial = (pinned, 0, 0, 0, 2, 3, 5)
        t.inflight = 100
        # Independent two remaining transfers, with the final C written once.
        expected_hbm = [2 * (2 * 5 + 5 * 3), 2 * (2 * 1 + 1 * 3 + 2 * 3)]
        for c in (1, 8, 64):
            expected = 100 + sum(max(2 + 2 * 8 + 8 - 1,
                                    hb / 16 * 64 / c, 2 * 3 / 1024)
                                 for hb in expected_hbm)
            self.assertAlmostEqual(t.get_remaining_estimated_time(c), expected)
        for now in (0, 17, 99):
            t.current_time = now
            self.assertEqual(t.possible(), POLICY.get_possible_num_cores(t, FREQ, model.units))

    def test_shared_servers_and_live_sram_both_architectures(self):
        for arch in ('Planaria-32', 'Tessera-8'):
            model = self.fixture(arch, capacity=1000)
            trace = []
            rr, _, _ = run_matched([(0, 'test', 1), (1, 'test', 1), (3, 'test', 2)],
                                  model, {'test': 100}, 23, asynchronous=arch == 'Tessera-8', trace=trace)
            self.assertEqual(len(rr), 3)
            for server in ('hbm_service', 'vu_service'):
                services = [r for r in trace if r['event'] == server]
                for a, b in zip(services, services[1:]):
                    self.assertLessEqual(a['finish_cycle'], b['start_cycle'])
            self.assertTrue(all(r['held_bytes'] <= 1000 for r in trace if r['event'] == 'group_start'))

    def test_planaria_long_composition_remains_legal(self):
        model = self.fixture('Planaria-32', shape=(1, 10, 128, 32), resident=True)
        from dataclasses import replace
        original = model.profiles['test'][model.units][0]
        long = replace(original, geometry=(32, 128, True, 1, 32, 128), tile=(10, 128, 32))
        for c in model.profiles['test']:model.profiles['test'][c] = (long,)
        trace = []
        result, _, _ = run_matched([(0, 'test', 1)], model, {'test': 100}, 7, trace=trace,
                                  allocation_policy=lambda q, p, n: {key: n for key in q})
        self.assertEqual(result[0]['finish_cycle'], 10 + 32 + 32 + 128 - 2)
        self.assertEqual(next(r['geometry'][-2:] for r in trace if r['event'] == 'tile_start'), [32, 128])

    def test_idle_gaps_and_layer_dependencies(self):
        for arch in ('Planaria-32', 'Tessera-8'):
            model = self.fixture(arch, shape=(1, 2, 3, 5), resident=True)
            for c, pp in model.profiles['test'].items():
                model.profiles['test'][c] = pp * 2
                d = model.suffix['test'][c][0]
                model.suffix['test'][c] = [2 * d, d, 0.]
            trace = []
            rr, _, _ = run_matched([(100, 'test', 1), (10000, 'test', 1)], model, {'test': 100}, 31,
                                  asynchronous=arch == 'Tessera-8', trace=trace)
            starts = [x for x in trace if x['event'] == 'group_start']
            self.assertEqual(starts[0]['cycle'], 100)
            self.assertEqual(starts[2]['cycle'], 10000)
            self.assertEqual(starts[1]['cycle'], starts[0]['finish_cycle'])
            self.assertEqual(starts[3]['cycle'], starts[2]['finish_cycle'])
            self.assertEqual(rr[0]['latency_ms'], rr[1]['latency_ms'])
# Codex: decision end


if __name__ == '__main__':
    unittest.main()
