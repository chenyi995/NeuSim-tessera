"""Check that replay preserves the native execution and power contract."""
from contextlib import redirect_stdout
from io import StringIO
from math import ceil
import unittest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.npusim_lib import compute_node_cost_mxu_time_from_num_ops
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info


def evaluate(shape, config, cycles=None, analyze_energy=True, **updates):
    m, n, k = shape
    op = create_einsum_op([m, k], [k, n], "MK;KN->MN")
    if cycles is not None:
        op.tessera_spec = dict(kind="native_sa_replay", B=1, M=m, N=n, K=k,
                              pe_count=config.num_sa * config.sa_dim ** 2,
                              sa_cycles=cycles, **updates)
    with redirect_stdout(StringIO()):
        return fill_operators_execution_info([op], config, analyze_energy=analyze_energy)[0]


class NativeReplayTests(unittest.TestCase):
    def test_identical_sa_time_preserves_every_native_stat(self):
        for shape in ((1, 128, 128), (1024, 4096, 4096), (17, 259, 61)):
            config = ChipConfig(num_sa=1, freq_GHz=1, vmem_size_MB=12)
            count = 1
            for dim in shape:
                count *= (dim + 127) // 128
            cycles = compute_node_cost_mxu_time_from_num_ops(count, config)
            stock = evaluate(shape, config)
            replay = evaluate(shape, config.model_copy(update={"array_backend": "tessera_native_replay"}), cycles)
            a, b = stock.stats.model_dump(), replay.stats.model_dump()
            a.pop("tessera_details"); b.pop("tessera_details")
            self.assertEqual(a, b)
            self.assertEqual(stock.stats.total_energy_J, replay.stats.total_energy_J)

    def test_native_fallback_memory_and_energy_remain_active(self):
        config = ChipConfig(num_sa=1, freq_GHz=1, vmem_size_MB=12,
                            array_backend="tessera_native_replay")
        op = evaluate((1, 128, 128), config, 100000)
        self.assertEqual(op.stats.sa_time_ns, 0)
        self.assertGreater(op.stats.vu_time_ns, 0)
        before_power = evaluate((1, 128, 128), config, 100000, analyze_energy=False)
        self.assertEqual(before_power.stats.memory_time_ns, config.hbm_latency_ns)
        # Native NONE policy selects 1.7 GHz even when ChipConfig says 1 GHz.
        self.assertEqual(op.stats.memory_time_ns, ceil(config.hbm_latency_ns / op.dvfs_hbm.frequency_GHz))
        self.assertEqual(op.stats.memory_traffic_bytes, 2 * (128 + 128 * 128 + 128))
        self.assertGreater(op.stats.static_energy_J, 0)
        self.assertGreater(op.stats.dynamic_energy_J, 0)
        forced = evaluate((1, 128, 128), config.model_copy(update={"use_vu_for_small_matmul": False}), 100000)
        self.assertEqual(forced.stats.sa_time_ns, ceil(100000 / forced.dvfs_sa.frequency_GHz))
        self.assertEqual(forced.stats.memory_traffic_bytes, op.stats.memory_traffic_bytes)

    def test_bad_replay_is_rejected(self):
        config = ChipConfig(num_sa=1, freq_GHz=1, array_backend="tessera_native_replay")
        for cycles in (0, -1, 1.5, 1):
            with self.assertRaises(ValueError):
                evaluate((128, 128, 128), config, cycles)
        with self.assertRaises(ValueError):
            evaluate((128, 128, 128), config)
        from neusim.npusim.backend.tessera_native import replay_sa_time
        op = evaluate((128, 128, 128), ChipConfig(num_sa=1))
        op.tessera_spec = dict(kind="native_sa_replay", B=1, M=1, N=128, K=128,
                              pe_count=16384, sa_cycles=500)
        with self.assertRaises(ValueError):
            replay_sa_time(op, config, 1, 128, 128, 128)
        op.tessera_spec.update(M=128, pe_count=8192)
        with self.assertRaises(ValueError):
            replay_sa_time(op, config, 1, 128, 128, 128)

    def test_native_fallback_threshold_can_reward_slower_sa_candidate(self):
        # Diagnostic regression: the original native four-times policy is
        # intentionally preserved. A slower SA candidate can select a faster VU.
        shape = (1024, 4096, 4096)
        config = ChipConfig(num_sa=1, freq_GHz=1, vmem_size_MB=12,
                            array_backend="tessera_native_replay")
        vu = evaluate(shape, config, 100000000, analyze_energy=False).stats.vu_time_ns
        at_threshold = evaluate(shape, config, 4 * vu, analyze_energy=False)
        after_threshold = evaluate(shape, config, 4 * vu + 1, analyze_energy=False)
        self.assertGreater(at_threshold.stats.sa_time_ns, 0)
        self.assertEqual(after_threshold.stats.sa_time_ns, 0)
        self.assertLess(after_threshold.stats.execution_time_ns, at_threshold.stats.execution_time_ns)


if __name__ == "__main__":
    unittest.main()
