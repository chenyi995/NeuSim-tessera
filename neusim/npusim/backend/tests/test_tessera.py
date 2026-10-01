"""Independent timing, byte-count, graph and energy conservation checks."""
from dataclasses import replace
import json
from pathlib import Path
import unittest

from neusim.npusim.backend.tessera import (TesseraConfig, array_cost, bank_cycles,
                                         gemm_cost, plan_memory, VARIANTS)


ROOT = Path(__file__).resolve().parents[4]


def load_config():
    return TesseraConfig(**json.loads((ROOT / "configs/chips/tessera_revision.json").read_text())["parameters"])


def golden_bank_events(m, d, folds, skew):
    port_free = 0
    bank_free = [0, 0]
    input_free = 0
    events = []
    for f in range(folds):
        bank = f % 2
        start = max(port_free, bank_free[bank])
        port_free = start + d
        stream = max(port_free, input_free)
        input_free = stream + m
        bank_free[bank] = input_free + (d - 1 if skew else 0)
        end = stream + m + (2 * d - 2 if skew else d - 1)
        events.append((start, stream, end))
    return events


class ArrayTests(unittest.TestCase):
    def test_bank_timing_against_event_schedule(self):
        for d in (2, 4, 8, 16, 32):
            for m in (1, max(1, d - 1), d, d + 1, 3 * d):
                for folds in range(1, 18):
                    for skew in (False, True):
                        self.assertEqual(bank_cycles(m, d, folds, skew),
                                         golden_bank_events(m, d, folds, skew)[-1][2])

    def test_source_rtl_anchors(self):
        # References: gemmini/partition/TIMING_AUDIT.md, first array-timing table.
        # Numbers are parsed from the source table by the test, never copied as expected values.
        data = json.loads((ROOT / "configs/chips/tessera_revision.json").read_text())
        path = Path(data["workspace"]) / "gemmini/partition/TIMING_AUDIT.md"
        for line in path.read_text().splitlines():
            if not line.startswith("| 8x8 square |") and not line.startswith("| 32x32 square"):
                continue
            columns = [x.strip() for x in line.strip("|").split("|")]
            d = int(columns[0].split("x")[0])
            m, folds, first, ii = map(int, columns[1:5])
            grain = 8 if "strips" in columns[0] else d
            a = array_cost(m, d, d, 1, "tessera", d, d, d, grain)
            self.assertEqual(a.cycles - a.ingress_cycles, first)
            self.assertEqual(max(m, d), ii)
            self.assertEqual(bank_cycles(m, d, folds, False) + (d // grain - 1),
                             first + (folds - 1) * ii)

    def test_no_padding_energy_and_fixed_skew_mapping(self):
        for shape in ((1, 33, 17), (19, 128, 8), (7, 8, 128), (128, 129, 129)):
            m, n, k = shape
            c = load_config()
            rect = gemm_cost(m, n, k, 1, "tessera", c, (32, 64))
            square = gemm_cost(m, n, k, 1, "square", c, (32, 64))
            skew = gemm_cost(m, n, k, 1, "skew", c, (32, 64))
            for key in ("useful_macs", "array_energy_J", "sram_read_bytes", "sram_write_bytes"):
                self.assertEqual(getattr(rect, key), getattr(square, key))
                self.assertEqual(getattr(rect, key), getattr(skew, key))
            self.assertGreaterEqual(skew.array_cycles, rect.array_cycles)

    def test_capacity_and_mac_conservation(self):
        c = load_config()
        for batch in (1, 8):
            for m, n, k in ((1, 14336, 4096), (512, 4096, 14336), (17, 31, 19)):
                p = plan_memory(m, n, k, c, batch)
                self.assertLessEqual(p.peak_live_bytes, c.sram_bytes)
                self.assertEqual(sum(mm * nn * count for mm, nn, count, _ in p.chunks), m * n)
                for variant in VARIANTS:
                    r = gemm_cost(m, n, k, batch, variant, c)
                    self.assertEqual(r.useful_macs, batch * m * n * k)
                    self.assertGreaterEqual(r.array_cycles * c.pe_count, r.useful_macs)


class GraphTests(unittest.TestCase):
    def test_native_flash_attention_and_default_dispatch(self):
        from neusim.npusim.frontend.llm_ops_lib import create_multi_head_flash_attention_op, create_einsum_op
        from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
        from neusim.run_scripts.run_tessera import chip_config
        from neusim.configs.chips.ChipConfig import ChipConfig
        for batch, q, kv, hq, hkv, dim in ((1, 1, 271, 8, 2, 80), (2, 19, 145, 8, 8, 64)):
            op = create_multi_head_flash_attention_op([batch,q,hq,dim], [batch,kv,hkv,dim], [batch,kv,hkv,dim])
            fill_operators_execution_info([op], chip_config(load_config(), "tessera"))
            self.assertEqual(op.stats.tessera_details["useful_macs"], 2*batch*hq*q*kv*dim)
            self.assertAlmostEqual(op.stats.total_energy_J, op.stats.tessera_details["energy_J"])
        # The original dispatch must remain usable with a default ChipConfig.
        old = fill_operators_execution_info([create_einsum_op([128,256], [256,128], "MK;KN->MN")], ChipConfig())[0]
        self.assertEqual(old.stats.tessera_details, {})
        self.assertGreater(old.stats.execution_time_ns, 0)
        self.assertGreater(old.stats.total_energy_J, 0)

    def test_real_prefill_compaction_and_phi_bias(self):
        from neusim.npusim.frontend.tessera_workloads import llm_graph
        from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
        from neusim.run_scripts.run_tessera import chip_config, aggregate, trace_batches, requests_from_batch
        config = json.loads((ROOT / "configs/chips/tessera_revision.json").read_text())
        for tag in ("llama3-8b_conv300", "phi2_conv300", "llama2-70b-tp4_conv300"):
            requests = requests_from_batch(next(trace_batches(config["traces"][tag]))[1])
            for variant in ("tessera", "skew", "square_fixed", "independent", "ws"):
                values = []
                for compact in (False, True):
                    ops = llm_graph(config["models"][tag], requests, load_config().operand_bytes,
                        layers=3, fixed_mapping=variant in ("skew", "square_fixed"), compact_layers=compact)
                    fill_operators_execution_info(ops, chip_config(load_config(), variant))
                    values.append(aggregate(ops, config["models"][tag], "test", variant))
                    if tag == "phi2_conv300":
                        bias = [op for op in ops if op.name.endswith("_bias")]
                        self.assertEqual(len(bias), (1 if compact else 3) * 4 + 1)
                for key in ("seconds", "energy_J", "hbm_read_bytes", "hbm_write_bytes", "useful_macs"):
                    self.assertAlmostEqual(values[0][key], values[1][key], delta=max(1e-12, values[0][key] * 1e-12))

    def test_depthwise_lowering_against_channelwise_definition(self):
        from neusim.run_scripts.tessera_supplement import load_networks, dnn_graph
        config = json.loads((ROOT / "configs/chips/tessera_revision.json").read_text())
        nn = next(nn for nn in load_networks(config["workspace"]) if nn.net_name == "MobileNet-v1")
        ops, coverage = dnn_graph(nn, load_config().operand_bytes)
        tested = 0
        for row in coverage:
            if row["type"] != "DWConvLayer":
                continue
            layer = nn[row["layer"]]
            # Each output channel uses only its own input channel and spatial filter.
            expected = layer.nofm * layer.hofm * layer.wofm * layer.sfil * layer.sfil
            self.assertEqual(row["useful_macs"], expected)
            spec = next(op.tessera_spec for op in ops if op.name == row["layer"] + "/conv")
            self.assertEqual(spec["N"], 1)
            self.assertEqual(spec["batch"], layer.nofm)
            tested += 1
        self.assertGreater(tested, 0)

    def test_compact_layer_accounting_against_expanded_graph(self):
        from neusim.npusim.frontend.tessera_workloads import llm_graph
        from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
        from neusim.run_scripts.run_tessera import chip_config, aggregate
        models = json.loads((ROOT / "configs/chips/tessera_revision.json").read_text())["models"]
        for tag in ("llama3-8b_conv300", "phi2_conv300"):
            for requests in ([(1, 271)], [(19, 145)], [(1, 512), (3, 80)]):
                for variant in ("tessera", "ws"):
                    values = []
                    for compact in (False, True):
                        ops = llm_graph(models[tag], requests, load_config().operand_bytes,
                                        layers=3, compact_layers=compact)
                        fill_operators_execution_info(ops, chip_config(load_config(), variant))
                        values.append(aggregate(ops, models[tag], "test", variant))
                    for key in ("seconds", "energy_J", "hbm_read_bytes", "hbm_write_bytes", "useful_macs"):
                        self.assertAlmostEqual(values[0][key], values[1][key], delta=max(1e-12, values[0][key] * 1e-12))

    def test_native_operator_and_energy_totals(self):
        from neusim.configs.chips.ChipConfig import ChipConfig
        from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
        from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
        from dataclasses import asdict
        op = create_einsum_op([3, 17], [17, 33], "MK;KN->MN")
        chip = ChipConfig(array_backend="tessera", tessera_parameters=asdict(load_config()))
        fill_operators_execution_info([op], chip)
        self.assertEqual(op.stats.tessera_details["useful_macs"], 3 * 17 * 33)
        self.assertAlmostEqual(op.stats.total_energy_J, op.stats.tessera_details["energy_J"])

    def test_residency_and_graph_reduction(self):
        from neusim.npusim.frontend.tessera_workloads import make_op
        from neusim.npusim.frontend.tessera_analysis import analyze_operators
        from neusim.configs.chips.ChipConfig import ChipConfig
        from dataclasses import asdict
        c = load_config()
        chip = ChipConfig(array_backend="tessera", tessera_parameters=asdict(c))
        ops = [make_op("produce", dict(kind="vector", inputs={"a": 128}, outputs={"b": 128}, vector_ops=64)),
               make_op("consume", dict(kind="vector", inputs={"b": 128}, outputs={"c": 128}, vector_ops=64, final=True))]
        analyze_operators(ops, chip)
        self.assertEqual(sum(o.stats.tessera_details["hbm_read_bytes"] for o in ops), 128)
        self.assertEqual(sum(o.stats.tessera_details["hbm_write_bytes"] for o in ops), 128)
        self.assertAlmostEqual(sum(o.stats.dynamic_energy_hbm_J for o in ops),
                               128 * (c.hbm_read_pj_per_byte + c.hbm_write_pj_per_byte) * 1e-12)

    def test_gqa_useful_work_independent_of_grouping(self):
        from neusim.npusim.frontend.tessera_analysis import attention_cost
        for q, length in ((1, 271), (19, 145), (128, 512)):
            for hq, hkv in ((8, 8), (8, 2), (8, 1)):
                s = dict(q=q, kv=length, hq=hq, hkv=hkv, d=80, batch=1)
                r = attention_cost(s, load_config(), "tessera")
                self.assertEqual(r["useful_macs"], 2 * hq * q * length * 80)
                self.assertLessEqual(r["peak_live_bytes"], load_config().sram_bytes)


if __name__ == "__main__":
    unittest.main()
