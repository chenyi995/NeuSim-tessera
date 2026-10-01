"""Independent loop counters, RTL timing anchors and native end-to-end checks."""
from contextlib import redirect_stdout
from io import StringIO
import json
from math import ceil
from pathlib import Path
import unittest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.tessera_partitioned import counts, geometries, BACKEND, VARIANTS, select_geometry
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op, create_multi_head_flash_attention_op
from neusim.npusim.frontend.op_analysis_lib import calculate_vmem_time_ns, fill_operators_execution_info

ROOT = Path(__file__).resolve().parents[4]


def chip(variant="full", side=128, grain=32, **kwargs):
    return ChipConfig(array_backend=BACKEND, tessera_variant=variant, num_sa=1, sa_dim=side,
                      freq_GHz=1, vmem_size_MB=12, use_vu_for_small_matmul=False,
                      tessera_parameters={"grain": grain}, **kwargs)


def run(op, config):
    with redirect_stdout(StringIO()):
        return fill_operators_execution_info([op], config)[0]


def golden_count(b, m, n, k, tile, region):
    """Explicit non-vectorized loops; no production count/tail helper calls."""
    mt, nt, kt = tile
    rk, rn = region
    a_reads = b_reads = partial_writes = additions = 0
    hbm_a = hbm_b = 0
    for bi in range(b):
        for i in range(0, m, mt):
            mm = min(mt, m-i)
            for j in range(0, n, nt):
                nn = min(nt, n-j)
                groups = 0
                for t in range(0, k, kt):
                    kk = min(kt, k-t)
                    hbm_a += mm * kk * 2
                    hbm_b += kk * nn * 2
                    for x in range(0, kk, rk):
                        depth = min(rk, kk-x)
                        groups += 1
                        for y in range(0, nn, rn):
                            width = min(rn, nn-y)
                            a_reads += mm * depth * 2
                            b_reads += depth * width * 2
                            partial_writes += mm * width * 4
                additions += mm * nn * (groups-1)
    return a_reads, b_reads, partial_writes, additions, hbm_a, hbm_b


class PartitionTests(unittest.TestCase):
    def test_traffic_against_explicit_loops(self):
        c = chip(side=8, grain=2)
        for b, m, n, k in ((1, 3, 5, 7), (2, 9, 19, 17), (3, 4, 9, 12)):
            for geometry in geometries(m, n, k, c):
                r = counts(b, m, n, k, 2, 2, 2, geometry, "full", 2, 8, 1024, 1, 2000)
                gold = golden_count(b, m, n, k, r["memory_tile"], geometry[:2])
                actual = tuple(r[key] for key in ("array_a_read_bytes", "array_b_read_bytes", "partial_write_bytes",
                    "reduction_ops", "hbm_a_read_bytes", "hbm_b_read_bytes"))
                self.assertEqual(actual, gold)
                self.assertLessEqual(r["peak_live_bytes"], 1024)
                self.assertEqual(r["hbm_bytes"], gold[-2] + gold[-1] + 2*b*m*n)
                self.assertGreaterEqual(r["sa_cycles"]*64, b*m*n*k)

    def test_rtl_timing_anchors_and_tail_only_skew(self):
        # Read measured array latency from the RTL audit, not copied expectations.
        audit = ROOT.parents[1] / "gemmini/partition/TIMING_AUDIT.md"
        for line in audit.read_text().splitlines():
            if line.startswith("| 8x8 square |") or line.startswith("| 32x32 square, g=32 |"):
                fields = [s.strip() for s in line.strip("|").split("|")]
                dim = int(fields[0].split("x")[0])
                m, rounds, first, ii = map(int, fields[1:5])
                geometry = dim, dim, False, 1, dim, dim
                r = counts(1, m, dim, dim*rounds, 2, 2, 2, geometry, "full", dim, dim, 1<<20, 1, 2000)
                self.assertEqual(r["sa_cycles"]-dim, first+(rounds-1)*ii)
                skew = counts(1, m, dim, dim*rounds, 2, 2, 2, geometry, "skew", dim, dim, 1<<20, 1, 2000)
                self.assertEqual(skew["sa_cycles"]-r["sa_cycles"], dim-1)
                for key in ("hbm_bytes", "sram_bytes", "reduction_ops", "memory_tile"):
                    self.assertEqual(r[key], skew[key])

    def test_partition_reuse_stays_in_sram_when_capacity_suffices(self):
        shape = (1, 64, 128, 128)
        small = counts(*shape, 2, 2, 2, (32,32,False,1,32,32), "full", 32,128,12<<20,1,2000)
        large = counts(*shape, 2, 2, 2, (128,128,False,1,128,128), "full",32,128,12<<20,1,2000)
        self.assertEqual(small["hbm_bytes"], large["hbm_bytes"])
        self.assertEqual(small["array_a_read_bytes"], 4*large["array_a_read_bytes"])
        self.assertEqual(small["partial_write_bytes"], 4*large["partial_write_bytes"])
        self.assertEqual(large["reduction_ops"], 0)
        self.assertGreater(small["reduction_ops"], 0)

    def test_native_pipeline_selection_energy_and_frequency(self):
        for shape in ((4,128,80), (257,256,512)):
            m,n,k=shape
            results={}
            for variant in ("full", "square", "independent_noskew", "skew"):
                op=run(create_einsum_op([m,k],[k,n],"MK;KN->MN"), chip(variant))
                s=op.stats; d=s.tessera_details
                self.assertEqual(s.memory_traffic_bytes,d["hbm_bytes"])
                self.assertEqual(d["sram_bandwidth_model"], "per_pe_double_buffer")
                self.assertLessEqual(s.vmem_time_ns, max(s.sa_time_ns, s.vu_time_ns, s.memory_time_ns, s.ici_time_ns))
                self.assertEqual(op.dvfs_sa.frequency_GHz,1)
                self.assertEqual(s.execution_time_ns,max(s.sa_time_ns,s.vu_time_ns,s.memory_time_ns,s.ici_time_ns))
                self.assertAlmostEqual(s.total_energy_J,s.static_energy_J+s.dynamic_energy_J)
                results[variant]=op
            self.assertLessEqual(results["full"].stats.total_energy_J*results["full"].stats.execution_time_ns,
                                 results["square"].stats.total_energy_J*results["square"].stats.execution_time_ns)
            for key in ("geometry","memory_tile","hbm_bytes","sram_bytes","reduction_ops"):
                self.assertEqual(results["full"].stats.tessera_details[key],results["skew"].stats.tessera_details[key])

    def test_attention_and_full_graph(self):
        for b,q,kv,hq,hkv,d in ((1,1,271,8,2,80),(2,19,145,8,8,64),(1,257,1024,8,2,64)):
            op=run(create_multi_head_flash_attention_op([b,q,hq,d],[b,kv,hkv,d],[b,kv,hkv,d]),chip())
            detail=op.stats.tessera_details
            self.assertEqual(detail["useful_macs"],2*b*hq*q*kv*d)
            self.assertLessEqual(detail["peak_live_bytes"],12<<20)
            self.assertEqual(op.stats.memory_traffic_bytes,detail["hbm_bytes"])
            self.assertLessEqual(op.stats.vmem_time_ns, max(op.stats.sa_time_ns, op.stats.vu_time_ns, op.stats.memory_time_ns))
        from neusim.npusim.frontend.tessera_workloads import native_llm_graph
        model=json.loads((ROOT/"configs/chips/tessera_revision.json").read_text())["models"]["phi2_conv300"]
        for requests in ([(1,271)],[(19,145)],[(1,512),(3,80)]):
            totals=[]
            for compact in (False,True):
                ops=native_llm_graph(model,requests,chip(),layers=2,compact_layers=compact)
                with redirect_stdout(StringIO()):
                    result=fill_operators_execution_info(ops,chip())
                self.assertTrue(any("lm_head" in o.name for o in result))
                self.assertTrue(any("bias" in o.name for o in result))
                for o in result:
                    s = o.stats
                    self.assertEqual(s.execution_time_ns, max(s.sa_time_ns, s.vu_time_ns, s.memory_time_ns, s.ici_time_ns))
                    self.assertLessEqual(s.vmem_time_ns, s.execution_time_ns)
                    if s.tessera_details["sram_bytes"]:
                        self.assertGreater(s.dynamic_energy_sram_J, 0)
                totals.append((sum(o.stats.count*o.stats.execution_time_ns for o in result),
                               sum(o.stats.count*o.stats.total_energy_J for o in result)))
            self.assertEqual(totals[0][0],totals[1][0])
            self.assertAlmostEqual(totals[0][1],totals[1][1])

    def test_vu_fallback_does_not_execute_rejected_sa_activity(self):
        c=chip().model_copy(update={"use_vu_for_small_matmul": True})
        op=run(create_einsum_op([1,4096],[4096,4096],"MK;KN->MN"),c)
        d=op.stats.tessera_details
        self.assertEqual(d["engine"],"VU")
        for key in ("sa_arithmetic_ops","sa_cycles","rounds","reduction_ops","array_a_read_bytes",
                    "array_b_read_bytes","partial_write_bytes","reduction_read_bytes","reduction_write_bytes"):
            self.assertEqual(d[key],0)
        self.assertGreater(d["sram_bytes"],0)
        self.assertEqual(d["hbm_bytes"],op.stats.memory_traffic_bytes)
        self.assertLessEqual(op.stats.vmem_time_ns, max(op.stats.vu_time_ns, op.stats.memory_time_ns))

    # Codex: decision start — check supply, energy conservation and the native
    # control independently of mapping selection under the requested SRAM rule.
    def test_per_pe_supply_and_double_buffer_overlap(self):
        for side in (8, 32, 128):
            for freq in (0.5, 1, 2):
                c = chip(side=side, grain=8).model_copy(update={"freq_GHz": freq})
                op = create_einsum_op([3, 8], [8, 8], "MK;KN->MN")
                # Independent byte ledger: each PE has two 2-byte input paths
                # and one 4-byte output path; ping-pong does not double the width.
                lane_bytes = sum(2 + 2 + 4 for _ in range(side * side))
                traffic = 100 * lane_bytes
                op.stats.tessera_details = dict(model=BACKEND, engine="SA", sram_bytes=traffic)
                visible = calculate_vmem_time_ns(op, 7, 3, 11, c)
                d = op.stats.tessera_details
                self.assertEqual(d["sram_bandwidth_bytes_per_ns"], lane_bytes * freq)
                self.assertEqual(d["sram_service_time_ns"], 100 / freq)
                self.assertEqual(visible, 11)
                self.assertEqual(visible + d["sram_overlap_hidden_ns"], d["sram_service_time_ns"])
                self.assertEqual(d["sram_bytes"], traffic)

    def test_all_variants_have_same_supply_at_equal_pe_count(self):
        bandwidths = []
        for variant in VARIANTS:
            c = chip(variant)
            op = run(create_einsum_op([4, 80], [80, 128], "MK;KN->MN"), c)
            s = op.stats
            bandwidths.append(s.tessera_details["sram_bandwidth_bytes_per_ns"])
            self.assertEqual(s.execution_time_ns, max(s.sa_time_ns, s.vu_time_ns, s.memory_time_ns, s.ici_time_ns))
        self.assertEqual(len(set(bandwidths)), 1)

    def test_same_traffic_keeps_sram_dynamic_energy_when_activity_shortens(self):
        from neusim.npusim.backend.power_model import analyze_dynamic_energy
        for shape in ((4, 128, 80), (257, 256, 512)):
            m, n, k = shape
            c = chip()
            op = run(create_einsum_op([m, k], [k, n], "MK;KN->MN"), c)
            traffic = op.stats.tessera_details["sram_bytes"]
            # Native coefficient is power divided by the original byte rate,
            # independent of provisioned per-PE bandwidth or hidden service time.
            expected = traffic * c.dynamic_power_vmem_W / (c.vmem_bw_GBps * 1e9)
            for active_time in (0, 1, op.stats.vmem_time_ns):
                probe = op.model_copy(deep=True)
                probe.stats.vmem_time_ns = active_time
                analyze_dynamic_energy(probe, c)
                self.assertTrue(abs(probe.stats.dynamic_energy_sram_J / expected - 1) < 1e-12)
            probe.stats.tessera_details["sram_bytes"] *= 2
            analyze_dynamic_energy(probe, c)
            self.assertTrue(abs(probe.stats.dynamic_energy_sram_J / expected - 2) < 1e-12)

    def test_native_bandwidth_control_remains_available(self):
        for shape in ((4, 128, 80), (257, 256, 512)):
            m, n, k = shape
            c = chip().model_copy(update={"tessera_parameters": {"grain": 32, "sram_bandwidth_model": "native"}})
            op = run(create_einsum_op([m, k], [k, n], "MK;KN->MN"), c)
            s = op.stats
            self.assertEqual(s.vmem_time_ns, ceil(s.tessera_details["sram_bytes"] / c.vmem_bw_GBps))
            expected = c.dynamic_power_vmem_W * s.vmem_time_ns / 1e9
            actual = s.dynamic_energy_sram_J * op.dvfs_sram.voltage_conversion_power_efficiency_percent / 100
            self.assertTrue(abs(actual / expected - 1) < 1e-12)

    def test_invalid_bandwidth_model_is_rejected(self):
        c = chip().model_copy(update={"tessera_parameters": {"grain": 32, "sram_bandwidth_model": "typo"}})
        with self.assertRaisesRegex(ValueError, "unknown Tessera SRAM bandwidth model"):
            run(create_einsum_op([4, 80], [80, 128], "MK;KN->MN"), c)
    # Codex: decision end


if __name__ == "__main__":
    unittest.main()
