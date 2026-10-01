"""Independent loop counters, RTL timing anchors and native end-to-end checks."""
from contextlib import redirect_stdout
from io import StringIO
import json
from math import ceil
from pathlib import Path
import unittest
import random

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.tessera_partitioned import counts, geometries, BACKEND, VARIANTS, select_geometry
from neusim.npusim.backend.tessera_partitioned import FirstFitFabric, first_fit_rounds, packed_lanes
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op, create_multi_head_flash_attention_op
from neusim.npusim.frontend.op_analysis_lib import calculate_vmem_time_ns, fill_operators_execution_info

ROOT = Path(__file__).resolve().parents[4]


# chenyi9: decision start — independently check Algorithm 2 and its extra async mode.
class Algorithm2Tests(unittest.TestCase):
    def test_rounds_against_cell_reference(self):
        """Boolean-cell golden, independent of production bit masks/counters."""
        for seed in range(7):
            rng = random.Random(seed)
            weights = [(i, rng.choice((8,16,32)), rng.choice((8,16,32))) for i in range(80)]
            gold = [[]]
            used = set()
            for identity,h,w in weights:
                while True:
                    options = [(r,c) for r in range(0,33-h,8) for c in range(0,33-w,8)
                        if not used.intersection((y,x) for y in range(r,r+h,8) for x in range(c,c+w,8))]
                    if options:
                        r,c = options[0]
                        used.update((y,x) for y in range(r,r+h,8) for x in range(c,c+w,8))
                        gold[-1].append((identity,(r,c,h,w)))
                        break
                    gold.append([]);used.clear()
            self.assertEqual(first_fit_rounds(weights,32,8),gold)
        # Source: diagnosed GoogleNet conv2_3x3: 9 K blocks x 3 N strips.
        rounds = first_fit_rounds([(i,64,64) for i in range(9*3)],128,8)
        self.assertEqual(list(map(len,rounds)),[4,4,4,4,4,4,3])

    def test_cross_gemm_partial_round_and_fragmentation(self):
        weights = [('A',8,8),('B',8,8),('C',8,32)]
        result = first_fit_rounds(weights,32,8)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0][-1],('C',(8,0,8,32)))
        fabric = FirstFitFabric(32,8)
        for i in range(16):
            self.assertIsNotNone(fabric.place(i,8,8))
        # Four free cells in different rows cannot make one 8 x 32 lane.
        for i in (0,5,10,15):fabric.release(i)
        before = fabric.rows.copy()
        self.assertIsNone(fabric.place('lane',8,32))
        self.assertEqual(fabric.rows,before)
        self.assertEqual(len(fabric.available(8,8)),4)
        for i in (1,2,3):fabric.release(i)
        self.assertEqual(fabric.place('lane',8,32),(0,0,8,32))

    def test_packed_rounds_keep_producer_storage_bound(self):
        for nk in range(1,12):
            for nn in range(1,8):
                for slots in (1,3,4,9,16):
                    for planes in (1,2,4,8):
                        # Explicit FCFS weights, closing when slots or a
                        # per-output producer plane runs out. No count helper.
                        rounds = 1;current=[]
                        for k in range(nk):
                            for n in range(nn):
                                if len(current)==slots or sum(x[1]==n for x in current)==planes:
                                    rounds+=1;current=[]
                                current.append((k,n))
                                self.assertLessEqual(sum(x[1]==n for x in current),planes)
                        predicted=ceil(nk*nn/packed_lanes(nk,nn,slots,planes))
                        self.assertEqual(rounds,predicted)
        for planes,want in ((1,9),(2,7),(4,7)):
            row=counts(1,1568,192,576,2,2,2,(64,64,False,planes,64,64),
                       'full',8,128,12<<20,1,2000,forced_tile=(1568,192,576),
                       sa_energy_accounting='padded_tiles',sram_read_accounting='padded_tiles')
            self.assertEqual(row['rounds'],want)
            self.assertLessEqual(row['peak_live_bytes'],12<<20)
            self.assertEqual(row['charged_sa_macs'],1568*27*64**2)

    @staticmethod
    def fixture(specs,capacity=1<<20,resident=True):
        """Hand-derived resident GEMMs, not paper workload/performance data.

        Each group is one fold. Independent no-skew timing golden:
        H load + M stream + H-1 drain + W transpose/extension + seams.
        Dtypes and FP32 workspace follow paper Strip memory, as production.
        """
        profiles={};infos={}
        for name,layers in specs.items():
            records=[];ll=[]
            for m,h,w in layers:
                time=m+2*h+w-1+(w//8-1)
                hbm=0 if resident else 2*(m*h+h*w+m*w)
                row=dict(kind='matrix',key=f'1x{m}x{w}x{h}',peak_live_bytes=4*(m*h+h*w)+8*m*w,
                         mapping_json=json.dumps(dict(geometry=[h,w,False,1,h,w],memory_tile=[m,w,h])),
                         time_ns=time,sa_ns=time,vu_ns=0,hbm_ns=hbm/16,hbm_bytes=hbm,reduction_ops=0)
                records.append(row);ll.append((name,len([row]),time,0.))
            suffix=[sum(x[2] for x in ll[i:]) for i in range(len(ll)+1)]
            profiles[name]={c:records for c in range(1,17)}
            infos[name]={c:(ll,suffix) for c in range(1,17)}
        return infos,profiles,dict(side=32,grain=8,capacity_bytes=capacity,frequency_Hz=1e9,hbm_bytes_per_cycle=16)

    def test_async_arrival_without_global_barrier_and_dependencies(self):
        from neusim.run_scripts.run_feature_qos_scheduler import run_regions
        specs={'old':[(100,8,8)],'new':[(10,8,32)],'chain':[(10,8,8),(12,8,8)]}
        infos,profiles,c=self.fixture(specs)
        for arrival in (1,5,17):
            trace=[]
            records,_,_=run_regions([(0,'old',1),(arrival,'new',1)],infos,{'old':1,'new':1},9,profiles,c,trace)
            finish={r['network']:r['finish_cycle'] for r in records}
            self.assertEqual(finish,{'old':123,'new':arrival+60})
            start=next(r for r in trace if r['event']=='region_start' and r['task_id']==1)
            self.assertEqual(start['cycle'],arrival)
            self.assertEqual(start['box'][2:],[8,32])
            self.assertLess(start['cycle'],finish['old'])
            # The requested base fix is distinct from the extra async feature.
            synchronous=[]
            run_regions([(0,'old',1),(arrival,'new',1)],infos,{'old':1,'new':1},9,
                        profiles,c,synchronous,asynchronous=False)
            sync_start=next(r for r in synchronous if r['event']=='region_start' and r['task_id']==1)
            self.assertEqual(sync_start['cycle'],123)
        simultaneous=[]
        run_regions([(0,'old',1),(0,'new',1)],infos,{'old':1,'new':1},9,
                    profiles,c,simultaneous,asynchronous=False)
        self.assertEqual([r['cycle'] for r in simultaneous if r['event']=='group_start'],[0,0])
        trace=[]
        records,_,_=run_regions([(0,'chain',1)],infos,{'chain':1},9,profiles,c,trace)
        self.assertEqual(records[0]['finish_cycle'],33+35)
        starts=[r for r in trace if r['event']=='group_start']
        self.assertEqual([r['cycle'] for r in starts],[0,33])

    def test_async_finite_sram_and_shared_hbm(self):
        from neusim.run_scripts.run_feature_qos_scheduler import run_regions
        specs={'old':[(100,8,8)],'new':[(10,8,32)]}
        infos,profiles,c=self.fixture(specs)
        # Source: same fixture workspace. Either fits, but both cannot coexist.
        c['capacity_bytes']=profiles['old'][16][0]['peak_live_bytes']
        trace=[]
        rr,_,_=run_regions([(0,'old',1),(5,'new',1)],infos,{'old':1,'new':1},9,profiles,c,trace)
        starts=[r for r in trace if r['event']=='group_start' and r['task_id']==1]
        self.assertEqual(starts[0]['cycle'],123)
        self.assertTrue(all(r['held_bytes']<=c['capacity_bytes'] for r in trace if r['event']=='group_start'))
        infos,profiles,c=self.fixture(specs,resident=False)
        trace=[]
        rr,_,_=run_regions([(0,'old',1),(0,'new',1)],infos,{'old':1,'new':1},9,profiles,c,trace)
        services=[r for r in trace if r['event']=='hbm_service']
        self.assertEqual(sum(r['bytes'] for r in services),sum(profiles[n][16][0]['hbm_bytes'] for n in specs))
        for a,b in zip(services,services[1:]):self.assertLessEqual(a['finish_cycle'],b['start_cycle'])
        for r in rr:
            end=next(x['finish_cycle'] for x in services if x['task_id']==r['task_id'])
            self.assertGreaterEqual(r['finish_cycle'],end)

    def test_ready_gemm_fills_partial_tail_without_enabling_later_arrivals(self):
        """Five 16x16 weights leave three lanes free before the last retires."""
        from unittest.mock import patch
        from neusim.run_scripts import run_feature_qos_scheduler as scheduler
        infos,profiles,c=self.fixture({'old':[(100,16,16)],'new':[(10,8,32)]})
        old=profiles['old'][16][0]
        old.update(key='1x100x16x80',peak_live_bytes=4*(100*80+80*16)+4*5*100*16,
                   mapping_json=json.dumps(dict(geometry=[16,16,False,4,16,16],memory_tile=[100,16,80])),
                   time_ns=248,sa_ns=248,vu_ns=0,reduction_ops=100*16*4)
        for share in infos['old']:infos['old'][share]=([('old',1,248,0.)],[248,0.])
        # Isolate placement from allocation heuristics: a valid all-to-first
        # target leaves the second ready job to the idle-hole borrowing pass.
        def allocation(queue,possible,units,*args):
            first=next(iter(queue));return {key:units if key==first else 0 for key in queue}
        for arrival,asynchronous,start in ((0,False,148),(5,False,248),(5,True,148)):
            trace=[]
            with patch.object(scheduler.POLICY,'assign_cores_if_tasks_fit',allocation), \
                 patch.object(scheduler.POLICY,'assign_cores_if_tasks_not_fit',allocation):
                scheduler.run_regions([(0,'old',1),(arrival,'new',1)],infos,{'old':1,'new':1},9,
                                      profiles,c,trace,asynchronous)
            new=next(r for r in trace if r['event']=='region_start' and r['task_id']==1)
            self.assertEqual(new['cycle'],start)

    def test_native_operator_dispatch_and_work_conservation(self):
        """System smoke on sourced CNN layers; no throughput/performance sweep."""
        from collections import defaultdict
        from neusim.run_scripts.run_feature_cnn_qos import INPUT,rows,chip,profile_job
        from neusim.run_scripts.run_feature_qos_scheduler import run_regions
        source=list(rows(INPUT/'cnn_layers.csv'))
        selections=[('googlenet','conv2_3x3'),('mobilenet_v1','conv2-dw')]
        c=chip(objective='e2e_latency')
        limits=dict(side=c.sa_dim,grain=c.tessera_parameters['grain'],
                    capacity_bytes=c.vmem_size_MB*1024**2,frequency_Hz=c.freq_GHz*1e9,
                    hbm_bytes_per_cycle=c.hbm_bw_GBps*1024**3/1e9)
        units=(limits['side']//limits['grain'])**2
        for net,layer in selections:
            src=next(r for r in source if r['network']==net and r['layer']==layer)
            dims=tuple(int(src[k]) for k in ('B','M','N','K'))
            row=profile_job(('matrix','x'.join(map(str,dims)),dims,'Tessera-8',16,'e2e_latency'))[0]
            n=int(row['tiles']);period=float(row['time_ns'])/n
            # Only the fully allocated entry is used in this isolated-task test.
            infos={'real':{i:([('source',n,period,row['energy_J'])],[n*period,0.]) for i in range(1,units+1)}}
            profiles={'real':{i:[row] for i in range(1,units+1)}}
            mp=json.loads(row['mapping_json']);mt,nt,kt=mp['memory_tile'];g=mp['geometry']
            for asynchronous in (False,True):
                trace=[]
                done,_,_=run_regions([(0,'real',1)],infos,{'real':1},11,profiles,limits,trace,asynchronous)
                self.assertEqual(len(done),1)
                self.assertGreaterEqual(done[0]['finish_cycle'],row['time_ns'])
                self.assertEqual(sum(r['bytes'] for r in trace if r['event']=='hbm_service'),row['hbm_bytes'])
                groups=defaultdict(list)
                charged=0
                for r in trace:
                    if r['event']!='region_start':continue
                    groups[r['group']].extend(r['first_weight']+i*r['weight_stride'] for i in range(r['weight_count']))
                    charged+=r['weight_count']*mt*g[-2]*g[-1]
                self.assertEqual(charged,row['charged_sa_macs'])
                self.assertEqual(len(groups),n)
                expected=list(range(ceil(kt/g[0])*ceil(nt/g[1])))
                for weights in groups.values():self.assertEqual(sorted(weights),expected)
                self.assertTrue(all(r['held_bytes']<=limits['capacity_bytes'] for r in trace if r['event']=='group_start'))
# chenyi9: decision end


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
