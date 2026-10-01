"""Goldens for the grain-independent Planaria-derived marginal selector."""
from collections import OrderedDict
from itertools import product
import math,random,unittest
from unittest.mock import patch
from neusim.run_scripts.run_feature_qos_scheduler import FREQ,POLICY,MatchedCosts,run_matched
from neusim.run_scripts.tessera_qos_joint import JointSLAAllocator


class Task:
    def __init__(self,times,priority,slack):
        self.times=[math.inf]+list(times);self.priority=priority
        self.start_time=self.current_time=0.;self.sla=slack/FREQ*1e3
    def get_remaining_estimated_time(self,c):return self.times[c]


def possible(queue,units):
    return {key:[c for c in range(units,0,-1) if task.times[c]<=task.sla*1e-3*FREQ]
            for key,task in queue.items()}


class JointSelectionTests(unittest.TestCase):
    # /root: decision start — regress live-SRAM and mapping-contract failures.
    def test_nonmonotone_mapping_fit_is_not_binary_searched(self):
        queue=OrderedDict([(0,Task([9,5,3,2],1,10))])
        # Two different legal geometries fit at one and three units; their
        # neighboring mappings do not fit the same fragmented free region.
        policy=JointSLAAllocator()
        allocation,wait,_=policy.select_epoch(queue,possible(queue,4),[
            (0,4,lambda k,c:c in (1,3),{},{})])
        self.assertEqual(allocation,{0:3});self.assertEqual(wait,0)

    def test_sram_lower_bound_can_advance_to_pinned_geometry(self):
        queue=OrderedDict([(0,Task([math.inf,8,6,5],1,10))])
        result=JointSLAAllocator()(queue,possible(queue,4),4,
            executable=lambda k,c:c>=2,minimum_units={0:1})
        self.assertEqual(result,{0:4})

    def test_selected_mapping_and_live_sram_share_are_preserved(self):
        from neusim.npusim.backend.tests.test_matched_qos import MatchedQoSTests
        for arch in ('Planaria-32','Tessera-8'):
            model=MatchedQoSTests().fixture(arch,capacity=1000)
            for seed in (7,19,41):
                trace=[]
                run_matched([(0,'test',1),(1,'test',3),(3,'test',2)],model,{'test':100},seed,
                    asynchronous=arch.startswith('Tessera'),trace=trace,allocation_policy=JointSLAAllocator())
                for row in trace:
                    if row['event']=='dispatch_resources':
                        self.assertGreaterEqual(row['reserved_units'],row['sram_units'])
                        self.assertLessEqual(row['reserved_units'],model.units)
                    if row['event']=='tile_start' and row['k0']==0:
                        selected=model.profiles['test'][row['share']][row['layer']]
                        self.assertEqual(tuple(row['memory_tile']),selected.tile)
                        self.assertEqual(tuple(row['geometry']),selected.geometry)
    # /root: decision end

    def test_never_calls_original_allocator_or_fallback(self):
        queue=OrderedDict([(0,Task([8,4,3,2],1,10))])
        with (patch.object(POLICY,'assign_cores_if_tasks_fit',side_effect=AssertionError('Old allocator called')),
              patch.object(POLICY,'assign_cores_if_tasks_not_fit',side_effect=AssertionError('Old allocator called'))):
            self.assertEqual(JointSLAAllocator()(queue,possible(queue,4),4),{0:4})

    def test_late_tasks_still_make_progress(self):
        queue=OrderedDict([(0,Task([20,10,8,7],1,1)),(1,Task([16,9,8,6],2,1))])
        allocation=JointSLAAllocator()(queue,possible(queue,4),4)
        self.assertGreater(sum(allocation.values()),0)
        self.assertLessEqual(sum(allocation.values()),4)

    def test_native_transfer_and_work_conservation_without_borrowing(self):
        from neusim.npusim.backend.tests.test_matched_qos import MatchedQoSTests
        from collections import Counter
        for arch in ('Planaria-32','Tessera-8'):
            model=MatchedQoSTests().fixture(arch,capacity=1000)
            for seed in (7,19,41):
                trace=[]
                done,_,_=run_matched([(0,'test',1),(1,'test',1),(3,'test',2)],model,{'test':100},seed,
                    asynchronous=arch=='Tessera-8',trace=trace,allocation_policy=JointSLAAllocator())
                self.assertEqual(len(done),3)
                ledger=Counter();hbm=0
                for r in trace:
                    self.assertNotEqual(r['event'],'allocation_progress_repair')
                    self.assertFalse(r.get('borrowed',False))
                    if r['event']=='dispatch_resources':self.assertLessEqual(r['reserved_units'],model.units)
                    if r['event']=='group_start':self.assertLessEqual(r['held_bytes'],1000)
                    if r['event']!='tile_start':continue
                    for m in range(r['m'],r['m']+r['h']):
                        for n in range(r['n'],r['n']+r['w']):
                            for k in range(r['k0'],r['k0']+r['k']):ledger[r['task_id'],r['b'],m,n,k]+=1
                    hbm+=2*(r['h']*r['k']+r['k']*r['w'])
                    if r['k0']+r['k']==11:hbm+=2*r['h']*r['w']
                expected={(i,b,m,n,k) for i in range(3) for b in range(2) for m in range(9)
                          for n in range(17) for k in range(11)}
                self.assertEqual(set(ledger),expected);self.assertEqual(set(ledger.values()),{1})
                self.assertEqual(sum(r['bytes'] for r in trace if r['event']=='hbm_service'),hbm)

    def test_native_single_group_latency_golden(self):
        import json
        for arch,grain in (('Planaria-32',32),('Tessera-8',8)):
            for resident in (False,True):
                units=(64//grain)**2;m=10
                row=dict(kind='matrix',key=f'1x{m}x{grain}x{grain}',hbm_bytes=0 if resident else 1,
                         vu_ns=0,reduction_ops=0,mapping_json=json.dumps(dict(
                             geometry=[grain,grain,False,1,grain,grain],memory_tile=[m,grain,grain])))
                model=MatchedCosts({'test':{c:[row] for c in range(1,units+1)}},dict(
                    side=64,grain=grain,capacity_bytes=1<<20,frequency_Hz=FREQ,
                    hbm_bytes_per_cycle=16,hbm_latency_cycles=0),arch)
                trace=[]
                done,_,_=run_matched([(100,'test',1)],model,{'test':100},7,trace=trace,
                    asynchronous=arch=='Tessera-8',allocation_policy=JointSLAAllocator())
                share=next(r['share'] for r in trace if r['event']=='tile_start')
                hbm=0 if resident else 2*(2*m*grain+grain*grain)
                array=m+3*grain-(2 if arch=='Planaria-32' else 1)
                self.assertEqual(done[0]['finish_cycle'],100+max(array,hbm/16*units/share))

    def test_planaria_uses_native_arrival_completion_events(self):
        from neusim.npusim.backend.tests.test_matched_qos import MatchedQoSTests
        for arch in ('Planaria-32','Tessera-8'):
            model=MatchedQoSTests().fixture(arch)
            trace=[]
            done,_,_=run_matched([(100,'test',1),(100000,'test',1)],model,{'test':100},7,
                                asynchronous=arch.startswith('Tessera-'),trace=trace,allocation_policy=JointSLAAllocator())
            self.assertEqual(len(done),2)
            self.assertGreater(sum(r['event']=='tile_start' for r in trace),len(done))
            self.assertEqual([r['cycle'] for r in trace if r['event']=='allocation'],[100,100000])

    def test_finished_array_keeps_fixed_completion_with_zero_quota(self):
        queue=OrderedDict([(0,Task([40]*4,1,100)),(1,Task([100,80,60,4],1,30))])
        policy=JointSLAAllocator()
        result=policy.select_epoch(queue,possible(queue,4),[
            (0,4,lambda k,c:True,{0:2},{0:40}),
            (20,4,lambda k,c:True,{}, {0:40})])
        self.assertEqual(result,({0:0,1:4},20,4))

    def test_monotone_probe_matches_full_feasibility_mask(self):
        queue=OrderedDict([(0,Task([20,10,8,7],1,10)),(1,Task([16,9,8,6],2,10))])
        for first,second in product(range(1,6),repeat=2):
            mask=lambda k,c:c>=(first,second)[k]
            a=JointSLAAllocator()(queue,possible(queue,4),4,mask)
            b=JointSLAAllocator()(queue,possible(queue,4),4,mask,monotone_executable=True)
            self.assertEqual(a,b)

    def test_remaining_vu_matches_independent_native_k_tile_sum(self):
        from neusim.npusim.backend.tests.test_matched_qos import MatchedQoSTests
        from neusim.run_scripts.run_feature_qos_scheduler import MatchedTask
        for arch in ('Planaria-32','Tessera-8'):
            model=MatchedQoSTests().fixture(arch)
            for c in (1,model.units//2,model.units):
                p=model.profiles['test'][c][0];kt=p.tile[2]
                for k0 in range(0,p.shape[3],kt):
                    expected=0.
                    for start in range(k0,p.shape[3],kt):
                        k=min(kt,p.shape[3]-start)
                        expected+=9*17*(math.ceil(k/p.geometry[0])-1+int(start>0))*p.vu_per_add
                    self.assertEqual(model.output_tile_vu(p,9,17,k0),expected)
                task=MatchedTask(0,'test',0,0,1,100,model)
                total=2*model.output_tile_vu(p,9,17,0)
                self.assertEqual(task.remaining_vu_work(c),total)
                self.assertEqual(task.get_shared_resource_estimated_time(c),max(task.get_remaining_estimated_time(c),total*model.units/c))
    def test_resource_refinement_preserves_equivalent_curves(self):
        # Exact same physical choices represented with four different units.
        for seed in (7,19,41):
            rng=random.Random(seed)
            coarse=OrderedDict((i,Task([rng.randint(2,30) for _ in range(4)],rng.randint(1,9),10)) for i in range(3))
            expected=JointSLAAllocator()(coarse,possible(coarse,4),4)
            for scale in (1,4,16,64):
                fine=OrderedDict((k,Task([t.times[c//scale] if c%scale==0 else math.inf for c in range(1,4*scale+1)],t.priority,10)) for k,t in coarse.items())
                actual=JointSLAAllocator()(fine,possible(fine,4*scale),4*scale)
                self.assertEqual(actual,{k:c*scale for k,c in expected.items()})

    def test_planaria_minimum_deadline_pass_is_preserved(self):
        # Exactly full feasible minima leave no spare-allocation ambiguity.
        queue=OrderedDict((i,Task([100]*(need-1)+[9]*(17-need),p,10))
                          for i,(need,p) in enumerate(((8,1),(5,3),(3,2))))
        feasible=possible(queue,16)
        self.assertEqual(JointSLAAllocator()(queue,feasible,16),
                         POLICY.assign_cores_if_tasks_fit(queue,feasible,16))

    def test_spare_units_cross_legal_gaps_and_avoid_slow_points(self):
        queue=OrderedDict([(0,Task([9,12,9,9],1,10)),
                           (1,Task([9,9,3,4],1,10))])
        choice=JointSLAAllocator()(queue,possible(queue,4),4)
        # Both initially need one. Two surplus units shorten only request 1.
        self.assertEqual(choice,{0:1,1:3})

    def test_finer_intermediate_choice_fills_unused_capacity(self):
        coarse=OrderedDict([(0,Task([9,9,9,9],1,10)),(1,Task([30,30,9,9],1,10))])
        self.assertEqual(JointSLAAllocator()(coarse,possible(coarse,4),4),{0:1,1:3})
        fine=OrderedDict([(0,Task([9]*16,1,10)),(1,Task([30]*8+[9]*8,1,10))])
        result=JointSLAAllocator()(fine,possible(fine,16),16)
        self.assertEqual(result,{0:1,1:9})
        self.assertLess(sum(result.values())/16,1)

    def test_live_reservations_and_executable_masks(self):
        queue=OrderedDict([(0,Task([8,4,3,2],1,10)),(1,Task([30,20,9,4],2,10))])
        for seed in (7,19,41):
            rng=random.Random(seed)
            for _ in range(20):
                a,b=rng.randint(0,4),rng.randint(0,4)
                result=JointSLAAllocator()(queue,possible(queue,4),4,minimum_units={0:a,1:b})
                if a+b>4:self.assertIsNone(result)
                else:
                    self.assertGreaterEqual(result[0],a);self.assertGreaterEqual(result[1],b)
                    self.assertLessEqual(sum(result.values()),4)
