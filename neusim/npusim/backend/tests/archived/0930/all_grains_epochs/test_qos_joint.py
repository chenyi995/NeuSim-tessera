"""Exhaustive goldens for the single Planaria-derived joint selector."""
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


def objective(queue,allocation):
    urgency=rate=0.
    for key,task in queue.items():
        c=allocation[key];t=task.times[c];slack=task.sla*1e-3*FREQ
        if c and t<=slack:urgency+=task.priority/slack
        if c and math.isfinite(t):
            rate+=task.priority/slack*math.log1p(slack/t) if slack>0 else task.priority/t
    return urgency,rate


class JointSelectionTests(unittest.TestCase):
    def test_knapsack_choice_beats_greedy_density(self):
        queue=OrderedDict((i,Task([20]*(minimum-1)+[9]*(5-minimum),priority,10))
                          for i,(minimum,priority) in enumerate(((3,5),(2,3),(2,3))))
        choice=JointSLAAllocator()(queue,possible(queue,4),4)
        self.assertEqual(choice,{0:0,1:2,2:2})
        self.assertGreater(objective(queue,choice),objective(queue,{0:3,1:1,2:0}))

    def test_matches_exhaustive_joint_search(self):
        for seed in (7,19,41):
            rng=random.Random(seed)
            for units in (4,8):
                for _ in range(12):
                    queue=OrderedDict((i,Task([rng.randint(1,30) for c in range(units)],rng.randint(1,9),rng.randint(5,25))) for i in range(3))
                    choices=[dict(enumerate(c)) for c in product(range(units+1),repeat=len(queue)) if sum(c)<=units]
                    best=max(objective(queue,c) for c in choices)
                    actual=JointSLAAllocator()(queue,possible(queue,units),units)
                    actual_score=objective(queue,actual)
                    self.assertEqual(actual_score[0],best[0])
                    # NumPy log1p and the independent scalar libm reference can
                    # differ by an ulp; the sum contains three task terms.
                    self.assertTrue(math.isclose(actual_score[1],best[1],rel_tol=4e-15))

    def test_finer_options_retain_the_coarse_feasible_set(self):
        coarse=OrderedDict((i,Task(times,priority,10)) for i,(times,priority) in enumerate((([20,20,9,8],5),([20,9,8,7],3),([20,9,8,7],3))))
        expanded=OrderedDict()
        for key,task in coarse.items():
            times=[task.times[c//4] if c%4==0 else math.inf for c in range(1,17)]
            expanded[key]=Task(times,task.priority,10)
        a=JointSLAAllocator()(coarse,possible(coarse,4),4)
        b=JointSLAAllocator()(expanded,possible(expanded,16),16)
        self.assertEqual(objective(coarse,a),objective(expanded,b))
        for task in expanded.values():
            for c in range(1,17):
                if c%4:task.times[c]=max(1,20-c)
        fine=JointSLAAllocator()(expanded,possible(expanded,16),16)
        self.assertGreaterEqual(objective(expanded,fine),objective(coarse,a))

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

    def test_executable_mask_and_zero_budget(self):
        queue=OrderedDict([(0,Task([20,10,8,7],1,10)),(1,Task([16,9,8,6],2,10))])
        allowed={(0,2),(1,1),(1,3)}
        candidates=[dict(enumerate(c)) for c in product(range(5),repeat=2)
                    if sum(c)<=4 and all(not n or (k,n) in allowed for k,n in enumerate(c))]
        actual=JointSLAAllocator()(queue,possible(queue,4),4,lambda k,c:(k,c) in allowed)
        self.assertEqual(objective(queue,actual),max(objective(queue,c) for c in candidates))
        for units in (0,4):
            self.assertEqual(JointSLAAllocator()(queue,possible(queue,4),units,lambda k,c:False),{0:0,1:0})

    def test_single_selector_authorizes_asynchronous_admission(self):
        import json
        profiles={}
        for name,m in (('slow',100),('new',10)):
            row=dict(kind='matrix',key=f'1x{m}x8x8',hbm_bytes=0,vu_ns=0,reduction_ops=0,
                     mapping_json=json.dumps(dict(geometry=[8,8,False,1,8,8],memory_tile=[m,8,8])))
            profiles[name]={c:[row] for c in range(1,65)}
        model=MatchedCosts(profiles,dict(side=64,grain=8,capacity_bytes=1<<20,
                           frequency_Hz=FREQ,hbm_bytes_per_cycle=16,hbm_latency_cycles=0),'Tessera-8')
        trace=[]
        with (patch.object(POLICY,'assign_cores_if_tasks_fit',side_effect=AssertionError('Old allocator')),
              patch.object(POLICY,'assign_cores_if_tasks_not_fit',side_effect=AssertionError('Old allocator'))):
            done,_,_=run_matched([(0,'slow',1),(5,'new',1)],model,{'slow':100,'new':100},7,
                asynchronous=True,trace=trace,allocation_policy=JointSLAAllocator())
        # Independent hand golden: one fold takes M + 3*grain - 1 cycles.
        self.assertEqual({r['network']:r['finish_cycle'] for r in done},{'slow':123,'new':38})
        targets={}
        for r in trace:
            if r['event']=='allocation':targets.update(r['targets'])
            if r['event']=='group_start':
                self.assertGreater(targets[r['task_id']],0)
                self.assertFalse(r['borrowed'])
        self.assertFalse(any(r['event']=='allocation_progress_repair' for r in trace))

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
        model=MatchedQoSTests().fixture('Planaria-32')
        trace=[]
        done,_,_=run_matched([(100,'test',1),(100000,'test',1)],model,{'test':100},7,
                            trace=trace,allocation_policy=JointSLAAllocator())
        self.assertEqual(len(done),2)
        self.assertEqual([r['cycle'] for r in trace if r['event']=='allocation'],[100,100000])


if __name__=='__main__':unittest.main()
