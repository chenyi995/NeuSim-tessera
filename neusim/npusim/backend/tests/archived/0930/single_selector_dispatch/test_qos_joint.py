"""Exhaustive goldens for the single Planaria-derived joint selector."""
from collections import OrderedDict
from itertools import product
import math,random,unittest
from unittest.mock import patch
from neusim.run_scripts.run_feature_qos_scheduler import FREQ,POLICY
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


if __name__=='__main__':unittest.main()
