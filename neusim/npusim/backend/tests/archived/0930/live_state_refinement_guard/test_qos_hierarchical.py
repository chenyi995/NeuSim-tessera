"""Hand-derived allocation tests and exact original-policy degeneration."""
from collections import OrderedDict
import random
import unittest
from neusim.run_scripts.run_feature_qos_scheduler import POLICY,FREQ
from neusim.run_scripts.tessera_qos_hierarchical import CoarseFirstAllocator


class Task:
    # Synthetic hand golden: enough resource costs one cycle, otherwise three.
    def __init__(self,priority,minimum):
        self.priority=priority;self.minimum=minimum
        self.start_time=self.current_time=0
        self.sla=2/FREQ*1e3
    def get_remaining_estimated_time(self,cores):
        return 1 if cores>=self.minimum else 3


class HierarchicalTests(unittest.TestCase):
    def possible(self,queue,units):
        return {k:POLICY.get_possible_num_cores(t,FREQ,units) for k,t in queue.items()}

    def test_reference_grain_exact_allocation_and_rng(self):
        for units in (4,16):
            for minima in ((1,1,1),(2,2,2),(units,units,units),(units+1,units+1,units+1)):
                queue=OrderedDict((i,Task(p,m)) for i,(p,m) in enumerate(zip((1,3,3),minima)))
                possible=self.possible(queue,units)
                for seed in (7,19,41):
                    random.seed(seed)
                    if POLICY.check_if_tasks_all_fit(possible,units):
                        expected=POLICY.assign_cores_if_tasks_fit(queue,possible,units)
                    else:expected=POLICY.assign_cores_if_tasks_not_fit(queue,possible,units,FREQ)
                    state=random.getstate()
                    random.seed(seed)
                    actual=CoarseFirstAllocator(32)(queue,possible,units)
                    self.assertEqual(actual,expected);self.assertEqual(random.getstate(),state)

    def test_fine_admission_retains_both_coarse_admissions(self):
        queue=OrderedDict((i,Task(p,m)) for i,(p,m) in enumerate(((10,30),(10,30),(1,4))))
        possible=self.possible(queue,64)
        coarse=CoarseFirstAllocator(8,refinement=False)(queue,possible,64)
        fine=CoarseFirstAllocator(8)(queue,possible,64)
        self.assertEqual(coarse,{0:32,1:32,2:0})
        self.assertEqual(fine,{0:30,1:30,2:4})

    def test_no_extra_admission_returns_exact_incumbent(self):
        queue=OrderedDict((i,Task(10,30)) for i in range(2))
        possible=self.possible(queue,64)
        self.assertEqual(CoarseFirstAllocator(8)(queue,possible,64),{0:32,1:32})

    def test_nonmonotone_feasibility_is_not_assumed(self):
        queue=OrderedDict((i,Task(p,m)) for i,(p,m) in enumerate(((10,30),(10,30),(1,4))))
        # A native profile at 31 units may be worse than both 30 and 32.
        possible=self.possible(queue,64)
        for key in (0,1):possible[key].remove(31)
        result=CoarseFirstAllocator(8)(queue,possible,64)
        self.assertTrue(all(result[k] in possible[k] for k in queue))

    def test_intermediate_grain_uses_same_algorithm(self):
        queue=OrderedDict((i,Task(p,m)) for i,(p,m) in enumerate(((10,7),(10,7),(1,2))))
        self.assertEqual(CoarseFirstAllocator(16)(queue,self.possible(queue,16),16),{0:7,1:7,2:2})


if __name__=='__main__':unittest.main()
