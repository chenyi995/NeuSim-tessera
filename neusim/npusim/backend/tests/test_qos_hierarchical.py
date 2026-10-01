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

    def test_live_guard_rejects_lost_coarse_sla(self):
        policy=CoarseFirstAllocator(8);policy.last_coarse={0:32,1:0}
        proposal={0:30,1:2}
        def forecast(assignment,bounds=None):
            # Both finish one task on time, but the proposal loses the original
            # protected success. Aggregate SLA count alone is insufficient.
            times=(1,3) if assignment==policy.last_coarse else (3,1)
            return [dict(task_id=i,latency_ms=t,sla_ms=2,finish_cycle=t) for i,t in enumerate(times)]
        self.assertIs(policy.guard(proposal,forecast),policy.last_coarse)
        self.assertEqual(policy.audit['rejected_refinements'],1)

    def test_live_guard_accepts_improvement_and_keeps_fallback(self):
        policy=CoarseFirstAllocator(8);policy.last_coarse={0:32,1:0}
        proposal={0:30,1:2};original=policy.last_coarse.copy()
        def forecast(assignment,bounds=None):
            times=(1,3) if assignment==policy.last_coarse else (1.5,1.7)
            return [dict(task_id=i,latency_ms=t,sla_ms=2,finish_cycle=t) for i,t in enumerate(times)]
        self.assertIs(policy.guard(proposal,forecast),proposal)
        self.assertEqual(policy.last_coarse,original)

    def test_forecast_cutoff_matches_complete_native_outcome(self):
        from neusim.npusim.backend.tests.test_qos_reservations import ReservationTests
        from neusim.run_scripts.run_feature_qos_scheduler import run_matched
        from neusim.run_scripts.tessera_qos_hierarchical import ForecastRejected
        model=ReservationTests().model('Tessera-8',shape=(1,10,8,8))
        work=[(0,'test',1)]
        reference,_,_=run_matched(work,model,{'test':1},7,asynchronous=True)
        for cycles in (20,33,40):
            bound={0:dict(sla_ms=cycles/FREQ*1e3,latency_ms=0,finish_cycle=cycles)}
            if reference[0]['latency_ms']>bound[0]['sla_ms']:
                with self.assertRaises(ForecastRejected):
                    run_matched(work,model,{'test':1},7,asynchronous=True,_guard_bounds=bound)
            else:
                actual,_,_=run_matched(work,model,{'test':1},7,asynchronous=True,_guard_bounds=bound)
                self.assertEqual(actual,reference)


if __name__=='__main__':unittest.main()
