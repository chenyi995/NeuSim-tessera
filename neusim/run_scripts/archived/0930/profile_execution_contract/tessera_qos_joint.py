"""Planaria-derived deadline allocation over measured resource-share curves.

The same procedure handles every fission grain. Planaria's minimum feasible
allocation and priority/slack ordering are retained. Spare resources advance
only to a faster legal operating point, ranked by marginal priority-weighted
service rate per fraction of the physical array. No reference allocator,
architecture branch, alternate schedule or hindsight is used.
Source: planaria.code/scheduler/scheduler.py, assign_cores_if_tasks_{not_,}fit.
"""
import math
import numpy as np


# chenyi9: decision start -- one Planaria-derived allocator across all fission grains.
class JointSLAAllocator:
    integrated_dispatch = True

    def __init__(self, trace=None):
        self.trace=trace
        self.audit=dict(calls=0,candidate_options=0,frontier_options=0,
                        selected_predicted_admissions=0,unassigned_requests=0,
                        epoch_candidates=0,deferred_decisions=0,marginal_upgrades=0)

    @staticmethod
    def after_wait(task,c,wait_cycles):
        estimate=getattr(task,'get_shared_resource_estimated_time',task.get_remaining_estimated_time)
        if not wait_cycles:return estimate(c)
        before=task.current_time
        try:
            task.current_time=before+wait_cycles
            return wait_cycles+estimate(c)
        finally:task.current_time=before

    def select_epoch(self,queue,possible,options):
        best=None
        for wait,capacity,predicate,*constraints in sorted(options,key=lambda x:x[0]):
            minimum,fixed=constraints if constraints else ({},{})
            allocation=self(queue,possible,capacity,predicate,wait_cycles=wait,
                            monotone_executable=True,minimum_units=minimum,fixed_times=fixed)
            self.audit['epoch_candidates']+=1
            if allocation is not None and (best is None or self.last_objective>best[0]):
                best=self.last_objective,allocation,wait,capacity
        assert best is not None
        _,allocation,wait,capacity=best
        self.audit['deferred_decisions']+=int(wait>0)
        return allocation,wait,capacity

    def __call__(self,queue,possible,units,executable=None,wait_cycles=0,
                 monotone_executable=False,minimum_units=None,fixed_times=None):
        from neusim.run_scripts.run_feature_qos_scheduler import FREQ
        assert units>=0 and wait_cycles>=0
        keys=list(queue);index={k:i for i,k in enumerate(keys)}
        minimum_units=minimum_units or {};fixed_times=fixed_times or {}
        allocation={k:minimum_units.get(k,0) for k in keys}
        if sum(allocation.values())>units:
            self.last_objective=(-math.inf,-math.inf);return None
        curves={};frontiers={};needs={};slacks={};ranks={}
        self.audit['calls']+=1
        budget=np.arange(units+1)
        for key,task in queue.items():
            minimum=allocation[key];fixed=fixed_times.get(key)
            times=(np.full(units+1,fixed) if fixed is not None else
                   task.get_shared_resource_estimated_times(units,wait_cycles)
                   if hasattr(task,'get_shared_resource_estimated_times') else
                   np.array([math.inf]+[self.after_wait(task,c,wait_cycles) for c in range(1,units+1)]))
            if executable is None:allowed=budget>=minimum
            elif monotone_executable:
                lo,hi=0,units+1
                while hi-lo>1:
                    mid=(lo+hi)//2
                    if executable(key,mid):hi=mid
                    else:lo=mid
                allowed=budget>=max(hi,minimum)
            else:allowed=np.array([False]+[executable(key,c) and c>=minimum for c in range(1,units+1)])
            if fixed is not None and minimum==0:allowed[0]=True
            times=np.where(allowed,times,math.inf)
            curves[key]=times
            slack=task.sla*1.e-3*FREQ-(task.current_time-task.start_time)
            slacks[key]=slack
            # A resource-share curve may be nonmonotone because its SRAM and
            # geometry winners change. Extra units must not force a slower map.
            choices=[];best=times[minimum] if minimum else times[0]
            if minimum and not math.isfinite(best):
                self.last_objective=(-math.inf,-math.inf);return None
            if minimum and math.isfinite(best):choices.append(minimum)
            for c in range(max(1,minimum+1),units+1):
                if times[c]<best:choices.append(c);best=times[c]
            frontiers[key]=choices
            feasible=[c for c in choices if times[c]<=slack]
            if times[minimum]<=slack:feasible.insert(0,minimum)
            if feasible:
                c=feasible[0];needs[key]=c
                # c/C is physical area fraction. The ordering reduces to the
                # original Planaria priority/slack/minimum-core rule at d=32.
                ranks[key]=math.inf if c==0 else task.priority/slack/(c/units)
            else:needs[key]=None;ranks[key]=-math.inf
            self.audit['candidate_options']+=int(np.count_nonzero(np.isfinite(times)))
            self.audit['frontier_options']+=len(choices)
        spare=units-sum(allocation.values())
        # Planaria's minimum-deadline admission pass. Stable request order
        # breaks equal scores; no random resource remainder is introduced.
        for key in sorted(keys,key=lambda k:(-ranks[k],index[k])):
            desired=needs[key]
            if desired is None:continue
            extra=desired-allocation[key]
            if extra<=spare:
                allocation[key]=desired;spare-=extra
        # /root: use measured marginal progress for remaining resource shares.
        # p/T is Planaria's spare-allocation service score. Its finite
        # difference per added PE fraction avoids giving a request resources
        # on a latency plateau. All larger efficient points are considered,
        # allowing useful upgrades across gaps in legal geometry sizes.
        while spare:
            best=None
            for key in keys:
                old=allocation[key];t=curves[key][old]
                rate=queue[key].priority/t if math.isfinite(t) else 0.
                for c in frontiers[key]:
                    extra=c-old
                    if extra<=0 or extra>spare:continue
                    gain=queue[key].priority/curves[key][c]-rate
                    if gain<=0:continue
                    score=gain/(extra/units)
                    candidate=(score,-extra,-index[key],key,c)
                    if best is None or candidate[:3]>best[:3]:best=candidate
            if best is None:break
            *_,key,c=best;spare-=c-allocation[key];allocation[key]=c
            self.audit['marginal_upgrades']+=1
        admitted=0;urgency=progress=0.
        for key,c in allocation.items():
            t=curves[key][c];slack=slacks[key]
            if math.isfinite(t):
                assert t>0
                progress+=queue[key].priority/t
                if t<=slack:
                    urgency+=queue[key].priority/slack;admitted+=1
        self.last_objective=(urgency,progress)
        assert sum(allocation.values())<=units
        self.audit['selected_predicted_admissions']+=admitted
        self.audit['unassigned_requests']+=sum(c==0 for c in allocation.values())
        if self.trace is not None:self.trace.append(dict(units=units,allocation=allocation,
            objective=self.last_objective,predicted_admissions=admitted,wait_cycles=wait_cycles))
        return allocation
# chenyi9: decision end
