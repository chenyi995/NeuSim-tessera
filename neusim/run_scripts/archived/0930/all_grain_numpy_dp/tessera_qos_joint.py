"""One joint resource selector extending Planaria's urgency objective.

Planaria scheduler.py::assign_cores_if_tasks_not_fit ranks by
priority / slack / minimum cores, a greedy urgency-per-resource score.
This selector instead maximizes total admitted urgency over the complete
resource allocation using multiple-choice dynamic programming. Among equal
urgency solutions it maximizes (priority / slack) * log(1 + slack / remaining time).
This concave progress objective prevents linear service rate from assigning
every spare unit to a fast request while retaining current deadline urgency.
At zero/negative slack its continuous limit is priority / remaining time.
Slack is sourced from the existing workload, not a fitted coefficient.
The original policy supplies the urgency/priority
basis; the joint objective and dynamic program are the new algorithm.
There is no architecture branch, coarse reference, incumbent or fallback.
"""
import math
import numpy as np


# chenyi9: decision start — replace greedy allocation with one joint selector at every grain.
class JointSLAAllocator:
    # The executor must use this selector for admission as well as allocation.
    # In particular, zero means wait; it must not trigger greedy borrowing.
    integrated_dispatch = True
    def __init__(self, trace=None):
        self.trace = trace
        self.audit = dict(calls=0,candidate_options=0,frontier_options=0,
                          selected_predicted_admissions=0,unassigned_requests=0,
                          epoch_candidates=0,deferred_decisions=0)

    @staticmethod
    def utilities(task, times, frequency):
        slack = task.sla * 1.e-3 * frequency - (task.current_time - task.start_time)
        finite = np.isfinite(times)
        # Live requests have positive remaining service. The native engine
        # retires completed requests before invoking resource selection.
        assert np.all(times[finite] > 0), 'A completed request reached allocation'
        urgent = task.priority / slack if slack > 0 else 0.
        admitted = finite & (times <= slack)
        primary = np.where(admitted, urgent, 0.)
        # /root: decision start — one deadline-scaled diminishing-return objective.
        # The failed linear-rate experiment is retained at
        # results/tessera/20260930_qos_joint_v1/diagnostic/. The normalization is
        # remaining deadline in simulation cycles, common to all grains. The
        # zero-slack branch is the analytic limit, not a second allocator.
        progress = np.zeros_like(times)
        if slack > 0:
            np.divide(slack, times, out=progress, where=finite)
            progress = urgent * np.log1p(progress)
        else:
            np.divide(task.priority, times, out=progress, where=finite)
        return primary,progress,admitted
        # /root: decision end

    @staticmethod
    def after_wait(task,c,wait_cycles):
        if not wait_cycles:return task.get_remaining_estimated_time(c)
        # Shift only the prediction clock. Native in-flight and layer latency
        # floors then overlap waiting correctly; no future requests are read.
        before=task.current_time
        try:
            task.current_time=before+wait_cycles
            return wait_cycles+task.get_remaining_estimated_time(c)
        finally:task.current_time=before

    def select_epoch(self,queue,possible,options):
        """Jointly choose a known resource-release epoch and resource shares.

        Each option is (wait cycles, capacity, placement predicate). Native
        placement predicates are monotone: a larger share retains every
        smaller share's legal tile through the prefix-of-profiles search.
        All use
        the same objective and DP. Waiting is an action in that optimization,
        not a comparison with another policy or a coarse reference schedule.
        """
        best=None
        for option in sorted(options,key=lambda option:option[0]):
            wait,capacity,predicate,*constraints=option
            minimum,fixed=constraints if constraints else ({},{})
            allocation=self(queue,possible,capacity,predicate,wait_cycles=wait,monotone_executable=True,
                            minimum_units=minimum,fixed_times=fixed)
            self.audit['epoch_candidates']+=1
            if allocation is None:continue
            if best is None or self.last_objective>best[0]:
                best=self.last_objective,allocation,wait,capacity
        assert best is not None
        _,allocation,wait,capacity=best
        self.audit['deferred_decisions']+=int(wait>0)
        return allocation,wait,capacity

    def __call__(self, queue, possible, units, executable=None,wait_cycles=0,monotone_executable=False,
                 minimum_units=None,fixed_times=None):
        from neusim.run_scripts.run_feature_qos_scheduler import FREQ
        assert units >= 0 and wait_cycles >= 0
        keys=list(queue);budget=np.arange(units+1)
        primary=np.full(units+1,-np.inf);rate=primary.copy()
        primary[0]=rate[0]=0.
        parents=[];details=[]
        self.audit['calls']+=1
        for key in keys:
            task=queue[key]
            minimum=(minimum_units or {}).get(key,0)
            fixed=(fixed_times or {}).get(key)
            times=(np.full(units+1,fixed) if fixed is not None else
                   np.array([math.inf]+[self.after_wait(task,c,wait_cycles) for c in range(1,units+1)]))
            if executable is not None:
                if monotone_executable:
                    lo,hi=0,units+1
                    while hi-lo>1:
                        mid=(lo+hi)//2
                        if executable(key,mid):hi=mid
                        else:lo=mid
                    allowed=budget>=hi
                else:allowed=np.array([False]+[executable(key,c) for c in range(1,units+1)])
            else:allowed=np.ones(units+1,dtype=bool)
            allowed &= budget>=minimum
            if fixed is not None and minimum==0:allowed[0]=True
            times=np.where(allowed,times,math.inf)
            reward,service,admitted=self.utilities(task,times,FREQ)
            native={c for c in possible[key] if c<=units and allowed[c]}
            positive=set(np.flatnonzero(admitted))-{0}
            assert (positive<=native if wait_cycles else positive==native)
            # Exact Pareto pruning: a smaller allocation with no worse reward
            # in either objective dominates a larger choice. Time curves need
            # not be monotone, and zero allocation remains an explicit option.
            choices=[0] if minimum==0 else []
            best_any=float(service[0]) if minimum==0 else 0.
            best_admitted=float(service[0]) if minimum==0 and admitted[0] else 0.
            for c in range(max(1,minimum),units+1):
                if not math.isfinite(times[c]):continue
                dominated=(service[c]<=best_admitted if admitted[c] else service[c]<=best_any)
                if not dominated:choices.append(c)
                best_any=max(best_any,service[c])
                if admitted[c]:best_admitted=max(best_admitted,service[c])
            choices=np.array(choices,dtype=np.int64)
            if not len(choices):
                self.last_objective=(-math.inf,-math.inf)
                return None
            selected=np.zeros(units+1,dtype=np.int64)
            if not parents:
                # Exact first DP row: only the zero-budget predecessor exists.
                primary=np.full(units+1,-np.inf);rate=primary.copy()
                primary[choices]=reward[choices];rate[choices]=service[choices]
                selected[choices]=choices
            else:
                next_primary=np.full(units+1,-np.inf);next_rate=next_primary.copy()
                # Host-only batching of the identical DP recurrence, bounded
                # like the existing native cost cache (262144 entries). It is
                # not a resource/mapping threshold and cannot change choices.
                block=max(1,262144//(units+1))
                for offset in range(0,len(choices),block):
                    cc=choices[offset:offset+block]
                    previous=budget[None,:]-cc[:,None]
                    valid=previous>=0;indices=np.maximum(previous,0)
                    p=np.where(valid,primary[indices]+reward[cc,None],-np.inf)
                    s=np.where(valid,rate[indices]+service[cc,None],-np.inf)
                    maximum=p.max(axis=0)
                    pick=np.argmax(np.where(p==maximum,s,-np.inf),axis=0)
                    pp,ss=p[pick,budget],s[pick,budget]
                    better=(pp>next_primary)|((pp==next_primary)&(ss>next_rate))
                    next_primary[better]=pp[better];next_rate[better]=ss[better]
                    selected[better]=cc[pick[better]]
                primary,rate=next_primary,next_rate
            parents.append(selected)
            details.append((times,admitted))
            self.audit['candidate_options']+=int(np.count_nonzero(np.isfinite(times)))+1
            self.audit['frontier_options']+=len(choices)
        best_primary=primary.max()
        if not math.isfinite(best_primary):
            self.last_objective=(-math.inf,-math.inf)
            return None
        finalists=np.where(primary==best_primary,rate,-np.inf)
        used=int(np.argmax(finalists));objective=(float(primary[used]),float(rate[used]))
        self.last_objective=objective
        allocation={}
        for index in reversed(range(len(keys))):
            chosen=int(parents[index][used]);allocation[keys[index]]=chosen;used-=chosen
        assert used==0 and 0<=sum(allocation.values())<=units
        allocation={key:allocation[key] for key in keys}
        good=sum(bool(details[i][1][allocation[key]]) for i,key in enumerate(keys))
        self.audit['selected_predicted_admissions']+=good
        self.audit['unassigned_requests']+=sum(c==0 for c in allocation.values())
        if self.trace is not None:
            self.trace.append(dict(units=units,allocation=allocation,objective=objective,
                                   predicted_admissions=good,wait_cycles=wait_cycles))
        return allocation
# chenyi9: decision end
