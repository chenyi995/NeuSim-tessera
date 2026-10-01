"""Original coarse Planaria allocation followed by optional fine admission.

The execution engine, memory lifetimes and event boundaries are unchanged.
At the reference grain, this is exactly the original allocation call, including
its RNG consumption. Fine refinement preserves the coarse admitted set under
the native remaining-time estimator; measured online QPS is checked separately.
"""
from collections import OrderedDict


class CoarseView:
    def __init__(self, task, quantum):
        self.task = task
        self.quantum = quantum

    def __getattr__(self, name):
        return getattr(self.task, name)

    def get_remaining_estimated_time(self, cores):
        return self.task.get_remaining_estimated_time(cores * self.quantum)


# chenyi9: decision start — the new algorithm must degenerate to original Planaria-32.
class CoarseFirstAllocator:
    def __init__(self, grain, refinement=True):
        # Source: chenyi9's comparison uses Planaria-32 as the coarse reference.
        reference_grain = 32
        assert grain > 0 and (grain >= reference_grain or reference_grain % grain == 0)
        self.quantum = max(1, reference_grain // grain) ** 2
        self.refinement = refinement
        self.audit = dict(calls=0,reference_identity_calls=0,coarse_retained=0,
                          refined_calls=0,additional_predicted_admissions=0,
                          protected_predicted_admissions=0)

    def __call__(self, queue, possible, units):
        from neusim.run_scripts.run_feature_qos_scheduler import POLICY, FREQ
        self.audit['calls'] += 1
        quantum = self.quantum
        assert units % quantum == 0
        if quantum == 1:
            views, coarse_possible = queue, possible
        else:
            views = OrderedDict((key, CoarseView(task, quantum)) for key, task in queue.items())
            coarse_possible = {key:[c // quantum for c in possible[key] if c % quantum == 0]
                               for key in queue}
        coarse_units = units // quantum
        if POLICY.check_if_tasks_all_fit(coarse_possible, coarse_units):
            assigned = POLICY.assign_cores_if_tasks_fit(views, coarse_possible, coarse_units)
        else:
            assigned = POLICY.assign_cores_if_tasks_not_fit(views, coarse_possible, coarse_units, FREQ)
        coarse = {key:value * quantum for key,value in assigned.items()}
        if quantum == 1 or not self.refinement:
            self.audit['reference_identity_calls' if quantum == 1 else 'coarse_retained'] += 1
            return coarse

        protected = {key for key in queue if coarse[key] in possible[key]}
        candidate = dict(coarse)
        for key in protected:
            candidate[key] = min(c for c in possible[key] if c <= coarse[key])
        spare = units - sum(candidate.values())

        # Keep the original *coarse* overload score. Recomputing this score in
        # fine-core units was the source of granularity-dependent admission order.
        def score(key):
            task = views[key]
            if coarse_possible[key]:
                slack = task.sla * 1.e-3 * FREQ - (task.current_time - task.start_time)
                return task.priority / slack / coarse_possible[key][-1]
            duration = task.get_remaining_estimated_time(coarse_units)
            return -task.priority / (duration if duration > 0 else 1)

        added = []
        for key in sorted((k for k in queue if k not in protected), key=score, reverse=True):
            if not possible[key]:
                continue
            # Preserve any original positive allocation until this task can
            # actually gain a predicted SLA-feasible share.
            choices = [c for c in possible[key] if candidate[key] <= c <= candidate[key] + spare]
            if choices:
                target = min(choices)
                spare -= target - candidate[key]
                candidate[key] = target
                added.append(key)
        if not added:
            self.audit['coarse_retained'] += 1
            return coarse

        # Return unused capacity to the original allocations, retaining native
        # feasibility even when a profile's time curve is nonmonotone.
        for key in assigned:
            if key not in protected:
                continue
            choices = [c for c in possible[key]
                       if candidate[key] <= c <= min(coarse[key], candidate[key] + spare)]
            target = max(choices)
            spare -= target - candidate[key]
            candidate[key] = target
        assert all(candidate[key] in possible[key] for key in protected | set(added))
        assert all(candidate[key] == coarse[key] for key in queue if key not in protected and key not in added)
        assert 0 < sum(candidate.values()) <= units
        self.audit['refined_calls'] += 1
        self.audit['additional_predicted_admissions'] += len(added)
        self.audit['protected_predicted_admissions'] += len(protected)
        return candidate
# chenyi9: decision end
