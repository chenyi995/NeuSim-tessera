"""Common conservative reservation scheduler with an explicit coarse fallback.

chenyi9 approved: retain the coarse complete schedule and admit refinements only
after checking array placement, finite SRAM and HBM/VU service reservations.
Source: Feitelson/Mu'alem, Utilization and Predictability in Scheduling, §2.1.
This adaptation uses native NeuSim profiles and physical FCR lifetimes. It is
an online constructive search, not a claim of globally optimal scheduling.
"""
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass
import math


# chenyi9: decision start — preserve concrete coarse reservations on both designs.
class Calendar:
    """Disjoint array/service intervals plus finite request workspace reservations."""
    def __init__(self, model):
        self.model = model
        self.cells = [[] for _ in range(model.units)]
        self.hbm = []
        self.vu = []
        self.memory = []

    def clone(self, exclude=None):
        other = Calendar(self.model)
        def keep(seq):
            return list(seq) if exclude is None else [x for x in seq if x[-1] != exclude]
        other.cells = [keep(seq) for seq in self.cells]
        other.hbm, other.vu, other.memory = keep(self.hbm), keep(self.vu), keep(self.memory)
        return other

    @staticmethod
    def conflicts(seq, start, end):
        index = max(0, bisect_right(seq, (start, math.inf, math.inf)) - 1)
        answer = []
        while index < len(seq) and seq[index][0] < end:
            if seq[index][1] > start:
                answer.append(seq[index])
            index += 1
        return answer

    @staticmethod
    def service(seq, start, duration, owner):
        if not duration:
            return start
        while True:
            blocked = Calendar.conflicts(seq, start, start + duration)
            if not blocked:
                insort(seq, (start, start + duration, owner))
                return start + duration
            start = blocked[0][1]

    def memory_blocker(self, start, end, amount):
        """Return an existing release to pass if the proposed interval overflows."""
        capacity = self.model.chip['capacity_bytes']
        if amount > capacity:
            raise ValueError('A mapping exceeds physical SRAM capacity')
        active = {}; events = []
        for index, (a, b, size, owner) in enumerate(self.memory):
            if b <= start or a >= end:
                continue
            if a <= start:
                active[index] = (b, size)
            else:
                events.append((a, 1, index, b, size))
            if b < end:
                events.append((b, 0, index, b, size))
        used = sum(x[1] for x in active.values())
        if used + amount > capacity:
            return min(x[0] for x in active.values())
        for _, begin, index, finish, size in sorted(events):
            if begin:
                active[index] = (finish, size); used += size
            elif index in active:
                used -= active.pop(index)[1]
            if used + amount > capacity:
                return min(x[0] for x in active.values())
        return None

    def lanes(self, start, durations, height, width, need, owner):
        """First-fit complete nonpreemptible regions, including future bookings."""
        from itertools import groupby
        from neusim.run_scripts.run_feature_qos_scheduler import available_boxes
        tessera = self.model.arch == 'Tessera-8'
        side = self.model.chip['side']; grain = self.model.chip['grain']
        columns = side // grain
        if tessera and (height > side or width > side):
            raise ValueError('Tessera cannot place an extra-long composition')
        while True:
            occupied = set(); result = []; releases = []
            # Equal-duration lanes have identical forbidden cells. Batch their
            # first-fit query without changing the sequential placement order.
            for duration, same in groupby(durations):
                count = sum(1 for _ in same)
                unavailable = set(occupied)
                for index, seq in enumerate(self.cells):
                    overlap = self.conflicts(seq, start, start + duration)
                    if overlap:
                        unavailable.add(index); releases.extend(x[1] for x in overlap)
                selections = []
                if tessera:
                    rows = [0]*columns
                    for index in unavailable:
                        rows[index//columns] |= 1 << (index%columns)
                    boxes = available_boxes(tuple(rows),side,grain,height,width,count)
                    for row,col,h,w in boxes:
                        selections.append(tuple(y*columns+x for y in range(row//grain,(row+h)//grain)
                                                for x in range(col//grain,(col+w)//grain)))
                else:
                    free = [x for x in range(self.model.units) if x not in unavailable]
                    selections = [tuple(free[i*need:(i+1)*need]) for i in range(min(count,len(free)//need))]
                if len(selections) < count:
                    break
                for selected in selections:
                    occupied.update(selected); result.append((selected, duration))
            if len(result) == len(durations):
                for selected, duration in result:
                    for cell in selected:
                        insort(self.cells[cell], (start, start + duration, owner))
                return start, result
            if not releases:
                raise ValueError('Requested lane bundle cannot fit an empty array')
            start = min(releases)


@dataclass
class Plan:
    owner: int
    network: str
    arrival: float
    finish: float
    share: int
    sla_ms: float
    priority: int
    macs: int
    groups: int
    pe_cycles: float
    workspace: int
    hbm_bytes: int


def workspace(model, name, share):
    peak = 0
    for p in model.profiles[name][share]:
        if p.kind == 'vector':
            peak = max(peak, p.vector_peak)
        else:
            _, m, n, k = p.shape
            mt, nt, kt = (min(a, b) for a, b in zip((m, n, k), p.tile))
            peak = max(peak, 4*(mt*kt+kt*nt) + 4*(p.geometry[3]+1)*mt*nt)
    return peak


def place_request(base, owner, name, arrival, priority, sla_ms, share):
    """Insert one complete request without moving any existing reservation.

    SRAM reservation is the maximum live tile workspace for this request,
    held from its first group until completion. This conservative allocation
    preserves all partial-C/ping-pong storage. SRAM bandwidth is still ideal.
    Array groups and HBM/VU use the same component-overlap algebra as the
    matched-tile engine; arbitration now honors immutable reservations.
    """
    from neusim.run_scripts.run_feature_qos_scheduler import FREQ
    from neusim.npusim.backend.tessera_partitioned import lane_cycles
    from neusim.npusim.backend.tessera import ceildiv
    model = base.model; chip = model.chip; peak = workspace(model, name, share)
    earliest = arrival
    lower = model.suffix[name][share][0]
    while True:
        blocker = base.memory_blocker(earliest, earliest + lower, peak)
        if blocker is not None:
            earliest = blocker; continue
        calendar = base.clone(); now = earliest; first_start = None
        macs = groups = traffic = 0; pe_cycles = 0.
        for p in model.profiles[name][share]:
            origin = None
            if p.kind == 'vector':
                duration, hb, vu, _ = model.vector_cost(p, share)
                first_start = now if first_start is None else first_start
                hb_end = calendar.service(calendar.hbm, now, hb/chip['hbm_bytes_per_cycle'], owner)
                vu_end = calendar.service(calendar.vu, now, vu, owner)
                now = max(now + duration, hb_end, vu_end, now + chip['hbm_latency_cycles'])
                traffic += hb; groups += 1
                continue
            batch, m, n, k = p.shape; mt, nt, kt = p.tile
            g = p.geometry; gh, gw = g[-2:]; need = gh*gw//chip['grain']**2
            for b in range(batch):
                for m0 in range(0, m, mt):
                    h = min(mt, m-m0)
                    for n0 in range(0, n, nt):
                        w = min(nt, n-n0)
                        for k0 in range(0, k, kt):
                            kk = min(kt, k-k0)
                            v = model.group(p, h, w, kk, k0 == 0, k0+kk == k, share)
                            if v is None or v[4] > chip['capacity_bytes']:
                                raise ValueError('Infeasible reserved native group')
                            duration, sa, hb, vu, _, width, weights = v
                            durations = []
                            for lane in range(width):
                                folds = ceildiv(weights-lane, width)
                                cyc = (folds*(max(h,32) if folds >= 3 else h)+32+gh+gw-2
                                       if model.arch == 'Planaria-32' else lane_cycles(h,g,'full',chip['grain'],folds))
                                durations.append(cyc*FREQ/chip['frequency_Hz'])
                            start, lanes = calendar.lanes(now, durations, gh, gw, need, owner)
                            if origin is None:
                                origin = start
                            if first_start is None:
                                first_start = start
                            hb_end = calendar.service(calendar.hbm, start, hb/chip['hbm_bytes_per_cycle'], owner)
                            vu_end = calendar.service(calendar.vu, start, vu, owner)
                            now = max(start+duration, hb_end, vu_end)
                            pe_cycles += sum(len(cells)*chip['grain']**2*d for cells,d in lanes)
                            macs += h*w*kk; traffic += hb; groups += 1
            if not p.resident:
                now = max(now, origin+chip['hbm_latency_cycles'])
        first_start = earliest if first_start is None else first_start
        blocker = base.memory_blocker(first_start, now, peak)
        if blocker is not None:
            earliest = blocker; continue
        insort(calendar.memory, (first_start, now, peak, owner))
        plan = Plan(owner,name,arrival,now,share,sla_ms,priority,macs,groups,pe_cycles,peak,traffic)
        return plan, calendar


def coarse_plan(calendar, tasks, coarse_step, counters):
    """Earliest-deadline insertion; retain a concrete coarse feasible incumbent."""
    from neusim.run_scripts.run_feature_qos_scheduler import FREQ
    result = {}
    for owner,name,arrival,priority,sla in sorted(tasks,key=lambda t:(t[2]+t[4]*FREQ/1e3,-t[3],t[0])):
        deadline = arrival + sla*FREQ/1e3; best = None
        for share in range(coarse_step, calendar.model.units+1, coarse_step):
            lower = arrival + calendar.model.suffix[name][share][0]
            # No calendar insertion can beat its isolated lower bound.
            if lower > deadline and share != calendar.model.units:
                continue
            candidate, trial = place_request(calendar,owner,name,arrival,priority,sla,share)
            counters['candidate_evaluations'] += 1
            if best is None or candidate.finish < best[0].finish:
                best = candidate, trial
            if candidate.finish <= deadline:
                best = candidate, trial; break
        assert best is not None
        plan, calendar = best; result[owner] = plan
    return result, calendar


def refine(calendar, plans, steps, counters):
    """Try finer minimum-deadline shares while retaining every coarse admission.

    Previously committed requests remain immutable. For the current arrival
    batch, a coarse on-time request must remain on time; a coarse late request
    may only finish earlier. Candidates are deterministic minima at each legal
    dyadic allocation quantum, plus the retained coarse mapping. This is a
    bounded constructive heuristic, not an exhaustive global optimizer.
    """
    from neusim.run_scripts.run_feature_qos_scheduler import FREQ
    for owner, original in list(plans.items()):
        deadline = original.arrival + original.sla_ms*FREQ/1e3
        limit = deadline if original.finish <= deadline else original.finish
        model = calendar.model; candidates = {original.share}
        for step in steps:
            feasible = [c for c in range(step,model.units+1,step)
                        if original.arrival+model.suffix[original.network][c][0] <= limit]
            if feasible:
                candidates.add(min(feasible))
        empty = calendar.clone(exclude=owner)
        best = original; best_calendar = calendar
        on_time = original.finish <= deadline
        def key(plan):
            return (plan.pe_cycles,plan.finish) if on_time else (plan.finish,plan.pe_cycles)
        for share in sorted(candidates):
            # Reinsert the same share too: earlier refinements can expose holes.
            candidate, trial = place_request(empty,owner,original.network,original.arrival,
                                              original.priority,original.sla_ms,share)
            counters['candidate_evaluations'] += 1
            if candidate.finish <= limit and key(candidate) < key(best):
                best, best_calendar = candidate, trial
        if best is not original:
            counters['accepted_refinements'] += 1
            if best.share < original.share:
                counters['smaller_share_refinements'] += 1
        else:
            counters['coarse_or_previous_retained'] += 1
        plans[owner] = best; calendar = best_calendar
    return plans, calendar


def verify_calendar(calendar):
    """Independent capacity readback over the final resource reservations."""
    for seq in (*calendar.cells,calendar.hbm,calendar.vu):
        assert all(a[1] <= b[0] for a,b in zip(seq,seq[1:]))
        assert all(a < b for a,b,_ in seq)
    events = []
    for a,b,amount,owner in calendar.memory:
        events.extend(((a,amount),(b,-amount)))
    used = peak = 0
    for _,delta in sorted(events):
        used += delta; peak = max(peak,used)
        assert 0 <= used <= calendar.model.chip['capacity_bytes']
    assert used == 0
    return dict(memory_peak_bytes=peak,array_reservations=sum(map(len,calendar.cells)),
                hbm_reservations=len(calendar.hbm),vu_reservations=len(calendar.vu))


def run_reserved(work, model, qos, seed, *, refinement=True, trace=None, audit=None, fail_budget=None):
    """Reserve only arrived requests; retain commitments across future arrivals."""
    from neusim.run_scripts.run_feature_qos_scheduler import FREQ, ProvenInfeasible
    from collections import Counter
    calendar = Calendar(model); done = []; plans = {}; index = batches = 0
    counters = dict(candidate_evaluations=0,accepted_refinements=0,smaller_share_refinements=0,
                    coarse_or_previous_retained=0,protected_requests_checked=0)
    # The paper's coarse reference is 32x32; physical grains remain 32 and 8.
    coarse_step = max(1,(32//model.chip['grain'])**2)
    steps = []; step = coarse_step
    while step >= 1:
        steps.append(step)
        if step == 1:break
        step //= 4
    while index < len(work):
        now = work[index][0]; tasks = []
        while index < len(work) and work[index][0] == now:
            arrival,name,priority = work[index]
            tasks.append((index,name,arrival,priority,qos[name])); index += 1
        prior = {key:plan.finish for key,plan in plans.items()}
        coarse, coarse_calendar = coarse_plan(calendar,tasks,coarse_step,counters)
        bounds = {key:(p.finish,p.arrival+p.sla_ms*FREQ/1e3) for key,p in coarse.items()}
        current = dict(coarse)
        if refinement:
            current, calendar = refine(coarse_calendar,current,steps,counters)
        else:
            calendar = coarse_calendar
        for key,plan in current.items():
            finish,deadline = bounds[key]
            assert plan.finish <= (deadline if finish <= deadline else finish)
            counters['protected_requests_checked'] += 1
        assert all(plans[key].finish == finish for key,finish in prior.items())
        plans.update(current); batches += 1
        if trace is not None:
            trace.append(dict(arrival_cycle=now,known_requests=index,coarse={k:vars(v) for k,v in coarse.items()},
                              selected={k:vars(v) for k,v in current.items()}))
        # A committed completion is immutable under subsequent arrivals. Its
        # miss is therefore already certain, even if the wall-clock deadline
        # has not passed. This proof never inspects unarrived requests.
        if fail_budget is not None:
            late = Counter('tr' if p.network == 'GNMT' else 'cd' for p in plans.values()
                           if (p.finish-p.arrival)/FREQ*1e3 > p.sla_ms)
            if any(late[group] > limit for group,limit in fail_budget.items()):
                if audit is not None:
                    audit.update(counters,batches=batches,coarse_step=coarse_step,refinement=refinement,
                                 proof='Immutable completion reservations already exceed the allowed miss count.')
                raise ProvenInfeasible([],batches,counters['candidate_evaluations'],late,now)
    for owner,plan in plans.items():
        expected = sum(math.prod(p.shape) for p in model.profiles[plan.network][plan.share] if p.kind == 'matrix')
        assert plan.macs == expected
        done.append(dict(task_id=owner,network=plan.network,start_cycle=plan.arrival,finish_cycle=plan.finish,
                         latency_ms=(plan.finish-plan.arrival)/FREQ*1e3,sla_ms=plan.sla_ms,
                         useful_macs=plan.macs,groups=plan.groups))
    readback = verify_calendar(calendar)
    if audit is not None:
        audit.update(counters,**readback,batches=batches,coarse_step=coarse_step,refinement=refinement)
    return sorted(done,key=lambda r:(r['finish_cycle'],r['task_id'])),batches,counters['candidate_evaluations']
# chenyi9: decision end
