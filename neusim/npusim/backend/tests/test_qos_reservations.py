"""Independent reservation goldens and checks of the retained coarse fallback."""
import json
import unittest
from neusim.run_scripts.run_feature_qos_scheduler import MatchedCosts,FREQ
from neusim.run_scripts.tessera_qos_reservations import Calendar,place_request,run_reserved,verify_calendar


class ReservationTests(unittest.TestCase):
    def model(self,arch,shape=(1,10,32,32),capacity=1<<20,hbm=0):
        grain=32 if arch=='Planaria-32' else 8
        units=(64//grain)**2
        row=dict(kind='matrix',key='x'.join(map(str,shape)),hbm_bytes=hbm,
                 vu_ns=0,reduction_ops=0,
                 mapping_json=json.dumps(dict(geometry=[grain,grain,False,1,grain,grain],memory_tile=list(shape[1:]))))
        return MatchedCosts({'test':{c:[row] for c in range(1,units+1)}},
            dict(side=64,grain=grain,capacity_bytes=capacity,frequency_Hz=FREQ,
                 hbm_bytes_per_cycle=16,hbm_latency_cycles=0),arch)

    def test_service_reservations_never_move_an_existing_interval(self):
        seq=[(10,20,0),(30,50,1)]
        self.assertEqual(Calendar.service(seq,0,7,2),7)
        self.assertEqual(Calendar.service(seq,5,4,3),24)
        self.assertEqual(Calendar.service(seq,8,9,4),59)
        self.assertIn((10,20,0),seq);self.assertIn((30,50,1),seq)

    def test_memory_release_at_exact_boundary_and_future_conflict(self):
        cal=Calendar(self.model('Planaria-32',capacity=100))
        cal.memory=[(0,10,60,0),(5,20,40,1)]
        self.assertEqual(cal.memory_blocker(3,8,30),10)
        self.assertIsNone(cal.memory_blocker(10,30,60))
        self.assertIsNone(cal.memory_blocker(20,30,100))

    def test_spatial_future_reservation_blocks_overlapping_fcr(self):
        model=self.model('Tessera-8');cal=Calendar(model)
        # A full array is reserved in a known future interval. An early region
        # that would cross its start must wait until the full-array release.
        for seq in cal.cells:seq.append((20,40,0))
        start,regions=cal.lanes(0,[30],8,8,1,1)
        self.assertEqual(start,40);self.assertEqual(regions[0][0],(0,))
        verify_calendar(cal)

    def test_single_native_group_hand_timing(self):
        for arch in ('Planaria-32','Tessera-8'):
            model=self.model(arch,shape=(1,10,32 if arch=='Planaria-32' else 8,32 if arch=='Planaria-32' else 8))
            plan,calendar=place_request(Calendar(model),0,'test',100,1,1,1)
            # Planaria: M+32+H+W-2; Tessera: load H + M + H-1 + W.
            self.assertEqual(plan.finish,100+(104 if arch=='Planaria-32' else 33))
            self.assertEqual(plan.macs,10*(32 if arch=='Planaria-32' else 8)**2)
            verify_calendar(calendar)

    def test_unseen_arrivals_cannot_change_committed_requests(self):
        for arch in ('Planaria-32','Tessera-8'):
            model=self.model(arch)
            before,_,_=run_reserved([(0,'test',1)],model,{'test':1},7)
            after,_,_=run_reserved([(0,'test',1),(1,'test',2)],model,{'test':1},7)
            self.assertEqual(before[0],next(r for r in after if r['task_id']==0))
            self.assertGreaterEqual(next(r for r in after if r['task_id']==1)['start_cycle'],1)

    def test_fine_search_preserves_each_coarse_deadline(self):
        for arch in ('Planaria-32','Tessera-8'):
            model=self.model(arch,shape=(2,10,32,32),hbm=1)
            for seed in (7,19,41):
                work=[(0,'test',i+1) for i in range(3)]+[(10000,'test',1)]
                coarse,_,_=run_reserved(work,model,{'test':.002},seed,refinement=False)
                trace=[];audit={}
                fine,_,_=run_reserved(work,model,{'test':.002},seed,trace=trace,audit=audit)
                for batch in trace:
                    for key,old in batch['coarse'].items():
                        selected=batch['selected'][key]
                        limit=old['arrival']+old['sla_ms']*FREQ/1e3
                        self.assertLessEqual(selected['finish'],limit if old['finish']<=limit else old['finish'])
                self.assertEqual(len(coarse),len(fine));self.assertEqual(audit['protected_requests_checked'],len(work))

    def test_failed_refinement_keeps_the_actual_coarse_plan(self):
        from unittest.mock import patch
        import neusim.run_scripts.tessera_qos_reservations as r
        model=self.model('Tessera-8');cal=Calendar(model)
        original,cal=place_request(cal,0,'test',0,1,1,16)
        original.sla_ms=original.finish/FREQ*1e3
        real=r.place_request
        def late(*args,**kwargs):
            plan,newcal=real(*args,**kwargs)
            plan.finish=original.finish+1
            return plan,newcal
        counters=dict(candidate_evaluations=0,accepted_refinements=0,smaller_share_refinements=0,coarse_or_previous_retained=0)
        with patch.object(r,'place_request',side_effect=late):
            selected,after=r.refine(cal,{0:original},[16,4,1],counters)
        self.assertIs(selected[0],original);self.assertIs(after,cal)
        self.assertEqual(counters['accepted_refinements'],0)

    def test_planaria_extra_long_compositions_remain_legal(self):
        from dataclasses import replace
        model=self.model('Planaria-32',shape=(1,10,128,32))
        original=model.profiles['test'][model.units][0]
        model.profiles['test'][model.units]=(replace(original,geometry=(32,128,True,1,32,128)),)
        plan,cal=place_request(Calendar(model),0,'test',0,1,1,model.units)
        self.assertEqual(plan.finish,10+32+32+128-2)
        verify_calendar(cal)


if __name__=='__main__':unittest.main()
