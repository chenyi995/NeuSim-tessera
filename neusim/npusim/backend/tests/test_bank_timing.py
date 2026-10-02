"""Independent live-bank schedules for continuous WS/Planaria folds."""
import json
import math
from pathlib import Path
import random

import numpy as np
import pytest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend import tessera_baselines as base, tessera_joint as joint
from neusim.npusim.backend import tessera_native_tail as tail, tessera_partitioned as part
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
from neusim.run_scripts.run_tessera_partitioned import native_record

ROOT=Path(__file__).resolve().parents[4]


def events(m,h,w,folds,load):
    """Schedule actual resources, without the production closed form."""
    load_port=stream_port=0
    bank_available=[0,0]
    timeline=[]
    for f in range(folds):
        bank=f%2
        start=max(load_port,bank_available[bank])
        load_port=start+load
        stream=max(load_port,stream_port)
        stream_port=stream+m
        bank_available[bank]=stream_port+w-1
        end=stream_port+h+w-2
        timeline.append((start,stream,end,bank_available[bank]))
    return timeline


@pytest.mark.parametrize('seed',[7,29,101])
def test_closed_form_and_vector_against_event_resources(seed):
    rng=random.Random(seed)
    for _ in range(200):
        grain=rng.choice((8,16,32,64))
        h=grain*2**rng.randrange(5);w=grain*2**rng.randrange(5)
        m=rng.choice((1,grain-1,grain,grain+1,w,grain+w-1,grain+w+1))
        folds=rng.randrange(1,100)
        gold=events(m,h,w,folds,grain)
        assert base.ws_lane_cycles(m,h,w,folds,grain)==gold[-1][2]
        for i,(load,stream,end,free) in enumerate(gold):
            assert stream>=load+grain
            if i:
                assert load>=gold[i-1][0]+grain and stream>=gold[i-1][1]+m
            if i>=2:assert load>=gold[i-2][3]
        # One occupied geometry per physical fabric, with exactly folds weight tiles.
        if h*w<=128**2 and 128**2%(h*w)==0:
            slots=128**2//(h*w)
            g=(h,w,True,slots,h,w);n=w*slots*folds;k=h
            scalar=base.array_cost('Planaria',m,n,k,g,grain,True)
            vector=joint.array_vectors('Planaria',*[np.array([x]) for x in (m,n,k)],
                                       g,'full',grain,128,bank_timing=True)
            assert scalar[0]==gold[-1][2]==int(vector[0][0])
            assert tail.lane_time(m,g,folds,grain,'Planaria',True)==gold[-1][2]


def test_bank_reuse_regression_and_large_m_source_limit():
    # FissionSA README section 6.1's old stream schedule is unsafe below its
    # stated M >= d+W-1 condition. No synthetic performance target is asserted.
    d,h,w,m=8,128,128,1
    old_stream=d+2*max(m,d)
    assert old_stream-d < events(m,h,w,3,d)[0][3]
    assert base.ws_lane_cycles(m,h,w,3,d)>3*max(m,d)+d+h+w-2
    for d in (8,16,32,64):
        for h,w in ((d,d),(128,128),(d,128),(128,d)):
            for m in (d+w-1,2*(d+w)):
                for folds in (1,2,3,17):
                    assert base.ws_lane_cycles(m,h,w,folds,d)==folds*m+d+h+w-2
        for m in (1,d-1,d,d+1,3*d):
            for folds in (1,2,3,4,17):
                assert base.ws_lane_cycles(m,d,d,folds,d)==events(m,d,d,folds,d)[-1][2]


def test_tessera_seams_change_only_final_tail():
    g=(128,128,True,1,128,128)
    for m in (1,8,127,128,512):
        for folds in (1,2,3,17):
            for grain in (8,16,32,64):
                assert tail.lane_time(m,g,folds,grain,'Tessera',True)==part.lane_cycles(m,g,'full',grain,folds)
                assert (part.lane_cycles(m,g,'full',grain,folds+1)
                        -part.lane_cycles(m,g,'full',grain,folds))==max(m,128)


def test_native_energy_and_vector_use_identical_bank_timing():
    path=ROOT/'artifacts/tessera-20261001/ppa/joint_edp/main_costs/configs.json'
    for key,raw in json.loads(path.read_text()).items():
        raw['tessera_parameters'].update(sram_tiling_model='transfer_only',array_timing_model='bank_events_v1')
        chip=ChipConfig.model_validate(raw)
        arch=chip.tessera_parameters.get('baseline_architecture')
        if not arch:continue
        grain=chip.tessera_parameters['grain']
        for m,n,k in ((1,131,259),(17,193,257),(129,129,129)):
            gg=list(base.geometries(chip))
            for g in gg[::max(1,len(gg)//4)]:
                tiles=joint.factor_tiles(m,n,k,2,g[3],chip.vmem_size_MB*1024**2)
                v=joint.candidate_vectors(1,m,n,k,2,g,chip,tiles)
                _,times,energies=joint.native_objective(v,m*n*k,chip,2_800_000_000_000)
                for i in (0,len(tiles[0])-1):
                    tile=tuple(int(a[i]) for a in tiles[:3])
                    result=part.evaluate_candidate(1,m,n,k,'DT_BFLOAT16',g,chip,forced_tile=tile).stats
                    assert result.execution_time_ns==times[i],(key,g,tile)
                    assert math.isclose(result.total_energy_J,energies[i],rel_tol=2e-12)


def test_fixed_ws_compute_equal_under_both_tiling_policies():
    for arch in ('WS','WS-independent-8','WS-independent-64'):
        for m,n,k in ((1,4096,4096),(512,4096,4096)):
            result=[]
            for mode in ('native_tail','joint_edp'):
                path=ROOT/'artifacts/tessera-20261001/ppa'/mode/'main_costs/configs.json'
                raw=json.loads(path.read_text())[arch+'@2800000000000']
                raw['tessera_parameters'].update(sram_tiling_model='transfer_only',array_timing_model='bank_events_v1')
                chip=ChipConfig.model_validate(raw)
                result.append(native_record(create_einsum_op([m,k],[k,n],'MK;KN->MN'),chip))
            for field in ('sa_ns','charged_sa_macs','sram_bytes','array_a_read_bytes','array_b_read_bytes'):
                assert result[0][field]==result[1][field],(arch,m,field)
