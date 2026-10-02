"""Whole-region load and hidden transpose: paper boundary and replay invariants."""
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend import tessera_baselines as base, tessera_joint as joint
from neusim.npusim.backend import tessera_native_tail as tail, tessera_partitioned as part
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op, create_multi_head_flash_attention_op
from neusim.run_scripts.run_tessera_partitioned import native_record
from neusim.run_scripts.tessera_request_dispatch import segments, tile_plan

ROOT=Path(__file__).resolve().parents[4]


def chips(mode):
    path=ROOT/'artifacts/tessera-20261001/ppa'/mode/'main_costs/configs.json'
    for name,raw in json.loads(path.read_text()).items():
        raw['tessera_parameters'].update(sram_tiling_model='transfer_only',array_timing_model='merged_load_v2')
        yield name.split('@')[0],ChipConfig.model_validate(raw)


@pytest.mark.parametrize('seed',[7,29,101])
def test_whole_region_load_against_independent_resource_schedule(seed):
    rng=random.Random(seed)
    for _ in range(100):
        grain=rng.choice((8,16,32,64));h=rng.choice([d for d in (8,16,32,64,128) if d>=grain])
        w=rng.choice([d for d in (8,16,32,64,128) if d>=grain])
        m=rng.choice((1,h-1,h,h+1,2*(h+w)))
        folds=rng.randrange(1,100)
        port=stream=0;free=[0,0]
        for f in range(folds):
            begin=max(port,free[f%2]);port=begin+h
            start=max(port,stream);stream=start+m
            free[f%2]=stream+w-1
        expected=stream+h+w-2
        g=(h,w,True,1,h,w)
        assert tail.lane_time(m,g,folds,grain,'Planaria',merged_load=True)==expected
        scalar=base.array_cost('Planaria',m,w,h*folds,g,grain,merged_load=True)
        vector=joint.array_vectors('Planaria',*[np.array([x]) for x in (m,w,h*folds)],
            g,'full',grain,128,active_pe_budget=h*w,merged_load=True)
        assert scalar[0]==int(vector[0][0])==expected
        # Source: paper III-D / Gemmini TIMING_AUDIT.md, first weight to last output.
        d=h;gg=(d,d,True,1,d,d)
        tt=tail.lane_time(m,gg,1,d,'Tessera',merged_load=True)
        ws=tail.lane_time(m,gg,1,d,'Planaria',merged_load=True)
        assert tt==m+2*d-1 and ws==m+3*d-2 and ws-tt==d-1
        # Registered strip seams change only the final tail.
        t=tail.lane_time(m,gg,folds,grain,'Tessera',merged_load=True)
        u=tail.lane_time(m,gg,folds+1,grain,'Tessera',merged_load=True)
        assert u-t==max(m,d)
        assert t==d+(folds-1)*max(m,d)+m+d-1+(d//grain-1)


def test_mapping_ranking_matches_scalar_native_energy():
    for name,chip in chips('joint_edp'):
        arch=chip.tessera_parameters.get('baseline_architecture');grain=chip.tessera_parameters['grain']
        for m,n,k in ((1,131,259),(17,193,257),(129,129,129)):
            gg=list(base.geometries(chip) if arch else part.geometries(m,n,k,chip))
            for g in gg[::max(1,len(gg)//4)]:
                tiles=joint.factor_tiles(m,n,k,2,g[3],chip.vmem_size_MB*1024**2)
                vectors=joint.candidate_vectors(1,m,n,k,2,g,chip,tiles)
                _,times,energies=joint.native_objective(vectors,m*n*k,chip,chip.hbm_bw_GBps*1024**3)
                for i in (0,len(tiles[0])-1):
                    tile=tuple(int(a[i]) for a in tiles[:3])
                    stats=part.evaluate_candidate(1,m,n,k,'DT_BFLOAT16',g,chip,forced_tile=tile).stats
                    assert stats.execution_time_ns==times[i]
                    assert math.isclose(stats.total_energy_J,energies[i],rel_tol=2e-12)


@pytest.mark.parametrize('mode',['native_tail','joint_edp'])
def test_profile_and_asynchronous_replay_have_same_array_boundary(mode):
    chip=dict(chips(mode))['Tessera-8'];costs=SimpleNamespace(side=128,grain=8,chip=chip)
    ops=[create_einsum_op([1,4096],[4096,4096],'MK;KN->MN'),
         create_einsum_op([17,257],[257,193],'MK;KN->MN'),
         create_multi_head_flash_attention_op([1,17,4,64],[1,129,1,64],[1,129,1,64])]
    for op in ops:
        r=native_record(op,chip);mapping=json.loads(r['mapping_json'])
        phases=mapping.get('phase_plans') or [{**mapping,'sa_ns':r['sa_ns']}]
        total=0
        for s in segments(dict(phases=phases),costs):
            if 'mixed_tail' in s:
                p=tail.tail_plan((0,)*16,128,8,s['m'],tuple(s['mixed_tail']),s['family'],merged_load=True)
                cycles=max(t for t,_ in p[1])
            else:
                p=tile_plan((0,)*16,128,8,tuple(s['g']),s['m'],s['nk'],s['nn'],chip.tessera_variant,
                            chip.freq_GHz,merged_load=True)
                cycles=max(t for t,_ in p[1])*chip.freq_GHz
            total+=cycles*s['groups']/chip.freq_GHz
        assert total==r['sa_ns']
        assert r['peak_live_bytes']<=chip.vmem_size_MB*1024**2
