"""Physical-work invariants for the repaired transfer-only PPA accounting."""
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend import tessera_baselines as base, tessera_joint as joint
from neusim.npusim.backend import tessera_partitioned as part, tessera_native_tail as tail
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op, create_multi_head_flash_attention_op
from neusim.run_scripts.run_tessera_partitioned import native_record
from neusim.run_scripts.tessera_request_dispatch import segments, tile_plan

ROOT = Path(__file__).resolve().parents[4]


def chips(mode='joint_edp'):
    path = ROOT / 'artifacts/tessera-20261001/ppa' / mode / 'main_costs/configs.json'
    for name, raw in json.loads(path.read_text()).items():
        raw['tessera_parameters']['sram_tiling_model'] = 'transfer_only'
        yield name.split('@')[0], ChipConfig.model_validate(raw)


@pytest.mark.parametrize('seed', [7, 29, 101])
def test_ws_physical_work_independent_of_transfer_boundaries(seed):
    rng = random.Random(seed)
    for grain in (128, 64, 32, 16, 8):
        m,n,k = (rng.randrange(1, 300) for _ in range(3))
        g=(grain,grain,True,(128//grain)**2,grain,grain)
        # Explicit physical weight-block enumeration; no production padding helper.
        blocks=[(i,j) for i in range(0,k,grain) for j in range(0,n,grain)]
        charged=sum(m*grain*grain for _ in blocks)
        expected_a=sum(m*grain for _ in blocks)*2
        expected_b=sum(grain*grain for _ in blocks)*2
        gold=None
        for tile in ((1,1,1),(min(m,16),min(n,16),k),(m,n,1),(m,n,k)):
            # Some full output tiles exceed capacity with many producer planes.
            # Check their rejection; do not enlarge SRAM to make the test fit.
            tm,tn,tk=tile
            footprint=4*(tm*tk+tk*tn)+4*(g[3]+1)*tm*tn
            if footprint>12*1024**2:
                with pytest.raises(ValueError,match='exceeds capacity'):
                    base.counts(1,m,n,k,2,2,2,g,'WS-independent',12*1024**2,1,2607,
                                forced_tile=tile,grain=grain,transfer_only=True)
                continue
            v=base.counts(1,m,n,k,2,2,2,g,'WS' if grain==128 else 'WS-independent',
                12*1024**2,1,2607,forced_tile=tile,sa_energy_accounting='padded_tiles',
                sram_read_accounting='padded_tiles',grain=grain,transfer_only=True)
            assert v['charged_sa_macs']==charged
            assert v['array_a_read_bytes']==expected_a and v['array_b_read_bytes']==expected_b
            work=tuple(v[f] for f in ('sa_cycles','charged_sa_macs','reduction_ops','partial_write_bytes'))
            if gold is None:gold=work
            assert work==gold
            ha=sum(min(tile[0],m-i)*k*2 for i in range(0,m,tile[0]) for _ in range(0,n,tile[1]))
            hb=sum(k*min(tile[1],n-j)*2 for _ in range(0,m,tile[0]) for j in range(0,n,tile[1]))
            assert v['hbm_bytes']==ha+hb+m*n*2
            assert v['peak_live_bytes']<=12*1024**2


def test_all_families_scalar_vector_and_native_energy():
    for name,chip in chips():
        arch=chip.tessera_parameters.get('baseline_architecture')
        grain=chip.tessera_parameters['grain']
        for b,m,n,k in ((1,17,131,259),(2,31,193,65)):
            gg=list(base.geometries(chip) if arch else part.geometries(m,n,k,chip))
            for g in gg[::max(1,len(gg)//4)]:
                tiles=joint.factor_tiles(m,n,k,2,g[3],chip.vmem_size_MB*1024**2)
                v=joint.candidate_vectors(b,m,n,k,2,g,chip,tiles)
                assert len(set(v['sa_ns']))==len(set(v['charged_sa_macs']))==1
                _,time,energy=joint.native_objective(v,b*m*n*k,chip,chip.hbm_bw_GBps*1024**3)
                for i in (0,len(tiles[0])//2,len(tiles[0])-1):
                    tile=tuple(int(axis[i]) for axis in tiles[:3])
                    kwargs=dict(forced_tile=tile,sa_energy_accounting='padded_tiles',
                                sram_read_accounting='padded_tiles',transfer_only=True)
                    if arch:
                        s=base.counts(b,m,n,k,2,2,2,g,arch,chip.vmem_size_MB*1024**2,
                            chip.freq_GHz,chip.hbm_bw_GBps,grain=grain,**kwargs)
                    else:
                        s=part.counts(b,m,n,k,2,2,2,g,chip.tessera_variant,grain,128,
                            chip.vmem_size_MB*1024**2,chip.freq_GHz,chip.hbm_bw_GBps,**kwargs)
                    for field in ('charged_sa_macs','sram_bytes','hbm_bytes','reduction_ops','rounds','peak_live_bytes'):
                        assert int(v[field][i])==s[field],(name,g,tile,field)
                    assert v['sa_ns'][i]==math.ceil(s['sa_cycles']/chip.freq_GHz)
                    op=part.evaluate_candidate(b,m,n,k,'DT_BFLOAT16',g,chip,forced_tile=tile)
                    assert op.stats.execution_time_ns==time[i]
                    assert math.isclose(op.stats.total_energy_J,energy[i],rel_tol=2e-12)


@pytest.mark.parametrize('mode',['native_tail','joint_edp'])
def test_tessera_replay_uses_compute_span_and_attention_order(mode):
    chip=dict(chips(mode))['Tessera-8']
    ops=[create_einsum_op([512,4096],[4096,22016],'MK;KN->MN'),
         create_einsum_op([17,257],[257,193],'MK;KN->MN'),
         create_multi_head_flash_attention_op([1,17,4,64],[1,129,1,64],[1,129,1,64])]
    costs=SimpleNamespace(side=128,grain=8,chip=chip)
    for op in ops:
        r=native_record(op,chip);mapping=json.loads(r['mapping_json'])
        phases=mapping.get('phase_plans') or [{**mapping,'sa_ns':r['sa_ns']}]
        total=0
        for s in segments(dict(phases=phases),costs):
            if 'mixed_tail' in s:
                p=tail.tail_plan((0,)*16,128,8,s['m'],tuple(s['mixed_tail']),s['family'])
                cycles=max(t for t,_ in p[1])
            else:
                p=tile_plan((0,)*16,128,8,tuple(s['g']),s['m'],s['nk'],s['nn'],chip.tessera_variant,chip.freq_GHz)
                cycles=max(t for t,_ in p[1])*chip.freq_GHz
            total+=cycles*s['groups']/chip.freq_GHz
        assert total==r['sa_ns']
        assert r['peak_live_bytes']<=chip.vmem_size_MB*1024**2


@pytest.mark.parametrize('shape',[(4096,4096,4096),(1,4096,4096),(512,22016,4096),(1,22016,4096)])
def test_single_ws_array_has_identical_compute_under_both_mappers(shape):
    m,n,k=shape
    records=[]
    for mode in ('native_tail','joint_edp'):
        chip=dict(chips(mode))['WS']
        r=native_record(create_einsum_op([m,k],[k,n],'MK;KN->MN'),chip)
        records.append(r)
        assert json.loads(r['mapping_json'])['array_tile']==list(shape)
        assert r['charged_sa_macs']==m*((k+127)//128)*((n+127)//128)*128**2
    for field in ('sa_ns','charged_sa_macs','array_a_read_bytes','array_b_read_bytes','reduction_ops'):
        assert records[0][field]==records[1][field],field
