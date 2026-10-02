"""Equal-PE area/energy exploration using the submitted RTL plot and native costs."""
import argparse
import ast
import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTL_PLOT = ROOT.parents[1] / 'Tessera-HPCA-2026/fig/plotting/make_area_figs.py'
RTL_REPORT = ROOT.parents[1] / 'SA-rtl/docs/SYN22_ARRAY_CHIP_DATA.md'
ENERGY = ROOT / 'configs/chips/tessera_joules_energy.json'
REFERENCE = ROOT.parents[1] / 'FissionSA'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rtl_data():
    """Read literal arrays; importing the figure source would configure pyplot."""
    values = {}
    for node in ast.parse(RTL_PLOT.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            key = getattr(node.targets[0], 'id', '')
            if key in ('A_FMA', 'A_LOG', 'A_SRAM', 'P_FMA', 'P_LOG', 'P_SRAM'):
                if isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Mult):
                    values[key] = ast.literal_eval(node.value.left) * ast.literal_eval(node.value.right)
                else:
                    values[key] = ast.literal_eval(node.value)
    assert len(values) == 6 and all(len(v) == 10 for v in values.values())
    return values


# chenyi9: decision start -- independent WS uses same-grain Planaria SRAM and WS logic.
def with_partitioned_ws_sram(record, source=None):
    """Apply the requested area proxy without changing simulated energy or delay."""
    result = dict(record)
    if result['family'] != 'WS-independent':
        return result
    source = rtl_data() if source is None else source
    # Source: RTL_PLOT A_FMA/A_LOG index 0 (WS); A_SRAM Planaria tiers
    # at indices 2,4,6,8. Scaling preserves 128x128 PEs from 32x32 RTL.
    assert result['rtl_index'] == 0
    sram_grain = result['grain'] * result['rtl_side'] // math.isqrt(result['total_pes'])
    sram_index = 2 + 2 * (16, 8, 4, 2).index(sram_grain)
    array_area = result['scale'] * (source['A_FMA'][0] + source['A_LOG'][0])
    sram_area = result['scale'] * source['A_SRAM'][sram_index]
    result.update(area_model='ws_logic_planaria_sram', sram_rtl_index=sram_index,
                  sram_rtl_grain=sram_grain, array_area_mm2=array_area,
                  sram_area_mm2=sram_area, area_mm2=array_area+sram_area)
    return result
# chenyi9: decision end


def configurations(bandwidth, mapping='joint_edp'):
    from neusim.run_scripts.run_tessera_partitioned import config
    from neusim.run_scripts.run_tessera_joint import apply_array_energy
    data = rtl_data()
    output, metadata = {}, []
    # chenyi9: decision start -- 128x128 total PEs; scale matching 32x32 RTL by 16.
    # Source: user ruling; the plot's family order is d=32,16,8,4,2.
    specs = [('WS', 'WS', 128, 0)]
    specs += [(f'WS-independent-{g}', 'WS-independent', g, 0) for g in (64, 32, 16, 8)]
    for grain in (64, 32, 16, 8):
        tier = (16, 8, 4, 2).index(grain // 4)
        specs += [(f'Planaria-{grain}', 'Planaria', grain, 2+2*tier),
                  (f'Tessera-{grain}', 'Tessera', grain, 3+2*tier)]
    for name, family, grain, index in specs:
        chip = config('full', grain=grain)
        chip.name = name
        chip.hbm_bw_GBps = bandwidth / 1024**3
        if family != 'Tessera':
            chip.tessera_parameters.update(baseline_architecture=family, baseline_grain=grain)
        if mapping not in ('joint_edp','native_tail'):
            raise ValueError('PPA comparison requires native-tail or joint EDP mapping')
        chip.tessera_parameters.update(mapping_policy=mapping, selection_bandwidths=[bandwidth],
            sa_energy_accounting='padded_tiles', sram_read_accounting='padded_tiles')
        # chenyi9: decision start -- repaired PPA uses native compute/transfer separation.
        chip.tessera_parameters['sram_tiling_model'] = 'transfer_only'
        chip.tessera_parameters['array_timing_model'] = 'bank_events_v1'
        # chenyi9: decision end
        # chenyi9: decision start -- compare native bulk/tail with joint EDP tiling.
        chip.tessera_parameters['mapping_objective'] = 'e2e_edp' if mapping=='joint_edp' else 'native_hbm_reuse'
        # chenyi9: decision end
        apply_array_energy(chip, 'WS' if family.startswith('WS') else name)
        # chenyi9: decision start -- use revision NeuSim energy; RTL scales area only.
        # Source: 20260930_energy_corrected_v1/main_costs/configs.json. Retain
        # its calibrated compute proxy and native SRAM/HBM/NoPG/regulator terms.
        # Total vectorless RTL power is not an energy-per-access multiplier.
        sram_ratio = 1.0
        logic_ratio = 1.0
        # chenyi9: decision end
        scale = (chip.sa_dim / 32)**2
        array_area = scale*(data['A_FMA'][index] + data['A_LOG'][index])
        sram_area = scale*data['A_SRAM'][index]
        chip.tessera_parameters.update(ppa_sram_power_ratio=sram_ratio,
            ppa_static_array_power_ratio=logic_ratio, ppa_rtl_index=index,
            ppa_area_source=str(RTL_PLOT), ppa_area_scale=scale)
        assert chip.num_sa*chip.sa_dim**2 == 128**2
        metadata.append(with_partitioned_ws_sram(dict(architecture=name, family=family, grain=grain, mapping_policy=mapping,
            total_pes=chip.sa_dim**2, subarrays=(chip.sa_dim//grain)**2,
            rtl_index=index, rtl_side=32, rtl_grain=grain//4 if family in ('Planaria','Tessera') else 32,
            scale=scale, array_area_mm2=array_area, sram_area_mm2=sram_area,
            area_mm2=array_area+sram_area, sram_power_ratio=sram_ratio,
            static_array_power_ratio=logic_ratio,
            compute_pj_per_op=chip.tessera_parameters['array_energy_pj_per_op']), data))
        output[name, bandwidth] = chip
    # chenyi9: decision end
    return output, metadata


def verify(out):
    """Check variable-grain kernels against unchanged FissionSA source functions."""
    import random
    import sys
    import types
    import numpy as np
    from neusim.npusim.backend import tessera_baselines as base
    from neusim.npusim.backend import tessera_joint as joint
    from neusim.npusim.backend import tessera_partitioned as part
    from neusim.run_scripts.run_tessera_partitioned import config
    from neusim.run_scripts.run_tessera_joint import native_record
    from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
    # Read-only original references, not archived snapshots.
    for name, directory in (('fissionsa', REFERENCE/'fissionsa'),
                            ('fissionsa.modes', REFERENCE/'fissionsa/modes')):
        module = types.ModuleType(name); module.__path__ = [str(directory)]; sys.modules[name] = module
    from fissionsa.modes import combo0, combo4
    from fissionsa.workload import GEMMLayer
    out.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    configs, area = configurations(2_800_000_000_000)  # Existing HBM4 sweep endpoint.
    checks = dict(source_kernel=0, vector_scalar=0, native_energy=0, full_grid=0)
    # chenyi9: decision start -- verify every energy coefficient against revision.
    revision_path = ROOT / 'artifacts/tessera-20261001/ppa/joint_edp/main_costs/configs.json'
    revision = json.loads(revision_path.read_text())
    for (name, bw), chip in configs.items():
        reference = 'WS' if name.startswith('WS') else 'Tessera-8' if name.startswith('Tessera') else 'Planaria-32'
        golden = revision[reference + '@' + str(bw)]
        for key, value in chip.model_dump().items():
            if 'power' in key:
                assert value == golden[key], (name, key, value, golden[key])
        assert chip.tessera_parameters['array_energy_pj_per_op'] == golden['tessera_parameters']['array_energy_pj_per_op']
        assert chip.vmem_size_MB == golden['vmem_size_MB']
        assert chip.vmem_bw_GBps == config('full').vmem_bw_GBps
    checks['revision_energy_configurations'] = len(configs)
    # chenyi9: decision end
    source_samples = []
    for (name, _), chip in configs.items():
        arch = chip.tessera_parameters.get('baseline_architecture')
        grain = chip.tessera_parameters['grain']
        shapes = [(1,1,1), (31,63,65), (127,128,128), (257,193,259)]
        for seed in (7, 29, 101):
            rng = random.Random(seed)
            shapes += [tuple(rng.randrange(1,600) for _ in range(3)) for __ in range(6)]
        if arch:
            for g in base.geometries(chip):
                for m,n,k in shapes:
                    layer = GEMMLayer(0,m,n,k)
                    if arch == 'Planaria':
                        a,b = g[0]//grain,g[1]//grain
                        combo0._PLANARIA_D,combo0._PLANARIA_M = grain,128//grain
                        combo0._planaria_select_ab = lambda m,n,k,a=a,b=b:(a,b)
                        reference = combo0.simulate_layer_planaria_grid(layer,grain,128//grain,0)
                    else:
                        reference = combo4.simulate_layer_ws_disconnected_grid(layer,grain,128//grain,0)
                    actual = base.array_cost(arch,m,n,k,g,grain)
                    assert (actual[0],actual[1]+actual[2],actual[3]) == (reference.compute_cycles,reference.input_words,reference.output_words),(name,m,n,k,g,actual,reference)
                    checks['source_kernel'] += 1
            # A weight tile for every physical unit must execute as one wave.
            g = next(g for g in base.geometries(chip) if g[:2] == (grain,grain))
            expected = 17 + 3*grain-2
            assert base.array_cost(arch,17,128,128,g,grain)[0] == expected
            checks['full_grid'] += 1
        for shape in ((1,3,5,7),(2,17,33,67),(1,257,130,259)):
            b,m,n,k = shape
            geometries = list(base.geometries(chip) if arch else part.geometries(m,n,k,chip))
            for g in geometries:
                tiles = joint.factor_tiles(m,n,k,2,g[3],chip.vmem_size_MB*1024**2)
                vector = joint.candidate_vectors(*shape,2,g,chip,tiles)
                indices = np.unique(np.linspace(0,len(tiles[0])-1,min(7,len(tiles[0])),dtype=int))
                for i in indices:
                    tile = tuple(int(x[i]) for x in tiles[:3])
                    kwargs = dict(forced_tile=tile,sa_energy_accounting='padded_tiles',sram_read_accounting='padded_tiles',
                                  transfer_only=chip.tessera_parameters.get('sram_tiling_model')=='transfer_only')
                    if arch:
                        kwargs['bank_timing']=chip.tessera_parameters.get('array_timing_model')=='bank_events_v1'
                        scalar = base.counts(*shape,2,2,2,g,arch,chip.vmem_size_MB*1024**2,
                            chip.freq_GHz,chip.hbm_bw_GBps,grain=grain,**kwargs)
                    else:
                        scalar = part.counts(*shape,2,2,2,g,chip.tessera_variant,grain,128,
                            chip.vmem_size_MB*1024**2,chip.freq_GHz,chip.hbm_bw_GBps,**kwargs)
                    for field in ('sram_bytes','hbm_bytes','charged_sa_macs','padding_sram_read_bytes',
                                  'rounds','reduction_ops','peak_live_bytes'):
                        assert vector[field][i] == scalar[field],(name,shape,g,tile,field)
                    assert vector['sa_ns'][i] == math.ceil(scalar['sa_cycles']/chip.freq_GHz)
                    checks['vector_scalar'] += 1
                i = int(indices[len(indices)//2]); tile = tuple(int(x[i]) for x in tiles[:3])
                with __import__('contextlib').redirect_stdout(part.Sink()):
                    native = part.evaluate_candidate(*shape,'DT_BFLOAT16',g,chip,forced_tile=tile).stats
                _,time,energy = joint.native_objective(vector,b*m*n*k,chip,2_800_000_000_000)
                assert native.execution_time_ns == time[i]
                assert math.isclose(native.total_energy_J,energy[i],rel_tol=2e-12),(name,native.total_energy_J,energy[i])
                checks['native_energy'] += 1
        print('PASS',name,checks,flush=True)
    # Preserve the existing grain-32 default exactly (new label plus explicit grain).
    old=config('full',grain=32);old.tessera_parameters['baseline_architecture']='Planaria-32'
    explicit=old.model_copy(deep=True);explicit.tessera_parameters.update(baseline_architecture='Planaria',baseline_grain=32)
    assert list(base.geometries(old)) == list(base.geometries(explicit))
    for m,n,k in ((1,63,65),(17,193,257),(257,128,129)):
        a=native_record(create_einsum_op([m,k],[k,n],'MK;KN->MN'),old)
        b=native_record(create_einsum_op([m,k],[k,n],'MK;KN->MN'),explicit)
        for key in ('time_ns','energy_J','sa_ns','hbm_bytes','sram_bytes'):
            assert a[key] == b[key],key
    # Source: retained full-run failure_case.json and failure_vectors.json in
    # results/tessera/20260930_ppa_equal_pe_v1. A literal Python-int product
    # independently checks the same capacity bound without NumPy overflow.
    # Codex: retain the historical overflow regression under its original
    # execution semantics; the repaired model eliminates the repeated work.
    chip=configs['Planaria-8',2_800_000_000_000].model_copy(deep=True)
    chip.tessera_parameters['sram_tiling_model']='restart_per_tile'
    chip.tessera_parameters.pop('array_timing_model',None)
    shape=(1,2482,28672,4096);g=(8,2048,True,1,8,2048);tile=(1,1,1)
    peak=part.checked_tile(*shape[1:],2,2,4,g[3],chip.vmem_size_MB*1024**2,tile)
    tiles=tuple(np.array([v],dtype=np.int64) for v in (*tile,peak))
    vector=joint.candidate_vectors(*shape,2,g,chip,tiles)
    scalar=base.counts(*shape,2,2,2,g,'Planaria',chip.vmem_size_MB*1024**2,chip.freq_GHz,
        chip.hbm_bw_GBps,forced_tile=tile,sa_energy_accounting='padded_tiles',sram_read_accounting='padded_tiles',grain=8)
    assert scalar['sa_cycles']*128**2 > np.iinfo(np.int64).max
    assert scalar['charged_sa_macs'] <= scalar['sa_cycles']*128**2
    assert int(vector['sa_ns'][0]) == scalar['sa_cycles']
    assert int(vector['charged_sa_macs'][0]) == scalar['charged_sa_macs']
    checks['large_candidate_overflow']=1
    files=[Path(__file__),Path(__file__).with_name('run_tessera_joint.py'),RTL_PLOT,RTL_REPORT,ENERGY,revision_path,
        ROOT/'neusim/npusim/backend/tessera_baselines.py',ROOT/'neusim/npusim/backend/tessera_joint.py',
        ROOT/'neusim/npusim/backend/tessera_partitioned.py',REFERENCE/'fissionsa/modes/combo0.py',REFERENCE/'fissionsa/modes/combo4.py']
    result=dict(status='PASS',checks=checks,legacy_grain_32='identical',configurations=area,
                source_sha256={str(p):sha(p) for p in files})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    assert json.loads((out/'verification.json').read_text()) == result
    print(json.dumps(checks),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['verify'])
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='verify':verify(args.out.resolve())


if __name__=='__main__':main()
