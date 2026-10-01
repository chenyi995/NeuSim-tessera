"""Check requested pJ/op calibration, native replay and joint-EDP winners."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
from neusim.npusim.backend import tessera_joint as joint, tessera_partitioned as part, tessera_baselines as base
from neusim.run_scripts import run_tessera_joint as runner
from neusim.run_scripts.compare_tessera_native import ROOT, sha

def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=1e-18),(a,b)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    out=p.parse_args().out.resolve();assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    shutil.copy2(__file__,out/Path(__file__).name)
    grid=[394_000_000_000,2_800_000_000_000]
    runner.initialize(grid,array_energy_main=True)
    report=dict(status='running',coefficient_checks=0,native_energy_checks=0,frozen_mapping_checks=0,winner_checks=0)
    # Independent requested coefficients; one multiply and one add per MAC.
    expected={'WS':4.41,'Tessera-8':4.41,'Planaria-32':4.54,'SISA':4.54,'SOSA':4.54,'FlexSA':4.54}
    assert {a for a,bw in runner.CONFIGS}==set(expected)
    for (arch,bw),c in runner.CONFIGS.items():
        close(c.dynamic_power_sa_W/c.peak_SA_tflops_per_sec,expected[arch])
        assert c.tessera_parameters['array_energy_pj_per_op']==expected[arch]
        report['coefficient_checks']+=1
        for shape in ((1,3,14,16),(2,32,64,96),(1,128,192,256)):
            b,m,n,k=shape
            gs=list(base.geometries(c) if arch in base.ARCHITECTURES else part.geometries(m,n,k,c))
            for g in gs:
                tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                for resident in (False,True):
                    v=joint.candidate_vectors(*shape,2,g,c,tiles,resident)
                    edp,times,energies=joint.native_objective(v,b*m*n*k,c,bw)
                    for i in sorted({0,len(tiles[0])//2,len(tiles[0])-1}):
                        tile=tuple(int(t[i]) for t in tiles[:3])
                        op=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,c,resident,tile)
                        s=op.stats
                        assert s.execution_time_ns==times[i]
                        close(s.total_energy_J,energies[i]);close(s.total_energy_J*s.execution_time_ns,edp[i])
                        efficiency=op.dvfs_sa.voltage_conversion_power_efficiency_percent/100
                        close(s.dynamic_energy_sa_J*efficiency,2*b*m*n*k*expected[arch]*1e-12)
                        report['native_energy_checks']+=1
                        old=runner.config('full',grain=8 if arch=='Tessera-8' else 32)
                        old.dynamic_power_W_per_SA=28.19413333  # source: native ChipConfig.dynamic_power_W_per_SA default.
                        old=c.model_copy(update={'dynamic_power_W_per_SA':old.dynamic_power_W_per_SA},deep=True)
                        before=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,old,resident,tile).stats
                        assert before.execution_time_ns==s.execution_time_ns
                        assert before.memory_traffic_bytes==s.memory_traffic_bytes
                        close(s.total_energy_J,before.total_energy_J+before.dynamic_energy_sa_J*(c.dynamic_power_sa_W/old.dynamic_power_sa_W-1))
                        report['frozen_mapping_checks']+=1
        shape=(1,3,5,7);b,m,n,k=shape
        gs=list(base.geometries(c) if arch in base.ARCHITECTURES else part.geometries(m,n,k,c))
        for resident in (False,True):
            best=None
            for g in gs:
                tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                for values in zip(*tiles[:3]):
                    s=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,c,resident,tuple(map(int,values))).stats
                    value=s.total_energy_J*s.execution_time_ns
                    best=value if best is None else min(value,best)
            g,tile=joint.select_mapping(*shape,'DT_BFLOAT16',c.model_dump_json(),resident)
            s=part.evaluate_candidate(*shape,'DT_BFLOAT16',g,c,resident,tile).stats
            close(best,s.total_energy_J*s.execution_time_ns);report['winner_checks']+=1
        print(arch,bw,report,flush=True)
    sources=[Path(__file__),ROOT/'neusim/run_scripts/run_tessera_joint.py',ROOT/'neusim/run_scripts/run_tessera_partitioned.py',
             ROOT/'neusim/configs/chips/ChipConfig.py',ROOT/'neusim/npusim/backend/power_model.py',
             ROOT/'neusim/npusim/backend/tessera_joint.py',ROOT/'neusim/npusim/backend/tessera_partitioned.py',
             ROOT/'neusim/npusim/backend/tessera_baselines.py']
    report.update(status='PASS',source_sha256={str(p):sha(p) for p in sources})
    with (out/'verification.json').open('x') as stream:json.dump(report,stream,indent=2)
    assert json.loads((out/'verification.json').read_text())==report
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
