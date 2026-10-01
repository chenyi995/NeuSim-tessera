"""Validate vectorized joint ranking against scalar counters and native energy."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import traceback
import numpy as np

from neusim.npusim.backend import tessera_joint as joint
from neusim.npusim.backend import tessera_partitioned as part
from neusim.npusim.backend import tessera_baselines as baseline
from neusim.run_scripts.run_tessera_partitioned import config
from neusim.run_scripts.compare_tessera_native import ROOT, sha


def configurations():
    for arch in baseline.ARCHITECTURES:
        c=config("full")
        c.tessera_parameters["baseline_architecture"]=arch
        yield arch,c
    for grain in (32,8):
        for variant in ("full","square","independent_noskew"):
            yield f"Tessera-{grain}-{variant}",config(variant,grain=grain)


def scalar(shape,g,c,tile,resident):
    arch=c.tessera_parameters.get("baseline_architecture")
    if arch:
        return baseline.counts(*shape,2,2,2,g,arch,c.vmem_size_MB*1024**2,c.freq_GHz,c.hbm_bw_GBps,resident,tile)
    return part.counts(*shape,2,2,2,g,c.tessera_variant,c.tessera_parameters["grain"],c.sa_dim,
                       c.vmem_size_MB*1024**2,c.freq_GHz,c.hbm_bw_GBps,resident,tile)


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=1e-18),(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    shutil.copyfile(__file__,out/Path(__file__).name)
    result=dict(status="running",counter_checks=0,native_energy_checks=0,exhaustive_winner_checks=0)
    try:
        for name,c in configurations():
            c.tessera_parameters["mapping_policy"]="joint_edp"
            for shape in ((1,5,33,67),(2,32,64,96),(1,128,192,256)):
                b,m,n,k=shape
                gs=list(baseline.geometries(c) if c.tessera_parameters.get("baseline_architecture") else part.geometries(m,n,k,c))
                for g in gs:
                    tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                    v=joint.candidate_vectors(*shape,2,g,c,tiles)
                    for i in range(len(tiles[0])):
                        tile=tuple(int(x[i]) for x in tiles[:3]);r=scalar(shape,g,c,tile,False)
                        for field in ("reduction_ops","sram_bytes","hbm_bytes","peak_live_bytes","rounds"):
                            assert v[field][i]==r[field],(name,shape,g,tile,field,v[field][i],r[field])
                        assert v["sa_ns"][i]==math.ceil(r["sa_cycles"]/c.freq_GHz)
                        assert v["vu_ns"][i]==part.issue_ns(r["reduction_ops"],c)
                        result["counter_checks"]+=1
                    for i in sorted({0,len(tiles[0])//2,len(tiles[0])-1}):
                        tile=tuple(int(x[i]) for x in tiles[:3])
                        for bw in (394_000_000_000,2_800_000_000_000):
                            chip=c.model_copy(update={"hbm_bw_GBps":bw/1024**3})
                            actual=part.evaluate_candidate(*shape,"DT_BFLOAT16",g,chip,forced_tile=tile).stats
                            edp,time,energy=joint.native_objective(v,b*m*n*k,chip,bw)
                            assert actual.execution_time_ns==time[i]
                            close(actual.total_energy_J,energy[i]);close(actual.total_energy_J*actual.execution_time_ns,edp[i])
                            result["native_energy_checks"]+=1
            # Independent exhaustive reference runs every small candidate through
            # the actual frontend, rather than ranking with the vector objective.
            shape=(1,3,5,7);b,m,n,k=shape
            gs=list(baseline.geometries(c) if c.tessera_parameters.get("baseline_architecture") else part.geometries(m,n,k,c))
            for resident in (False,True):
                best=None
                for g in gs:
                    tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                    for vals in zip(*tiles[:3]):
                        tile=tuple(map(int,vals))
                        s=part.evaluate_candidate(*shape,"DT_BFLOAT16",g,c,resident, tile).stats
                        key=s.total_energy_J*s.execution_time_ns,s.execution_time_ns,g,tile
                        if best is None or key<best:best=key
                g,tile=joint.select_mapping(*shape,"DT_BFLOAT16",c.model_dump_json(),resident)
                s=part.evaluate_candidate(*shape,"DT_BFLOAT16",g,c,resident,tile).stats
                close(best[0],s.total_energy_J*s.execution_time_ns)
                result["exhaustive_winner_checks"]+=1
            print(name,result,flush=True)
        result["status"]="PASS"
    except BaseException:
        result["status"]="FAIL";(out/"failure.txt").write_text(traceback.format_exc());raise
    finally:
        sources=[Path(__file__),ROOT/"neusim/npusim/backend/tessera_joint.py",ROOT/"neusim/npusim/backend/tessera_partitioned.py",ROOT/"neusim/npusim/backend/tessera_baselines.py"]
        result["source_sha256"]={str(x):sha(x) for x in sources}
        (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==result
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
