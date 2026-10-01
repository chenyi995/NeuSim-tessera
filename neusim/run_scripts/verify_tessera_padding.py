"""Independent tile enumeration and native energy checks for padded MAC billing."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
from contextlib import redirect_stdout

from neusim.npusim.backend import tessera_joint as joint
from neusim.npusim.backend import tessera_partitioned as part
from neusim.npusim.backend import tessera_baselines as base
from neusim.run_scripts.run_tessera_partitioned import config
from neusim.run_scripts.compare_tessera_native import ROOT,sha
from neusim.npusim.frontend.llm_ops_lib import create_multi_head_flash_attention_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info


def golden(shape,geometry,tile,arch=None):
    b,m,n,k=shape;mt,nt,kt=tile;h,w=geometry[-2:]
    rk,rn=(h,w) if arch else geometry[:2]
    total=0
    for batch in range(b):
        for row in range(0,m,mt):
            for column in range(0,n,nt):
                for depth in range(0,k,kt):
                    for kr in range(0,min(kt,k-depth),rk):
                        for nr in range(0,min(nt,n-column),rn):
                            total += min(mt,m-row)*h*w
    return total


def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=2e-12,abs_tol=1e-18),(a,b)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve();assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    shutil.copy2(__file__,out/Path(__file__).name);assert sha(Path(__file__))==sha(out/Path(__file__).name)
    report=dict(status="running",integer_checks=0,native_checks=0,energy_ratio_checks=0,winner_checks=0,attention_checks=0)
    configs=[]
    for name,variant in (("WS","full"),("Tessera","full"),("Square","square")):
        c=config(variant,grain=8)
        if name=="WS":c.tessera_parameters["baseline_architecture"]="WS"
        c.tessera_parameters.update(mapping_policy="joint_edp",sa_energy_accounting="padded_tiles")
        configs.append((name,c))
    for name,c in configs:
        arch=c.tessera_parameters.get("baseline_architecture")
        for shape in ((1,3,14,16),(2,5,13,23),(1,16,32,8)):
            b,m,n,k=shape
            geometries=list(base.geometries(c) if arch else part.geometries(m,n,k,c))
            for g in geometries:
                tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                v=joint.candidate_vectors(*shape,2,g,c,tiles)
                for index,coords in enumerate(zip(*tiles[:3])):
                    tile=tuple(map(int,coords));expected=golden(shape,g,tile,arch)
                    if arch:
                        r=base.counts(*shape,2,2,2,g,arch,c.vmem_size_MB*1024**2,c.freq_GHz,c.hbm_bw_GBps,False,tile,"padded_tiles")
                    else:
                        r=part.counts(*shape,2,2,2,g,c.tessera_variant,8,c.sa_dim,c.vmem_size_MB*1024**2,c.freq_GHz,c.hbm_bw_GBps,False,tile,"padded_tiles")
                    assert expected==int(v["charged_sa_macs"][index])==r["charged_sa_macs"]
                    assert r["useful_macs"]==math.prod(shape)
                    assert r["padding_macs"]==expected-math.prod(shape)
                    report["integer_checks"]+=1
                for index in sorted({0,len(tiles[0])//2,len(tiles[0])-1}):
                    tile=tuple(int(x[index]) for x in tiles[:3])
                    for resident in (False,True):
                        vectors=joint.candidate_vectors(*shape,2,g,c,tiles,resident)
                        s=part.evaluate_candidate(*shape,"DT_BFLOAT16",g,c,resident,tile).stats
                        edp,time,energy=joint.native_objective(vectors,b*m*n*k,c,c.hbm_bw_GBps*1024**3)
                        assert s.execution_time_ns==time[index]
                        close(s.total_energy_J,energy[index]);close(s.total_energy_J*s.execution_time_ns,edp[index])
                        report["native_checks"]+=1
                        old=c.model_copy(deep=True);old.tessera_parameters["sa_energy_accounting"]="useful"
                        before=part.evaluate_candidate(*shape,"DT_BFLOAT16",g,old,resident,tile).stats
                        assert before.execution_time_ns==s.execution_time_ns
                        assert before.memory_traffic_bytes==s.memory_traffic_bytes
                        assert before.tessera_details["sram_bytes"]==s.tessera_details["sram_bytes"]
                        close(s.dynamic_energy_sa_J/before.dynamic_energy_sa_J,float(vectors["charged_sa_macs"][index])/(b*m*n*k))
                        report["energy_ratio_checks"]+=1
            print(name,shape,report,flush=True)
        # Search winners checked by scalar native replay of every small candidate.
        shape=(1,3,5,7);b,m,n,k=shape
        geometries=list(base.geometries(c) if arch else part.geometries(m,n,k,c))
        for resident in (False,True):
            best=None
            for g in geometries:
                tiles=joint.factor_tiles(m,n,k,2,g[3],c.vmem_size_MB*1024**2)
                for coords in zip(*tiles[:3]):
                    tile=tuple(map(int,coords));s=part.evaluate_candidate(*shape,"DT_BFLOAT16",g,c,resident,tile).stats
                    value=s.total_energy_J*s.execution_time_ns
                    best=value if best is None else min(best,value)
            g,tile=joint.select_mapping(*shape,"DT_BFLOAT16",c.model_dump_json(),resident)
            s=part.evaluate_candidate(*shape,"DT_BFLOAT16",g,c,resident,tile).stats
            close(best,s.total_energy_J*s.execution_time_ns);report["winner_checks"]+=1
        op=create_multi_head_flash_attention_op([1,3,4,16],[1,19,2,16],[1,19,2,16])
        with redirect_stdout(part.Sink()):s=fill_operators_execution_info([op],c)[0].stats
        d=s.tessera_details
        expected=sum(p["copies"]*golden((1,p["M"],p["N"],p["K"]),p["geometry"],p["memory_tile"],arch)
                     for p in d["phase_plans"])
        assert d["sa_arithmetic_ops"]==2*expected and d["charged_sa_macs"]==expected
        report["attention_checks"]+=1
    # User's exact example: a logical 16x14 weight tile occupies 16x16 PEs.
    c=config("square",grain=8);g=(16,16,True,1,16,16)
    r=part.counts(1,3,14,16,2,2,2,g,"square",8,128,12<<20,1,c.hbm_bw_GBps,False,(3,14,16),"padded_tiles")
    assert r["useful_macs"]==3*16*14 and r["charged_sa_macs"]==3*16*16
    report["example_16x14"]=dict(useful_macs=r["useful_macs"],charged_sa_macs=r["charged_sa_macs"],padding_macs=r["padding_macs"])
    report["status"]="PASS"
    files=[Path(__file__),ROOT/"neusim/npusim/backend/tessera_partitioned.py",ROOT/"neusim/npusim/backend/tessera_joint.py",
           ROOT/"neusim/npusim/backend/tessera_baselines.py",ROOT/"neusim/npusim/backend/power_model.py",
           ROOT/"neusim/run_scripts/run_tessera_joint.py",ROOT/"neusim/run_scripts/run_tessera_partitioned.py"]
    report["source_sha256"]={str(p):sha(p) for p in files}
    with (out/"verification.json").open("x") as stream:stream.write(json.dumps(report,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==report
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
