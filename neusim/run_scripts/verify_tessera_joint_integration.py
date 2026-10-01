"""Check actual workload joint mappings against native legacy and forced replay."""
import argparse
from contextlib import redirect_stdout
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import traceback

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.tessera_partitioned import Sink
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
from neusim.run_scripts.compare_tessera_native import ROOT,rows,sha
from neusim.run_scripts.run_tessera_joint import restore_operator,MAIN


def main():
    p=argparse.ArgumentParser();p.add_argument("--pilot",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();pilot=a.pilot.resolve()
    assert out.is_relative_to(ROOT) and not out.exists();out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1]);shutil.copyfile(__file__,out/Path(__file__).name)
    manifest=json.loads((pilot/"manifest.json").read_text());assert manifest["status"]=="PASS" and manifest["pilot"]
    configs=json.loads((pilot/"configs.json").read_text());base=Path(manifest["base"])
    raw=list(rows(pilot/"operator_costs.csv"));ids={int(r["operator_id"]) for r in raw if r["kind"]=="shape"}
    shapes={};ops={};sources=[Path(__file__),pilot/"manifest.json",pilot/"configs.json",pilot/"operator_costs.csv",base/"inputs_snapshot/inputs/workloads.csv",base/"speed/trace_graph_operators.jsonl"]
    for r in rows(base/"inputs_snapshot/inputs/workloads.csv"):
        if int(r["shape_id"]) in ids:shapes[int(r["shape_id"])]=tuple(int(r[k]) for k in ("M","N","K"))
    graph_ids={int(r["operator_id"]) for r in raw if r["kind"]=="graph"}
    for line in (base/"speed/trace_graph_operators.jsonl").open():
        r=json.loads(line)
        if r["operator_id"] in graph_ids:ops[r["operator_id"]]=restore_operator(r["operator"])
    result=dict(status="running",native_replay_checks=0,legacy_dominance_checks=0,resident_budget_checks=0)
    observations=[]
    try:
        low,high=min(manifest["bandwidth_bytes_per_second"]),max(manifest["bandwidth_bytes_per_second"])
        for r in raw:
            name=r["architecture"];bw=int(r["bandwidth_bytes_per_second"])
            if bw not in (low,high) or name not in MAIN:continue
            c=ChipConfig.model_validate(configs[f"{name}@{bw}"]);sid=int(r["operator_id"])
            if r["kind"]=="shape":
                m,n,k=shapes[sid];op=create_einsum_op([m,k],[k,n],"MK;KN->MN")
            else:op=ops[sid].model_copy(deep=True)
            with redirect_stdout(Sink()):s=fill_operators_execution_info([op.model_copy(deep=True)],c)[0].stats
            for field,actual in (("time_ns",s.execution_time_ns),("energy_J",s.total_energy_J),("hbm_bytes",s.memory_traffic_bytes),("sram_bytes",s.tessera_details["sram_bytes"])):
                assert math.isclose(float(r[field]),actual,rel_tol=2e-12,abs_tol=1e-18),(r,field,actual)
            result["native_replay_checks"]+=1
            if op.opcode!="Einsum":continue
            legacy=c.model_copy(deep=True);legacy.tessera_parameters.pop("mapping_policy");legacy.tessera_parameters.pop("selection_bandwidths")
            with redirect_stdout(Sink()):old=fill_operators_execution_info([op.model_copy(deep=True)],legacy)[0].stats
            new_edp=s.total_energy_J*s.execution_time_ns;old_edp=old.total_energy_J*old.execution_time_ns
            assert new_edp<=old_edp*(1+2e-12),(name,sid,bw,new_edp,old_edp)
            mapping=json.loads(r["mapping_json"]);forced=op.model_copy(deep=True)
            forced.tessera_spec=dict(geometry=mapping["geometry"],memory_tile=mapping["memory_tile"])
            with redirect_stdout(Sink()):fixed=fill_operators_execution_info([forced],legacy)[0].stats
            assert fixed.execution_time_ns==s.execution_time_ns
            assert math.isclose(fixed.total_energy_J,s.total_energy_J,rel_tol=2e-12)
            result["legacy_dominance_checks"]+=1
            detail=s.tessera_details;assert detail["peak_live_bytes"]<=c.vmem_size_MB*1024**2
            assert detail["hbm_partial_read_bytes"]==detail["hbm_partial_write_bytes"]==0
            observations.append(dict(kind=r["kind"],operator_id=sid,architecture=name,bandwidth_bytes_per_second=bw,
                old_edp_J_ns=old_edp,joint_edp_J_ns=new_edp,old_memory_tile=old.tessera_details["memory_tile"],joint_memory_tile=detail["memory_tile"],
                old_hbm_bytes=old.memory_traffic_bytes,joint_hbm_bytes=s.memory_traffic_bytes))
            result["resident_budget_checks"]+=1
        result["status"]="PASS"
    except BaseException:
        result["status"]="FAIL";(out/"failure.txt").write_text(traceback.format_exc());raise
    finally:
        result["source_sha256"]={str(x):sha(x) for x in sources}
        (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
        (out/"observations.json").write_text(json.dumps(observations,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==result
    assert json.loads((out/"observations.json").read_text())==observations
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
