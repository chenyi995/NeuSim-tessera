"""Check sweep reconstruction against native frontend timing and real graphs."""
import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
import math
import os
from pathlib import Path
import shutil

from neusim.run_scripts.compare_tessera_native import ROOT, sha
from neusim.run_scripts import run_tessera_speed as sweep
from neusim.run_scripts.run_tessera_partitioned import native_record, trace_jobs
from neusim.npusim.frontend.tessera_workloads import native_llm_graph
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op, create_multi_head_flash_attention_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--inputs",type=Path,required=True)
    p.add_argument("--baseline-verification",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    metadata=json.loads((a.inputs/"workload_metadata.json").read_text())
    grid=json.loads((a.baseline_verification/"bandwidth_grid.json").read_text())["bytes_per_second"]
    sweep.initialize(metadata,grid)
    checked=0
    for name,chip in sweep._CONFIGS.items():
        ops=[create_einsum_op([m,k],[k,n],"MK;KN->MN")
             for m,n,k in ((1,64,64),(13,193,257),(257,513,127))]
        ops.append(create_multi_head_flash_attention_op([1,7,8,64],[1,145,2,64],[1,145,2,64]))
        for original in ops:
            r=native_record(original.model_copy(deep=True),chip)
            previous=math.inf
            for bw in grid:
                # Independent native formula from op_analysis_lib; the mapping
                # is frozen, so its component costs/bytes stay constant here.
                hbm=max(math.ceil(r["hbm_bytes"]*1e9/bw),chip.hbm_latency_ns) if r["hbm_bytes"] else 0
                expected=max(r["sa_ns"],r["vu_ns"],r["ici_ns"],hbm)
                actual=sweep.time_at(r,bw)
                assert actual==expected and actual<=previous,(name,bw,actual,expected)
                previous=actual;checked+=1
            assert previous==r["time_ns"]
            if original.opcode=="Einsum":
                # Use the native frontend at another bandwidth and the same
                # geometry; check its measured traffic before asserting equality.
                mapping=json.loads(r["mapping_json"])
                original.tessera_spec={"geometry":mapping["geometry"]}
                low=chip.model_copy(deep=True);low.hbm_bw_GBps=grid[0]/1024**3
                with redirect_stdout(StringIO()):
                    actual=fill_operators_execution_info([original],low)[0].stats
                assert actual.memory_traffic_bytes==r["hbm_bytes"]
                assert actual.sa_time_ns==r["sa_ns"]
                assert actual.execution_time_ns==sweep.time_at(r,grid[0])
                checked+=1
    graph_checks=0
    for job in trace_jobs(metadata,1):
        histogram,batches=sweep.trace_job(job)
        for row in batches:
            expected=sum(repeat*sweep.time_at(dict(zip(sweep.METRICS,key[3:])),grid[-1])
                         for key,repeat in histogram if key[1]==row["architecture"])
            assert expected==row["reference_time_ns"]
            graph_checks+=1
    shutil.copyfile(__file__,out/Path(__file__).name)
    source=[Path(__file__),ROOT/"neusim/run_scripts/run_tessera_speed.py"]
    record=dict(status="PASS",component_timing_checks=checked,real_trace_architecture_checks=graph_checks,
                source_sha256={str(x):sha(x) for x in source})
    (out/"verification.json").write_text(json.dumps(record,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==record
    print(json.dumps(record))


if __name__=="__main__":main()
