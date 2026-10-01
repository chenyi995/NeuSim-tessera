"""Record equivalence of direct and globally reused native graph costs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
import multiprocessing
from pathlib import Path
import shutil
import traceback

from neusim.run_scripts import run_tessera_speed as sweep
from neusim.run_scripts.compare_tessera_native import ROOT, sha, rows
from neusim.run_scripts.run_tessera_partitioned import trace_jobs


def main():
    p=argparse.ArgumentParser();p.add_argument("--base",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();out=a.out.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    hashes={}
    for src in (Path(__file__),Path(sweep.__file__)):
        dst=out/src.name;shutil.copyfile(src,dst);assert sha(src)==sha(dst)
        hashes[str(src)]=sha(src)
    metadata=json.loads((a.base/"inputs_snapshot/workload_metadata.json").read_text())
    grid=json.loads((a.base/"baseline_verification/bandwidth_grid.json").read_text())["bytes_per_second"]
    sweep.initialize(metadata,grid)
    try:
        result=sweep.verify_graph_dedup(metadata)
        direct_hist=Counter();direct_checks=[]
        jobs=list(trace_jobs(metadata,1))
        for job in jobs:
            hist,checks=sweep.trace_job(job)
            direct_hist.update(dict(hist));direct_checks.extend(checks)
        with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context("fork")) as pool:
            hist,batches,operators=sweep.run_graphs(pool,metadata,out,jobs=jobs)
        assert hist==direct_hist
        got=list(rows(out/"trace_batch_checks.csv"))
        assert got==[{k:str(v) for k,v in r.items()} for r in direct_checks]
        result.update(process_pool_inventory_and_readback="PASS",model_batches=batches,distinct_operators=operators)
    except Exception as exc:
        result=dict(status="FAIL",error=f"{type(exc).__name__}: {exc}")
        (out/"failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        result["source_sha256"]=hashes
        (out/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    assert json.loads((out/"verification.json").read_text())==result
    print(json.dumps(result))


if __name__=="__main__":main()
