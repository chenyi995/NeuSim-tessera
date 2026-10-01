"""Freeze references and compare ported baseline kernels against original code."""
import argparse
import ast
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import random
import shutil
import sys
import types

from neusim.run_scripts.compare_tessera_native import ROOT, sha
from neusim.run_scripts.run_tessera_partitioned import config
from neusim.npusim.backend.tessera_baselines import ARCHITECTURES, array_cost, geometries
from neusim.npusim.frontend.llm_ops_lib import create_einsum_op, create_multi_head_flash_attention_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info


def main():
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();out=args.out.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    out.mkdir(parents=True)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    fission=ROOT.parents[1]/"FissionSA"
    hashes={}
    for src in (fission/"fissionsa").rglob("*.py"):
        dst=out/"reference"/src.relative_to(fission)
        dst.parent.mkdir(parents=True,exist_ok=True)
        before=sha(src);shutil.copyfile(src,dst);assert sha(dst)==sha(src)==before
        hashes[str(dst)]=before
    for relative in ("out/tessera-hpca/Batch6-BW-Sweep/scripts/run_batch6.py",
                     "Tessera-revision/experiments/planaria_native/run_native_workloads.py"):
        src=fission/relative;dst=out/Path(relative).name
        before=sha(src);shutil.copyfile(src,dst);assert sha(dst)==sha(src)==before
        hashes[str(dst)]=before
    # Load only frozen reference modules, not their package's broad re-exports.
    for name, directory in (("fissionsa",out/"reference/fissionsa"),
                            ("fissionsa.modes",out/"reference/fissionsa/modes")):
        pkg=types.ModuleType(name);pkg.__path__=[str(directory)];sys.modules[name]=pkg
    from fissionsa.modes import combo0,combo1,combo4,combo10
    from fissionsa.workload import GEMMLayer
    combo0._PLANARIA_D,combo0._PLANARIA_M=32,4
    rng=random.Random(29)
    shapes=[(m,n,k) for m in (1,31,127,128,129,257) for n,k in ((1,1),(63,65),(128,128),(257,193))]
    shapes.extend(tuple(rng.randrange(1,700) for _ in range(3)) for _ in range(32))
    checked=0
    for arch in ARCHITECTURES:
        chip=config("full")
        chip.tessera_parameters["baseline_architecture"]=arch
        for geometry in geometries(chip):
            if arch=="Planaria-32":
                a,b=geometry[0]//32,geometry[1]//32
                combo0._planaria_select_ab=lambda m,n,k:(a,b)
            for m,n,k in shapes:
                layer=GEMMLayer(0,m,n,k)
                if arch=="Planaria-32":ref=combo0.simulate_layer_planaria_grid(layer,32,4,0)
                elif arch=="WS":ref=combo4.simulate_layer_ws_disconnected_grid(layer,128,1,0)
                elif arch=="SOSA":ref=combo4.simulate_layer_ws_disconnected_grid(layer,32,4,0)
                elif arch=="FlexSA":ref=combo1.simulate_layer_ws_connected_grid(layer,64,2,dram_bw=0)
                else:ref=combo10.simulate_layer_os_sisa(layer,128,16,0)
                got=array_cost(arch,m,n,k,geometry)
                assert (got[0],got[1]+got[2],got[3])==(ref.compute_cycles,ref.input_words,ref.output_words),(arch,m,n,k,geometry,got,ref)
                checked+=1
    # Independent source extraction: preserve the established scan points/units.
    tree=ast.parse((out/"run_batch6.py").read_text())
    values={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ("BW_GRID","FREQ"):
                    values[target.id]=ast.literal_eval(node.value)
    scan=dict(source=str(out/"run_batch6.py"),source_fields=["BW_GRID","FREQ"],
              words_per_cycle=values["BW_GRID"],frequency_hz=values["FREQ"],
              bytes_per_word=2,bytes_per_second=[x*2*values["FREQ"] for x in values["BW_GRID"]])
    (out/"bandwidth_grid.json").write_text(json.dumps(scan,indent=2)+"\n")
    assert json.loads((out/"bandwidth_grid.json").read_text())==scan
    integration=0
    for arch in ARCHITECTURES:
        chip=config("full");chip.tessera_parameters["baseline_architecture"]=arch
        for op in (create_einsum_op([13,257],[257,193],"MK;KN->MN"),
                   create_multi_head_flash_attention_op([1,7,8,64],[1,145,2,64],[1,145,2,64])):
            with redirect_stdout(StringIO()):result=fill_operators_execution_info([op],chip)[0]
            st=result.stats
            assert st.sa_time_ns>0
            assert st.execution_time_ns==max(st.sa_time_ns,st.vu_time_ns,st.memory_time_ns,st.ici_time_ns)
            assert st.tessera_details["hbm_bytes"]==st.memory_traffic_bytes
            assert st.vmem_time_ns<=st.execution_time_ns
            integration+=1
    shutil.copyfile(__file__,out/Path(__file__).name)
    hashes[str(out/Path(__file__).name)]=sha(Path(__file__))
    record=dict(status="PASS",array_kernel_checks=checked,native_pipeline_checks=integration,
                reference_hashes=hashes,selection="Native NeuSim EDP at HBM4; fixed mapping during bandwidth scan",
                accounting="Only array cycles/useful words ported; native capacity, HBM and frontend retained")
    (out/"verification.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({k:v for k,v in record.items() if k!="reference_hashes"}))


if __name__=="__main__":main()
