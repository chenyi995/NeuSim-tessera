"""Reference DNN layer graphs and weighted revision operator workloads."""
from __future__ import annotations

from collections import defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import sys

from neusim.npusim.frontend.tessera_workloads import make_op
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
from neusim.run_scripts.run_tessera import VARIANTS, aggregate, chip_config, digest, sum_rows, write_csv


def load_networks(workspace):
    path = Path(workspace) / "planaria.code"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("tessera_reference_benchmarks", path / "src/benchmarks/benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    networks = []
    for name in module.benchlist:
        nn = module.get_bench_nn(name)
        networks.extend(nn if isinstance(nn, tuple) else (nn,))
    return networks


def dnn_graph(nn, operand_bytes, fixed_mapping=False):
    """Execute all layers present in the reference graph, at batch one.

    Implicit im2col supplies array operand reads without materializing a DRAM
    matrix. This is a coarse convolution lowering, not a convolution frontend
    timing model. The reference is a layer graph, not an exported framework graph.
    """
    from nn_dataflow.Layer import ConvLayer, DWConvLayer, PoolingLayer
    sizes = {name: layer.total_ofmap_size(operand_bytes) for name, layer in nn.layer_dict.items()}
    ops, coverage = [], []
    consumers = {p for prevs in nn.prevs_dict.values() for p in prevs}
    for name in nn:
        layer = nn[name]
        prevs, merge = nn.prev_layers(name)
        inputs = {p: sizes[p] for p in prevs}
        if len(prevs) > 1:
            merged = name + "/merged"
            elements = (sum(inputs.values()) if merge == "|" else next(iter(inputs.values()))) // operand_bytes
            ops.append(make_op(merged, dict(kind="vector", inputs=inputs, outputs={merged: elements * operand_bytes},
                vector_ops=elements * (len(prevs)-1) if merge == "+" else 0)))
            inputs = {merged: elements * operand_bytes}
        terminal = name not in consumers
        if isinstance(layer, ConvLayer):
            depthwise = isinstance(layer, DWConvLayer)
            m = layer.hofm * layer.wofm
            n = 1 if depthwise else layer.nofm
            k = layer.sfil ** 2 * (1 if depthwise else layer.nifm)
            batch = layer.nofm if depthwise else 1
            inputs[name + "/weight"] = batch * n * k * operand_bytes
            raw = name + "/raw"
            spec = dict(kind="gemm", inputs=inputs, outputs={raw: sizes[name]}, M=m, N=n, K=k, batch=batch)
            if fixed_mapping:
                spec["fixed_mapping_from"] = "tessera"
            ops.append(make_op(name + "/conv", spec))
            # Reference Layer.py::get_act_ops supplies one scalar activation per output.
            ops.append(make_op(name + "/activation", dict(kind="vector", inputs={raw: sizes[name]},
                outputs={name: sizes[name]}, vector_ops=layer.get_act_ops(), final=terminal)))
            macs = m * n * k * batch
        elif isinstance(layer, PoolingLayer):
            ops.append(make_op(name, dict(kind="vector", inputs=inputs, outputs={name: sizes[name]},
                vector_ops=layer.get_pool_ops(), final=terminal)))
            macs = 0
        else:
            raise NotImplementedError(type(layer).__name__)
        coverage.append(dict(network=nn.net_name, layer=name, type=type(layer).__name__,
            predecessors=";".join(prevs), merge=merge, useful_macs=macs,
            input_bytes=sum(inputs.values()), output_bytes=sizes[name]))
    return ops, coverage


def dnn(config, c, out):
    results, all_coverage = [], []
    workspace = Path(config["workspace"])
    files = [workspace / "planaria.code" / p for p in
             ("src/benchmarks/benchmarks.py", "nn_dataflow/Layer.py", "nn_dataflow/Network.py")]
    (out / "supplement_sources.json").write_text(json.dumps({str(p): digest(p) for p in files}, indent=2) + "\n")
    for nn in load_networks(workspace):
        model = dict(name=nn.net_name, tensor_parallel_size=1)
        for variant in VARIANTS:
            ops, coverage = dnn_graph(nn, c.operand_bytes, variant in ("square_fixed", "skew"))
            fill_operators_execution_info(ops, chip_config(c, variant))
            row = aggregate(ops, model, nn.net_name, variant)
            row["accounting"] = "reference_dnn_layer_graph"
            assert row["useful_macs"] == sum(r["useful_macs"] for r in coverage)
            results.append(row)
            write_csv(out / f"{nn.net_name}__{variant}__operators.csv",
                      [dict(op.stats.tessera_details, name=op.name, count=op.stats.count) for op in ops])
            if variant == "tessera":
                all_coverage.extend(coverage)
        print(f"Completed DNN reference graph {nn.net_name}", flush=True)
    write_csv(out / "graph_coverage.csv", all_coverage)
    with (out / "graph_coverage.csv").open() as f:
        restored = list(csv.DictReader(f))
    assert sum(int(r["useful_macs"]) for r in restored) == sum(r["useful_macs"] for r in all_coverage)
    return results


def operators(config, c, out):
    source = Path(config["workspace"]) / "FissionSA/Tessera-revision/experiments/ad_cycle/inputs/workloads.csv"
    (out / "supplement_sources.json").write_text(json.dumps({str(source): digest(source)}, indent=2) + "\n")
    grouped = defaultdict(lambda: dict(repeat=0, source_records=0))
    with source.open() as f:
        for row in csv.DictReader(f):
            key = (row["dataset"], row["cohort"], row["phase"], *(int(row[k]) for k in ("M", "N", "K")))
            grouped[key]["repeat"] += int(row["repeat"])
            grouped[key]["source_records"] += int(row["source_records"])
    results = []
    aggregates = defaultdict(list)
    model = dict(name="revision_operator_workloads", tensor_parallel_size=1)
    # Codex: decision start — repeat means independent occurrences, not concurrent GEMM batches.
    path = out / "weighted_operators.csv"
    with path.open("x", newline="") as f:
        writer = None
        for index, ((dataset, cohort, phase, m, n, k), meta) in enumerate(grouped.items()):
            case = f"{dataset}__{cohort}__{phase}"
            for variant in VARIANTS:
                spec = dict(kind="gemm", inputs={"A": m*k*c.operand_bytes, "B": k*n*c.operand_bytes},
                    outputs={"C": m*n*c.operand_bytes}, final=True, M=m, N=n, K=k, batch=1)
                if variant in ("skew", "square_fixed"):
                    spec["fixed_mapping_from"] = "tessera"
                op = make_op(case, spec, meta["repeat"])
                fill_operators_execution_info([op], chip_config(c, variant))
                row = aggregate([op], model, case, variant, M=m, N=n, K=k, **meta)
                row["accounting"] = "weighted_standalone_operators"
                assert row["useful_macs"] == meta["repeat"] * m * n * k
                if writer is None:
                    writer = csv.DictWriter(f, list(row))
                    writer.writeheader()
                writer.writerow(row)
                aggregates[(dataset, variant)].append(row)
            if (index + 1) % 1000 == 0:
                print(f"Operators: {index+1}/{len(grouped)} distinct cohort shapes", flush=True)
    # Codex: decision end
    for (dataset, variant), rows in aggregates.items():
        results.append(sum_rows(rows, model, dataset, variant, "weighted_standalone_operators"))
    with path.open() as f:
        actual_macs = sum(int(r["useful_macs"]) for r in csv.DictReader(f))
    expected_macs = sum(key[-3]*key[-2]*key[-1]*v["repeat"] for key,v in grouped.items()) * len(VARIANTS)
    assert actual_macs == expected_macs
    (out / "operator_checks.json").write_text(json.dumps(dict(readback="PASS", useful_macs=actual_macs,
        distinct_cohort_shapes=len(grouped)), indent=2) + "\n")
    return results


def supplemental(config, c, out, suite):
    return dnn(config, c, out) if suite == "dnn" else operators(config, c, out)
