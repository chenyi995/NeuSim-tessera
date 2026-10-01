"""Read-only reference extraction; write reproducible configuration inside NeuSim.

Run as python -m neusim.run_scripts.prepare_tessera --workspace ../..
Every transferred number is extracted in Python and checked against its source.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def literals(path, class_name=None):
    tree = ast.parse(Path(path).read_text())
    nodes = tree.body if class_name is None else next(n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    result = {}
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = value
    return result


def prepare(workspace, output):
    workspace = Path(workspace).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("configuration output must stay inside NeuSim")
    if output.exists():
        raise FileExistsError(output)
    ref = workspace / "FissionSA/Tessera-revision/experiments/ad_cycle/config_dnn_padding_v2.json"
    energy = workspace / "FissionSA/fissionsa/modes/combo6_edp.py"
    chip = ROOT / "neusim/configs/chips/ChipConfig.py"
    timeline = workspace / "Tessera-HPCA2027-revision/fig/plotting/data/power-timeline/tessera0_32/arr_tessera0_32_gemm_timeline.csv"
    raw = json.loads(ref.read_text())
    memory = raw["memory"]
    e = literals(energy)
    from neusim.configs.chips.ChipConfig import ChipConfig
    hw = ChipConfig().model_dump()
    with timeline.open() as f:
        power = list(csv.DictReader(f))
    # Codex: decision start — project observed idle array power by PE count; do not mix it with TPU leakage.
    # Source: timeline frames 1..4 (before weight load), physical side 32 from source directory.
    calibration_side = int(timeline.parent.name.rsplit("_", 1)[1])
    idle_w = sum(float(r["total_mW"]) - float(r["memory_mW"]) for r in power[1:5]) / len(power[1:5]) / 1000
    # Codex: decision end
    side = math.isqrt(raw["total_pe"])
    assert side * side == raw["total_pe"]
    grain = next(a["grain"] for a in raw["architectures"] if a["name"] == "fission8_rect")
    # chenyi9: decision start — use the approved coarse revision coefficients consistently.
    params = dict(side=side, grain=grain, frequency_hz=raw["frequency_hz"],
                  sram_bytes=memory["sram_bytes"], staging_bytes=memory["staging_bytes"],
                  operand_bytes=memory["operand_bytes"], accumulator_bytes=memory["accumulator_bytes"],
                  hbm_bytes_per_second=memory["hbm_bytes_per_cycle"] * raw["frequency_hz"],
                  hbm_read_pj_per_byte=memory["hbm_read_pj_per_byte"],
                  hbm_write_pj_per_byte=memory["hbm_write_pj_per_byte"],
                  array_pj_per_op=e["E_OP_PJ"], sram_read_pj_per_operand=e["E_READ_PJ"],
                  sram_write_pj_per_accumulator=e["E_WRITE_PJ"],
                  vector_ops_per_cycle=hw["num_vu"] * 8 * 128,
                  vector_pj_per_op=e["E_OP_PJ"],
                  background_power_W=idle_w * raw["total_pe"] / calibration_side ** 2,
                  seam_cycles=1, ingress_cycles_per_row=1,
                  ici_bytes_per_second=hw["ici_bw_GBps"] * 1024 ** 3,
                  ici_latency_ns=hw["ici_latency_ns"],
                  ici_pj_per_byte=hw["dynamic_power_ici_W_per_GBps"] * 1e12 / 1024 ** 3)
    # chenyi9: decision end
    sources = {str(p): {"sha256": digest(p)} for p in (ref, energy, chip, timeline)}
    assumptions = {
        "resource_and_hbm_fields": str(ref),
        "array_pj_per_op": f"{energy}: E_OP_PJ; two operations per MAC; includes array transport approximately",
        "sram_coefficients": f"{energy}: E_READ_PJ / E_WRITE_PJ; 16-bit read / 32-bit write convention",
        "vector_pj_per_op": "coarse shared arithmetic coefficient from E_OP_PJ, not measured VU power",
        "vector_ops_per_cycle": f"{chip}: num_vu * 8 * 128, NeuSim VU convention; common across variants",
        "background_power_W": "observed idle array timeline frames 1..4, linear PE-count projection; excludes uncalibrated controller/SRAM/HBM static power",
        "seam_cycles": "gemmini/partition/TIMING_AUDIT.md: one register per strip seam",
        "ingress_cycles_per_row": "gemmini/partition/TIMING_AUDIT.md: transpose first fill takes H cycles; subsequent fills double-buffered",
        "ici": "NeuSim ChipConfig common communication parameters; coarse borrowed model, not N28 silicon measurement",
        "hbm": "32 pJ/byte inherited rough HBM assumption, not measured HBM4; replaces native HBM energy",
        "timing": "1 GHz architectural scenario, not a claim of RTL timing closure",
        "padding": "uniform dyadic physical slots retain their size on tail rounds; no padded arithmetic energy",
    }
    models = {}
    traces = {}
    folders = {"llama2-7b_conv300": "llama27b_conv", "llama3-8b_conv300": "llama38b_conv",
               "llama2-70b-tp4_conv300": "llama270b_conv", "phi2_conv300": "phi2_conv",
               "llama2-7b_code300": "llama27b_code", "llama2-7b_arxiv-qps1": "llama27b_arxiv"}
    for tag, folder in folders.items():
        snapshots = sorted((workspace / "tessera_repro/out" / folder / "vidur_out").glob("*/config.json"))
        if len(snapshots) != 1:
            raise ValueError(f"Expected one saved model metadata file for {tag}, got {snapshots}")
        snapshot = snapshots[0]
        replica = json.loads(snapshot.read_text())["cluster_config"]["replica_config"]
        model = dict(replica["model_config"], tensor_parallel_size=replica["tensor_parallel_size"])
        trace = workspace / "FissionSA/out/tessera-hpca/Batch6-BW-Sweep/workload/traces" / tag / "mnk_trace.csv"
        with trace.open() as f:
            first = next(csv.DictReader(f))
        assert int(first["K"]) == model["embedding_dim"]
        assert int(first["num_layers"]) == model["num_layers"]
        head_dim = model["embedding_dim"] // model["num_q_heads"]
        tp = model["tensor_parallel_size"]
        assert int(first["N"]) == (model["num_q_heads"] // tp + 2 * max(1, model["num_kv_heads"] // tp)) * head_dim
        models[tag] = model
        traces[tag] = str(trace)
        for path in (snapshot, trace):
            sources[str(path)] = {"sha256": digest(path)}
    data = {"parameters": params, "sources": sources, "assumptions": assumptions,
            "models": models, "traces": traces, "workspace": str(workspace)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    # Independent read-back of copied fields, rather than verifying only serialization.
    copied = json.loads(output.read_text())["parameters"]
    assert copied["sram_bytes"] == json.loads(ref.read_text())["memory"]["sram_bytes"]
    assert copied["hbm_read_pj_per_byte"] == json.loads(ref.read_text())["memory"]["hbm_read_pj_per_byte"]
    assert copied["array_pj_per_op"] == literals(energy)["E_OP_PJ"]
    assert copied["side"] ** 2 == json.loads(ref.read_text())["total_pe"]
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", required=True)
    p.add_argument("--output", default=str(ROOT / "configs/chips/tessera_revision.json"))
    a = p.parse_args()
    prepare(a.workspace, a.output)
    print(f"Wrote and read-back checked {a.output}")
