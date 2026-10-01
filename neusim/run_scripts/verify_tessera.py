"""Independently read back completed runs and derive a reviewable delivery report."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ("array", "sram", "hbm", "vector", "ici", "background")


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-14), (a, b)


def verify(run):
    rows = read(run / "summary.csv")
    complete = json.loads((run / "completed.json").read_text())
    assert complete["status"] == "PASS" and len(rows) == complete["rows"]
    groups = {}
    for r in rows:
        total = sum(float(r[f"{component}_energy_J"]) for component in COMPONENTS)
        close(total, float(r["energy_J"]))
        close(total * int(r["chips"]), float(r["system_energy_J"]))
        close(total * int(r["chips"]) * float(r["seconds"]), float(r["edp_J_s"]))
        groups.setdefault(r["case"], {})[r["variant"]] = r
    for g in groups.values():
        ref = g["tessera"]
        assert len(g) == 6
        for v, r in g.items():
            assert int(r["useful_macs"]) == int(ref["useful_macs"])
            for metric, field in (("latency", "seconds"), ("energy", "system_energy_J"), ("edp", "edp_J_s")):
                close(float(r[f"{metric}_over_tessera"]), float(r[field]) / float(ref[field]))
        for v in ("skew", "square_fixed"):
            for key in ("array_energy_J", "sram_energy_J", "vector_energy_J", "hbm_energy_J", "ici_energy_J"):
                close(float(g[v][key]), float(ref[key]))
            assert float(g[v]["seconds"]) >= float(ref["seconds"]) - 1e-12
    return rows, complete


def verify_trace_batches(run, rows):
    config = json.loads((run / "configuration.json").read_text())
    checks = {}
    for tag, source in config["traces"].items():
        model = config["models"][tag]
        expected = {}
        # Independent grouping: do not call the simulation runner's batch reader.
        with Path(source).open() as f:
            for r in csv.DictReader(f):
                bid = int(r["batch_id"])
                value = expected.setdefault(bid, dict(macs=0, requests=0, tokens=0))
                value["macs"] += math.prod(int(r[k]) for k in ("M", "N", "K", "gemm_batch", "num_layers"))
                if r["op"] == "attn_qk":
                    value["requests"] += 1
                    value["tokens"] += int(r["M"])
        for value in expected.values():
            tp = model["tensor_parallel_size"]
            value["macs"] += value["requests"] * model["embedding_dim"] * ((model["vocab_size"] + tp - 1) // tp)
        for summary in (r for r in rows if r["case"] == tag):
            batch_rows = read(run / f"{tag}__{summary['variant']}__batches.csv")
            assert len(batch_rows) == len(expected)
            assert {int(r["batch_id"]) for r in batch_rows} == set(expected)
            for r in batch_rows:
                golden = expected[int(r["batch_id"])]
                assert int(r["useful_macs"]) == golden["macs"]
                assert int(r["query_tokens"]) == golden["tokens"]
                assert int(r["requests"]) == golden["requests"]
                close(sum(float(r[f"{key}_energy_J"]) for key in COMPONENTS), float(r["energy_J"]))
            for key in ("seconds", *(f"{c}_energy_J" for c in COMPONENTS)):
                close(math.fsum(float(r[key]) for r in batch_rows), float(summary[key]))
            for key in ("useful_macs", "hbm_read_bytes", "hbm_write_bytes"):
                assert sum(int(r[key]) for r in batch_rows) == int(summary[key])
        checks[tag] = dict(batches=len(expected), useful_macs=sum(r["macs"] for r in expected.values()))
    return checks


def main():
    p = argparse.ArgumentParser()
    for name in ("pilot", "traces", "dnn", "operators", "sensitivity"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
    out = ROOT / "results/tessera" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_review")
    out.mkdir(parents=True, exist_ok=False)
    runs, metadata = {}, {}
    for name, run in vars(args).items():
        rows, complete = verify(run)
        runs[name] = rows
        metadata[name] = dict(directory=str(run), completed=complete,
            summary_sha256=hashlib.sha256((run / "summary.csv").read_bytes()).hexdigest())
    coverage = verify_trace_batches(args.traces, runs["traces"])
    summaries = [r for r in runs["traces"] if r["accounting"] == "fixed_trace_cumulative_service"]
    groups = {tag: {r["variant"]: r for r in summaries if r["case"] == tag} for tag in coverage}
    facts = []
    for tag, group in groups.items():
        tes = group["tessera"]
        row = dict(trace=tag, batches=coverage[tag]["batches"],
            service_seconds=float(tes["seconds"]), system_energy_J=float(tes["system_energy_J"]),
            speedup_vs_ws=float(group["ws"]["seconds"]) / float(tes["seconds"]),
            energy_saving_vs_ws_percent=(1 - float(tes["system_energy_J"]) / float(group["ws"]["system_energy_J"])) * 100)
        for component in COMPONENTS:
            row[component + "_energy_percent"] = float(tes[component + "_energy_J"]) / float(tes["energy_J"]) * 100
        for variant in ("square", "square_fixed", "independent", "skew"):
            for metric in ("latency", "energy", "edp"):
                row[variant + "_" + metric + "_over_tessera"] = float(group[variant][metric + "_over_tessera"])
        facts.append(row)
    with (out / "facts.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, list(facts[0]))
        writer.writeheader()
        writer.writerows(facts)
    restored = read(out / "facts.csv")
    for original, copied in zip(facts, restored):
        for k, v in original.items():
            assert copied[k] == v if isinstance(v, str) else float(copied[k]) == v
    checks = dict(status="PASS", run_metadata=metadata, trace_coverage=coverage,
                  energy_sum="PASS", paired_dynamic_energy="PASS", batch_source_macs="PASS", numeric_readback="PASS")
    (out / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    lines = ["# NeuSim Tessera experiment review", "",
        "These are coarse analytical estimates from NeuSim's Tessera backend. Fixed traces preserve the source batch compositions.",
        "Service time sums model-step execution and excludes arrival gaps, queueing, and host scheduling. It is not TTFT or request latency.", "",
        "## Complete fixed-trace replay", "",
        "| Trace | Batches | Tessera service (s) | System energy (J) | Speedup vs WS | Energy saving vs WS (%) |",
        "|---|---:|---:|---:|---:|---:|"]
    for r in facts:
        lines.append(f"| {r['trace']} | {r['batches']} | {r['service_seconds']:.6g} | {r['system_energy_J']:.6g} | {r['speedup_vs_ws']:.5g} | {r['energy_saving_vs_ws_percent']:.5g} |")
    lines += ["", "## Tessera energy breakdown", "", "All shares are derived from per-chip energy; identical TP ranks have the same shares.", "",
        "| Trace | Array (%) | SRAM (%) | HBM (%) | Vector (%) | Inter-chip (%) | Background (%) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in facts:
        lines.append("| " + r["trace"] + " | " + " | ".join(f"{r[c+'_energy_percent']:.5g}" for c in COMPONENTS) + " |")
    for metric in ("latency", "energy", "edp"):
        lines += ["", f"## Ablation {metric} / Tessera", "",
            "| Trace | Square optimized | Square fixed logical tiles | Independent arrays | Restored skew |", "|---|---:|---:|---:|---:|"]
        for r in facts:
            lines.append("| " + r["trace"] + " | " + " | ".join(f"{r[v+'_'+metric+'_over_tessera']:.7g}" for v in ("square", "square_fixed", "independent", "skew")) + " |")
    lines += ["", "## Interpretation boundaries", "",
        "- Square optimized reselects logical tiles; square_fixed isolates padding while keeping rectangular logical tiles. The latter can amplify fragmentation and is not a competitive square-array mapper.",
        "- Skew and square_fixed preserve arithmetic, SRAM, HBM and inter-chip dynamic energy; any modeled energy difference comes from duration-dependent array background.",
        "- Independent arrays share SRAM and the vector reduction unit, but have no PE-to-PE links between arrays. This also uses skew-free small arrays.",
        "- Array and vector energy use inherited arithmetic coefficients. Skew-register and link switching energy are not separately calibrated.",
        "- DNN results cover the reference layer definitions with pooling, merges and scalar activations, using implicit im2col and channelwise depthwise convolution. They are not exported framework execution traces.",
        "- Revision operator results are weighted standalone kernels and cannot be reported as whole-model E2E results.",
        "- See docs/tessera.md and each configuration.json for the full accounting and sources.", "", "## Retained runs", ""]
    for name, run in vars(args).items():
        lines.append(f"- [{name}]({os.path.relpath(run / 'REPORT.md', out)}): completed and independently read-back checked.")
    lines += ["", "[Energy breakdown](../" + args.traces.name + "/figures/energy_breakdown.pdf)",
              "[Non-square](../" + args.traces.name + "/figures/non_square.pdf)",
              "[Independent arrays](../" + args.traces.name + "/figures/independent_arrays.pdf)",
              "[Skew](../" + args.traces.name + "/figures/skew.pdf)", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(out), status="PASS", traces=len(coverage),
                         batches=sum(c["batches"] for c in coverage.values()), facts=facts), indent=2))


if __name__ == "__main__":
    main()
