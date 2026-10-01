"""Run append-only Tessera analytical experiments through NeuSim's frontend."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

# Codex: decision start — prevent numerical libraries from exceeding the approved CPU budget.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Codex: decision end

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.tessera import TesseraConfig, gemm_cost
from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
from neusim.npusim.frontend.tessera_workloads import llm_graph, make_op

ROOT = Path(__file__).resolve().parents[2]
ENERGIES = ("array_energy_J", "sram_energy_J", "hbm_energy_J", "vector_energy_J",
            "ici_energy_J", "background_energy_J")
VARIANTS = ("tessera", "square", "square_fixed", "independent", "skew", "ws")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        writer.writerows(rows)


def trace_batches(path):
    current, rows = None, []
    with Path(path).open() as f:
        for row in csv.DictReader(f):
            bid = int(row["batch_id"])
            if current is not None and bid != current:
                yield current, rows
                rows = []
            current = bid
            rows.append(row)
    if rows:
        yield current, rows


def requests_from_batch(rows):
    requests = [(int(r["M"]), int(r["N"])) for r in rows if r["op"] == "attn_qk"]
    qkv = next(r for r in rows if r["op"] == "qkv_proj")
    if sum(q for q, _ in requests) != int(qkv["M"]):
        raise ValueError("request tokens disagree with the batch projection")
    return requests


def chip_config(c, variant):
    return ChipConfig(array_backend="tessera", tessera_variant="square" if variant == "square_fixed" else variant,
                      tessera_parameters=asdict(c), use_vu_for_small_matmul=False,
                      num_sa=1, sa_dim=c.side, freq_GHz=c.frequency_hz / 1e9)


def aggregate(ops, model, case, variant, **extra):
    stats = Counter()
    for op in ops:
        d = op.stats.tessera_details
        for key in (*ENERGIES, "seconds", "hbm_read_bytes", "hbm_write_bytes",
                    "sram_read_bytes", "sram_write_bytes", "useful_macs",
                    "array_cycles", "skew_extra_cycles", "seam_tail_cycles"):
            stats[key] += d.get(key, 0) * op.stats.count
        if not abs(op.stats.total_energy_J - d["energy_J"]) <= max(1e-18, d["energy_J"] * 1e-12):
            raise AssertionError("NeuSim operator energy and breakdown disagree")
    row = dict(case=case, variant=variant, model=model["name"], accounting="analytical_model_step",
               chips=model["tensor_parallel_size"], **dict(stats), **extra)
    row["energy_J"] = sum(row[k] for k in ENERGIES)
    row["system_energy_J"] = row["energy_J"] * row["chips"]
    row["edp_J_s"] = row["system_energy_J"] * row["seconds"]
    row["latency_ms"] = row["seconds"] * 1000
    return row


def run_case(model, requests, c, case, variant, out=None, *, layers=None):
    ops = llm_graph(model, requests, c.operand_bytes, layers=layers,
                    fixed_mapping=variant in ("skew", "square_fixed"), compact_layers=out is None)
    fill_operators_execution_info(ops, chip_config(c, variant))
    row = aggregate(ops, model, case, variant, query_tokens=sum(q for q, _ in requests),
                    requests=len(requests), operators=len(ops),
                    hbm_bytes_per_second=c.hbm_bytes_per_second,
                    hbm_pj_per_byte=c.hbm_read_pj_per_byte)
    if out:
        records = [dict(op.stats.tessera_details, case=case, variant=variant,
                        name=op.name, count=op.stats.count) for op in ops]
        write_csv(out / f"{case}__{variant}__operators.csv", records)
    return row


def add_ratios(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["case"], {})[row["variant"]] = row
    for group in groups.values():
        if "tessera" not in group:
            continue
        tes = group["tessera"]
        for variant, row in group.items():
            row["latency_over_tessera"] = row["seconds"] / tes["seconds"]
            row["energy_over_tessera"] = row["system_energy_J"] / tes["system_energy_J"]
            row["edp_over_tessera"] = row["edp_J_s"] / tes["edp_J_s"]
            if "ws" in group:
                row["speedup_vs_ws"] = group["ws"]["seconds"] / row["seconds"]


def pilot(config, c, out):
    results = []
    for tag in ("llama3-8b_conv300", "phi2_conv300"):
        selected = {}
        for bid, rows in trace_batches(config["traces"][tag]):
            reqs = requests_from_batch(rows)
            phase = "prefill" if all(q > 1 for q, _ in reqs) else "decode" if all(q == 1 for q, _ in reqs) else "mixed"
            if phase not in selected and phase in ("prefill", "decode"):
                selected[phase] = (bid, reqs)
            if len(selected) == 2:
                break
        for phase, (bid, reqs) in selected.items():
            case = f"{tag}_{phase}_batch{bid}"
            for variant in VARIANTS:
                row = run_case(config["models"][tag], reqs, c, case, variant, out)
                results.append(row)
                print(json.dumps({k: row[k] for k in ("case", "variant", "latency_ms", "system_energy_J")}), flush=True)
    return results


def sensitivity(config, c, out):
    tag = "llama3-8b_conv300"
    choices = []
    for bid, rows in trace_batches(config["traces"][tag]):
        reqs = requests_from_batch(rows)
        if not choices or all(q == 1 for q, _ in reqs):
            choices.append((bid, reqs))
        if len(choices) == 2:
            break
    results = []
    # Reference bandwidths: paper methodology, 800 GiB/s, 1.2 TB/s, 2.8 TB/s.
    # Energy factors are the user-approved 0.5x/1x/2x sensitivity, not measurements.
    for bw in (800 * 1024 ** 3, 1.2e12, c.hbm_bytes_per_second):
        for scale in (0.5, 1.0, 2.0):
            cc = replace(c, hbm_bytes_per_second=bw, hbm_read_pj_per_byte=c.hbm_read_pj_per_byte * scale,
                         hbm_write_pj_per_byte=c.hbm_write_pj_per_byte * scale)
            for bid, reqs in choices:
                case = f"{tag}_batch{bid}_bw{bw:g}_e{scale:g}"
                for variant in VARIANTS:
                    results.append(run_case(config["models"][tag], reqs, cc, case, variant))
            print(f"Completed sensitivity bandwidth={bw} energy_scale={scale}", flush=True)
    return results


SUM_FIELDS = (*ENERGIES, "seconds", "hbm_read_bytes", "hbm_write_bytes",
              "sram_read_bytes", "sram_write_bytes", "useful_macs", "array_cycles",
              "skew_extra_cycles", "seam_tail_cycles", "query_tokens", "requests")


def sum_rows(rows, model, case, variant, accounting):
    values = {key: sum(r.get(key, 0) for r in rows) for key in SUM_FIELDS}
    values.update(case=case, variant=variant, model=model["name"], chips=model["tensor_parallel_size"],
                  accounting=accounting, batches=len(rows))
    values["energy_J"] = sum(values[k] for k in ENERGIES)
    values["system_energy_J"] = values["energy_J"] * values["chips"]
    values["edp_J_s"] = values["system_energy_J"] * values["seconds"]
    values["latency_ms"] = values["seconds"] * 1000
    return values


def trace_worker(config, c, out, tag, variant):
    model = config["models"][tag]
    results = []
    source_macs = 0
    path = out / f"{tag}__{variant}__batches.csv"
    with path.open("x", newline="") as f:
        writer = None
        for bid, raw in trace_batches(config["traces"][tag]):
            reqs = requests_from_batch(raw)
            row = run_case(model, reqs, c, f"{tag}_batch{bid}", variant)
            phase = "decode" if all(q == 1 for q, _ in reqs) else "prefill" if all(q > 1 for q, _ in reqs) else "mixed"
            row.update(batch_id=bid, phase=phase)
            # Codex: decision start — cross-check complete-model MACs against the original trace.
            expected = sum(int(r["M"]) * int(r["N"]) * int(r["K"]) * int(r["gemm_batch"])
                           * int(r["num_layers"]) for r in raw)
            expected += len(reqs) * model["embedding_dim"] * ((model["vocab_size"] + model["tensor_parallel_size"] - 1) // model["tensor_parallel_size"])
            assert row["useful_macs"] == expected, (tag, bid, row["useful_macs"], expected)
            source_macs += expected
            # Codex: decision end
            if writer is None:
                writer = csv.DictWriter(f, list(row))
                writer.writeheader()
            writer.writerow(row)
            results.append(row)
            if len(results) % 1000 == 0:
                print(f"Trace {tag} {variant}: {len(results)} batches", flush=True)
    with path.open() as f:
        restored = list(csv.DictReader(f))
    assert len(restored) == len(results)
    assert sum(int(r["useful_macs"]) for r in restored) == source_macs
    for a, b in zip(results, restored):
        assert a["seconds"] == float(b["seconds"])
        assert a["energy_J"] == float(b["energy_J"])
    summaries = [sum_rows(results, model, tag, variant, "fixed_trace_cumulative_service")]
    for phase in ("prefill", "decode", "mixed"):
        subset = [r for r in results if r["phase"] == phase]
        if subset:
            summaries.append(sum_rows(subset, model, f"{tag}__{phase}", variant, "phase_cumulative_service"))
    (out / f"{tag}__{variant}__checks.json").write_text(json.dumps(dict(
        batches=len(results), useful_macs=source_macs, source_mac_check="PASS", batch_readback="PASS",
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), indent=2) + "\n")
    return summaries


def traces(config, c, out, cpus):
    rows = []
    with ProcessPoolExecutor(max_workers=cpus) as pool:
        jobs = {pool.submit(trace_worker, config, c, out, tag, variant): (tag, variant)
                for tag in config["traces"] for variant in VARIANTS}
        for job in as_completed(jobs):
            rows.extend(job.result())
            print(f"Completed trace {jobs[job]}", flush=True)
    rows.sort(key=lambda r: (r["case"], VARIANTS.index(r["variant"])))
    return rows


def report(rows, out, config, suite):
    add_ratios(rows)
    write_csv(out / "summary.csv", rows)
    # Read-back verification of all values written to the principal results table.
    with (out / "summary.csv").open() as f:
        restored = list(csv.DictReader(f))
    assert len(restored) == len(rows)
    for a, b in zip(rows, restored):
        for key in ("seconds", "energy_J", "system_energy_J", "edp_J_s"):
            assert float(b[key]) == a[key], (key, a, b)
        assert abs(sum(a[k] for k in ENERGIES) - a["energy_J"]) <= 1e-15
    lines = ["# Tessera NeuSim analytical results", "", f"Suite: `{suite}`.", "",
             "All latency and energy values below are analytical estimates, not hardware measurements.",
             "Energy is per chip in component columns and summed over TP ranks in system_energy_J.", "",
             "| Case | Variant | Latency (ms) | System energy (J) | EDP / Tessera |",
             "|---|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['case']} | {r['variant']} | {r['latency_ms']:.6g} | {r['system_energy_J']:.6g} | {r.get('edp_over_tessera', 1):.5g} |")
    lines += ["", "## Accounting", "", "- Fixed batch replay excludes arrival gaps and request queueing.",
              "- Uniform dyadic slots; SRAM port bandwidth is unconstrained.",
              "- Fused attention retains scores in SRAM; requests have distinct KV caches.",
              "- Skew uses the Tessera-selected mapping; square_fixed uses identical logical tiles with square allocation.",
              "- Array dynamic energy uses the inherited coarse 6 pJ/op coefficient, two operations per MAC.",
              "- No separate skew-register switching-energy claim; timing changes alter background energy.",
              "- Background is the PE-scaled idle-array projection; uncalibrated SRAM/HBM/controller static power is omitted.",
              "- Current graph charges KV append to HBM and re-reads the appended bytes with the cache.",
              "- Every replay batch produces one logit row per request, including incomplete prefill chunks.",
              "- See configuration.json for parameter provenance and source hashes.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    (out / "readback_checks.json").write_text(json.dumps(dict(rows=len(rows), numeric_readback="PASS", energy_sum="PASS"), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/chips/tessera_revision.json"))
    parser.add_argument("--suite", choices=("pilot", "sensitivity", "traces", "dnn", "operators"), default="pilot")
    parser.add_argument("--output")
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--memory-gb", type=float, required=True)
    args = parser.parse_args()
    if args.cpus < 1 or args.memory_gb <= 0:
        parser.error("resource budgets must be positive")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cpus])
    limit = int(args.memory_gb * 10 ** 9)
    # Codex: decision start — bound the sum of parent and worker address spaces.
    per_process_limit = limit // (args.cpus + 1) if args.suite == "traces" else limit
    resource.setrlimit(resource.RLIMIT_AS, (per_process_limit, per_process_limit))
    # Codex: decision end
    out = Path(args.output).resolve() if args.output else ROOT / "results/tessera" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not out.is_relative_to(ROOT):
        raise ValueError("all run output must remain inside NeuSim")
    out.mkdir(parents=True, exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib")
    config = json.loads(Path(args.config).read_text())
    c = TesseraConfig(**config["parameters"])
    (out / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    source_paths = set(ROOT.glob("neusim/**/*tessera*.py"))
    source_paths.update(ROOT / p for p in ("neusim/configs/chips/ChipConfig.py",
        "neusim/npusim/frontend/Operator.py", "neusim/npusim/frontend/op_analysis_lib.py"))
    sources = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    for p in source_paths:
        dest = out / "source" / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
        assert digest(dest) == digest(p)
    for path, info in config["sources"].items():
        assert digest(path) == info["sha256"], f"source changed: {path}"
    start = time.time()
    (out / "command.json").write_text(json.dumps(dict(argv=sys.argv, cpu_limit=args.cpus,
        memory_limit_bytes=limit, per_process_limit_bytes=per_process_limit,
        python=sys.version, sources=sources), indent=2) + "\n")
    print(f"Output: {out}", flush=True)
    try:
        if args.suite == "traces":
            rows = traces(config, c, out, args.cpus)
        elif args.suite in ("dnn", "operators"):
            from neusim.run_scripts.tessera_supplement import supplemental
            rows = supplemental(config, c, out, args.suite)
        else:
            rows = pilot(config, c, out) if args.suite == "pilot" else sensitivity(config, c, out)
        report(rows, out, config, args.suite)
        from neusim.run_scripts.plot_tessera import plot_results
        plot_results(out)
    except BaseException:
        (out / "failed.txt").write_text(traceback.format_exc())
        raise
    (out / "completed.json").write_text(json.dumps(dict(status="PASS", wall_seconds=time.time()-start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, rows=len(rows)), indent=2) + "\n")
    print(f"Completed and read-back checked {len(rows)} rows: {out}", flush=True)


if __name__ == "__main__":
    main()
