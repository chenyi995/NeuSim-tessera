"""Aggregate exact saved workload weights and verify paired simulator results."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import resource
import shutil

from neusim.run_scripts.compare_tessera_native import (
    ROOT, BASE, METRICS, ENERGY_FIELDS, ARCHS, SHORT, rows, sha, write_csv)

GROUPS = [
    ("flashinfer_attention", "gqa_decode", "GQA decode"),
    ("flashinfer_attention", "single_kv_head_decode", "Single KV-head decode"),
    ("llama2_vidur", "full_trace", "Llama2"),
    ("phi2_vidur", "full_trace", "Phi-2"),
    ("upstream_reuse_prefill", "CacheBlend_samsum", "CacheBlend SAMSum"),
    ("upstream_reuse_prefill", "CacheBlend_wikimqa", "CacheBlend WikiMQA"),
    ("upstream_reuse_prefill", "EPIC_hotpotqa", "EPIC HotpotQA"),
    ("upstream_reuse_prefill", "EPIC_multi_news", "EPIC Multi-News"),
]


def number(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-11, abs_tol=1e-12), (a, b)


def table(records, fields, digits=4):
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in records:
        cells = [f"{row[k]:.{digits}f}" if isinstance(row[k], float) else str(row[k]) for k in fields]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--replayed", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error("use a new output directory inside NeuSim")
    out.mkdir(parents=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    resource.setrlimit(resource.RLIMIT_AS, (6_000_000_000, 6_000_000_000))
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["status"] == "PASS" and manifest["limit"] == 0
    shape_path = args.run / "shape_results.csv"
    assert sha(shape_path) == manifest["output_sha256"][shape_path.name]
    workload_path = args.archive / "snapshot" / BASE / "inputs/workloads.csv"
    assert sha(workload_path) == manifest["source_sha256"][str(workload_path.resolve())]
    allowed = {(d, c): name for d, c, name in GROUPS}
    weights, meta, shapes = defaultdict(Counter), defaultdict(Counter), {}
    for r in rows(workload_path):
        group = r["dataset"], r["cohort"]
        assert group in allowed
        sid, repeat = int(r["shape_id"]), int(r["repeat"])
        assert repeat > 0
        shape = tuple(int(r[k]) for k in ("M", "N", "K"))
        assert sid not in shapes or shapes[sid] == shape
        shapes[sid] = shape
        weights[sid][group] += repeat
        meta[group]["records"] += 1
        meta[group]["logical_gemms"] += repeat
    totals, examples, seen = defaultdict(Counter), defaultdict(list), set()
    variants = None
    shape_count = raw_count = 0
    for sid, records in itertools.groupby(rows(shape_path), key=lambda r: int(r["shape_id"])):
        records = list(records)
        assert sid not in seen
        seen.add(sid)
        shape_count += 1
        by_variant = {r["variant"]: r for r in records}
        assert len(by_variant) == len(records)
        if variants is None:
            variants = set(by_variant)
        assert set(by_variant) == variants
        assert tuple(int(records[0][k]) for k in ("M", "N", "K")) == shapes[sid]
        parsed = {name: {key: number(row[key]) for key in METRICS} for name, row in by_variant.items()}
        for name, stats in parsed.items():
            raw_count += 1
            for group, repeat in weights[sid].items():
                acc = totals[group, name]
                acc["unique_shapes"] += 1
                acc["logical_gemms"] += repeat
                for field, value in stats.items():
                    acc[field] += repeat * value
                for kind in ("sa", "vu", "sram", "hbm"):
                    acc[kind + "_bound_time_ns"] += repeat * stats[kind + "_bound"] * stats["time_ns"]
                acc["fallback_time_ns"] += repeat * stats["fallback"] * stats["time_ns"]
                fixed = by_variant["native_frozen_full"]
                full = by_variant["native_full"]
                changed = any(full[k] != fixed[k] for k in ("kt", "nt", "physical_square"))
                acc["native_full_mapping_changed"] += repeat * changed
        f, n, ind = parsed["fission_full"], parsed["native_full"], parsed["fission_independent32"]
        for group, repeat in weights[sid].items():
            examples[group].append(dict(shape_id=sid, M=shapes[sid][0], N=shapes[sid][1], K=shapes[sid][2], repeat=repeat,
                extra_fission_independent_energy_J=repeat * (ind["energy_J"] - f["energy_J"]),
                fission_hbm_bytes=f["hbm_bytes"], native_hbm_bytes=n["hbm_bytes"],
                fission_time_ns=f["time_ns"], native_pre_time_ns=n["pre_time_ns"], native_time_ns=n["time_ns"],
                native_fallback=n["fallback"], native_pre_sa_ns=n["pre_sa_ns"], native_pre_hbm_ns=n["pre_hbm_ns"]))
    assert seen == set(shapes) and shape_count == manifest["unique_shapes"]
    totals_rows = []
    for d, c, name in GROUPS:
        for variant in sorted(variants):
            stats = totals[(d, c), variant]
            assert stats["logical_gemms"] == meta[d, c]["logical_gemms"]
            totals_rows.append(dict(workload=name, dataset=d, cohort=c, variant=variant,
                                    **stats, edp_J_ns=stats["energy_J"] * stats["time_ns"]))
    write_csv(out / "totals.csv", totals_rows)
    # Codex: decision start — independently cross-check weighted Fission results
    # against the archive tool's aggregation, then read back every emitted CSV field.
    for row in rows(args.replayed / "workload_totals.csv"):
        group = row["dataset"], row["cohort"]
        stat = totals[group, "fission_" + SHORT[row["architecture"]]]
        assert stat["logical_gemms"] == int(row["logical_gemms"])
        assert stat["time_ns"] == int(row["total_cycles"])
        assert stat["hbm_bytes"] == int(row["hbm_read_bytes"]) + int(row["hbm_write_bytes"])
        close(stat["energy_J"], float(row["energy_pj"]) * 1e-12)
    for expected, actual in zip(totals_rows, rows(out / "totals.csv"), strict=True):
        for key in expected:
            assert str(expected[key]) == actual[key]
    # chenyi9 requested repeat-weighted complete collected workloads, not a selected subset.
    ablations, energy, causes, absolute, mapping_rows, legacy = [], [], [], [], [], []
    for d, c, name in GROUPS:
        group = d, c
        get = lambda variant: totals[group, variant]
        f, n = get("fission_full"), get("native_frozen_full")
        record = dict(workload=name)
        for prefix in ("fission", "native", "native_frozen", "native_1GHz", "native_forceSA"):
            base = get(prefix + "_full")
            for variant in ("square", "independent32", "skew"):
                alt = get(prefix + "_" + variant)
                ratio = alt["time_ns"] / base["time_ns"]
                record[prefix + "_" + variant + "_speedup"] = ratio
                record[prefix + "_" + variant + "_edp_ratio"] = ratio * alt["energy_J"] / base["energy_J"]
        ablations.append(record)
        en = dict(workload=name, total_J=n["energy_J"])
        for component in ("sa", "vu", "sram", "hbm", "ici", "other"):
            en[component + "_percent"] = 100 * (n["dynamic_energy_" + component + "_J"] + n["static_energy_" + component + "_J"]) / n["energy_J"]
        en["static_percent"] = 100 * n["static_J"] / n["energy_J"]
        en["fission_HBM_percent"] = 100 * f["fission_hbm_J"] / f["energy_J"]
        close(sum(en[k + "_percent"] for k in ("sa", "vu", "sram", "hbm", "ici", "other")), 100)
        energy.append(en)
        ca = dict(workload=name,
            native_HBM_bound_time_percent=100 * n["hbm_bound_time_ns"] / n["time_ns"],
            native_fallback_calls_percent=100 * n["fallback"] / n["logical_gemms"],
            native_fallback_time_percent=100 * n["fallback_time_ns"] / n["time_ns"],
            native_remap_fallback_calls_percent=100 * get("native_full")["fallback"] / n["logical_gemms"],
            fission_to_native_HBM_bytes=f["hbm_bytes"] / n["hbm_bytes"],
            fission_square_to_full_HBM_bytes=get("fission_square")["hbm_bytes"] / f["hbm_bytes"],
            fission_independent_to_full_HBM_bytes=get("fission_independent32")["hbm_bytes"] / f["hbm_bytes"],
            fission_common_square_edp=get("fission_common_square")["energy_J"] * get("fission_common_square")["time_ns"] / (get("fission_common_full")["energy_J"] * get("fission_common_full")["time_ns"]),
            fission_common_independent_edp=get("fission_common_independent32")["energy_J"] * get("fission_common_independent32")["time_ns"] / (get("fission_common_full")["energy_J"] * get("fission_common_full")["time_ns"]),
            native_1GHz_to_stock_time=get("native_1GHz_full")["time_ns"] / get("native_full")["time_ns"],
            native_1GHz_to_stock_energy=get("native_1GHz_full")["energy_J"] / get("native_full")["energy_J"],
            fission_serial_to_overlap_time=get("fission_serial_full")["time_ns"] / f["time_ns"],
            native_forcedSA_skew_speedup=get("native_forceSA_skew")["time_ns"] / get("native_forceSA_full")["time_ns"])
        causes.append(ca)
        absolute.append(dict(workload=name, records=meta[group]["records"], logical_gemms=n["logical_gemms"],
                             unique_shapes=n["unique_shapes"], fission_seconds=f["time_ns"] * 1e-9,
                             native_frozen_seconds=n["time_ns"] * 1e-9,
                             native_remap_seconds=get("native_full")["time_ns"] * 1e-9,
                             native_remap_1GHz_seconds=get("native_1GHz_full")["time_ns"] * 1e-9,
                             fission_J=f["energy_J"], native_J=n["energy_J"],
                             fission_HBM_GB=f["hbm_bytes"] * 1e-9, native_HBM_GB=n["hbm_bytes"] * 1e-9))
        mapping_rows.append(dict(workload=name, changed_calls_percent=100 * n["native_full_mapping_changed"] / n["logical_gemms"],
                                 frozen_to_remap_time=n["time_ns"] / get("native_full")["time_ns"],
                                 frozen_to_remap_energy=n["energy_J"] / get("native_full")["energy_J"]))
        old = dict(workload=name)
        for grain in (8, 32):
            b = get(f"legacy{grain}_full")
            old[f"legacy{grain}_array_cycles_to_fission"] = b["sa_cycles"] / f["sa_cycles"]
            old[f"legacy{grain}_time_to_fission"] = b["time_ns"] / f["time_ns"]
            old[f"legacy{grain}_energy_to_fission"] = b["energy_J"] / f["energy_J"]
            for variant in ("square", "independent", "skew"):
                a = get(f"legacy{grain}_{variant}")
                old[f"legacy{grain}_{variant}_speedup"] = a["time_ns"] / b["time_ns"]
                old[f"legacy{grain}_{variant}_edp_ratio"] = a["time_ns"] * a["energy_J"] / (b["time_ns"] * b["energy_J"])
            a, b = get(f"native_local{grain}_full"), get("native_full")
            old[f"native_local{grain}_time_to_replay"] = a["time_ns"] / b["time_ns"]
            old[f"native_local{grain}_time_to_frozen"] = a["time_ns"] / get("native_frozen_full")["time_ns"]
        legacy.append(old)
    outputs = dict(ablations=ablations, energy=energy, causes=causes, absolute=absolute, mapping=mapping_rows, legacy=legacy)
    for label, records in outputs.items():
        write_csv(out / (label + ".csv"), records)
        for expected, actual in zip(records, rows(out / (label + ".csv")), strict=True):
            for key in expected:
                assert str(expected[key]) == actual[key]
    selected_examples = []
    for d, c, name in GROUPS:
        for row in sorted(examples[d, c], key=lambda r: r["extra_fission_independent_energy_J"], reverse=True)[:10]:
            selected_examples.append(dict(workload=name, **row))
    write_csv(out / "examples.csv", selected_examples)
    report = ["# Same-workload simulator comparison", "",
        "All tables are generated from retained per-shape outputs using the original workload repeat weights. "
        "Time and energy are summed separately; workload EDP is their product. A ratio above one favors full Tessera. "
        "These are collected GEMM service-time sums, not complete application latency.", "",
        "The Fission reference replays the frozen final-three-ablations candidates and Planaria flow templates. "
        "Native NeuSim replaces only SA timing with the identical candidate cycles, then executes its own VU fallback, "
        "tiling, HBM/SRAM, overlap, utilization, power and regulator functions. Mapping is reselected by native EDP. "
        "The primary paired table fixes revision mappings. Native remapping is retained as a diagnostic: "
        "every remapped full-Tessera call falls back to VU because slower SA candidates can cross the native "
        "4x VU threshold and receive a lower cost. This is not evidence of improved Tessera array execution. "
        "Forced-SA, fixed 1 GHz, shared HBM templates and the previous custom model are separate controls.", "",
        "## Workload coverage and absolute totals", table(absolute, list(absolute[0])), "",
        "## Three ablations (speedup / EDP ratio)"]
    for variant in ("square", "independent32", "skew"):
        report += ["", "### " + variant, table(ablations, ["workload"] + [prefix + "_" + variant + suffix
                     for prefix in ("fission", "native_frozen", "native_forceSA") for suffix in ("_speedup", "_edp_ratio")])]
    report += ["", "## Native energy at frozen mappings (component static + dynamic, percent)", table(energy, list(energy[0])),
               "", "## Controlled causes", table(causes, list(causes[0])),
               "", "## Mapping control", table(mapping_rows, list(mapping_rows[0])),
               "", "## Previous custom model, now on the same isolated inputs", table(legacy, list(legacy[0])), "",
               "The legacy independent variant is skew-free; the revision independent32 baseline uses traditional WS "
               "bank/drain timing. The legacy skew variant changes bank release and drain; the final revision skew "
               "ablation changes only final drain with fixed mapping. These definitions must not be conflated.", "",
               "## Model boundaries", "",
               "The native host has one 128x128 SA, 6 native VUs, 12 MiB SRAM, and 2.8e12 bytes/s HBM "
               "(converted to the binary GB/s convention used by native code). Native 500 ns HBM latency, "
               "NoPG, BF16 input/output and all existing power coefficients are retained. The imported SA candidates "
               "use 12,000,000 B for their internal compute chunks. Native memory remains mapping-independent here. "
               "This isolates the surrounding models; it is not a partition-aware native memory or VU implementation.", "",
               "Native NONE DVFS hardcodes component frequency to 1.7 GHz despite ChipConfig.freq_GHz=1 and "
               "enable_dvfs=False. The primary results preserve this behavior. The 1GHz control uses the existing "
               "per-component configuration API, with the same native functions and selected mappings.", "",
               "Native HBM power models controller/PHY/data transfer; it does not establish detailed external "
               "DRAM-array/refresh energy. Its default chip coefficients are borrowed TPU parameters, not calibrated "
               "Tessera silicon. Revision HBM is a 32 pJ/B assumption and counts Planaria BF16 A/B plus 64-bit C "
               "initial reads, partial writes and padding. Native GEMM uses BF16 output and its own tiler. "
               "Revision has no equivalent full-chip static/regulator terms.", ""]
    (out / "REPORT.md").write_text("\n".join(report))
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    verification = dict(status="PASS", workload_records=sum(m["records"] for m in meta.values()),
        unique_shapes=shape_count, variants=len(variants), shape_result_rows=raw_count,
        source_sha256={str(shape_path): sha(shape_path), str(workload_path): sha(workload_path)},
        checks="All shape identities/weights; no missing variants; Fission independent aggregate cross-check; all generated table CSVs read back field-for-field.",
        output_sha256={path.name: sha(path) for path in sorted(out.iterdir()) if path.is_file()})
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    # Codex: decision end


if __name__ == "__main__":
    main()
