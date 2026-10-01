#!/usr/bin/env python3
"""Build the twelve paper workload cohorts from saved inputs; no model calls."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
TRACES = ROOT / "out/tessera-hpca/Batch6-BW-Sweep/workload/traces"
COMPOSITIONS = ROOT / "out/tes-old/continuous_batching/traces"
SELECTED = ROOT / "out/tessera-revision/fissionsa_selected8_hbm4/sram_12mb/inputs/workloads.csv"
NATIVE = ROOT / "out/tessera-revision/planaria_all_collected_800_hbm4"
TRACE_SPECS = [
    ("llama2-7b_conv300", "llama2_vidur", "meta-llama/Llama-2-7b-hf", "Llama-2-7B / conv"),
    ("llama3-8b_conv300", "llama3_vidur", "meta-llama/Meta-Llama-3-8B", "Llama-3-8B / conv"),
    ("llama2-70b-tp4_conv300", "llama2_70b_tp4_vidur", "meta-llama/Llama-2-70b-hf", "Llama-2-70B TP4 / conv"),
    ("phi2_conv300", "phi2_vidur", "microsoft/phi-2", "Phi-2 / conv"),
    ("llama2-7b_code300", "llama2_code_vidur", "meta-llama/Llama-2-7b-hf", "Llama-2-7B / code"),
    ("llama2-7b_arxiv-qps1", "llama2_arxiv_vidur", "meta-llama/Llama-2-7b-hf", "Llama-2-7B / arXiv-1QPS"),
]
NEW_LABELS = {
    ("flashinfer_attention", "gqa_decode"): "GQA decode",
    ("flashinfer_attention", "single_kv_head_decode"): "Single KV-head decode",
    ("upstream_reuse_prefill", "CacheBlend_samsum"): "CacheBlend SAMSum",
    ("upstream_reuse_prefill", "CacheBlend_wikimqa"): "CacheBlend WikiMQA",
    ("upstream_reuse_prefill", "EPIC_hotpotqa"): "EPIC HotpotQA",
    ("upstream_reuse_prefill", "EPIC_multi_news"): "EPIC Multi-News",
}
OP_NAMES = {"attn_qk": "QK", "attn_pv": "PV"}


def rows(path):
    with path.open(newline="") as stream:
        yield from csv.DictReader(stream)


def sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def rel(path):
    return path.resolve().relative_to(ROOT).as_posix()


def fingerprint(path, purpose):
    return dict(path=rel(path), bytes=path.stat().st_size, sha256=sha(path), purpose=purpose)


def mnk(row):
    value = tuple(int(row[k]) for k in ("M", "N", "K"))
    assert min(value) > 0
    return value


def write_csv(path, fields, records):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def trace_records(spec):
    """Preserve every raw row and validate phase against batch composition."""
    tag, dataset, model, label = spec
    source = TRACES / tag / "mnk_trace.csv"
    composition = COMPOSITIONS / tag / "batch_composition.csv"
    requests = {}
    batches = defaultdict(list)
    for row in rows(composition):
        key = row["batch_id"], row["request_id"]
        assert key not in requests, (tag, key)
        requests[key] = row
        batches[row["batch_id"]].append(row)
    batch_info = {}
    for bid, members in batches.items():
        phases = {r["phase"] for r in members}
        batch_info[bid] = (sum(int(r["tokens_this_iter"]) for r in members),
                           next(iter(phases)) if len(phases) == 1 else "mixed", len(members))
    for line, raw in enumerate(rows(source), 2):
        M, N, K = mnk(raw)
        count = int(raw["gemm_batch"]) * int(raw["num_layers"])
        assert count > 0
        if raw["scope"] == "request":
            request = requests[raw["batch_id"], raw["request_id"]]
            phase = request["phase"]
            query = int(request["tokens_this_iter"])
            context = query + int(request["kv_cache_before"])
            assert M == query
            assert raw["op"] in ("attn_qk", "attn_pv")
            assert (N if raw["op"] == "attn_qk" else K) == context
            mapping = "saved_Vidur_per_head_attention"
        else:
            tokens, phase, _ = batch_info[raw["batch_id"]]
            assert M == tokens
            query = context = ""
            mapping = "batch_tokens_share_model_weights"
        result = dict(raw)
        result.update(dataset=dataset, cohort="full_trace", model=model,
            workload_label=label, trace_tag=tag, phase=phase,
            op=OP_NAMES.get(raw["op"], raw["op"]), source_op=raw["op"],
            repeat=count, source_records=1, source=rel(source), source_row=line,
            source_scope=raw["scope"], mapping=mapping,
            evidence="existing_Vidur_simulation", evaluation_batch="primary",
            collection_role="complete_saved_trace", query_tokens=query,
            context_tokens=context, batch_size=batch_info[raw["batch_id"]][2],
            input_table_id="paper_vidur_" + tag, input_csv_row=line,
            paper_input_source_file=rel(source), paper_input_source_row=line,
            paper_input_group="original_six", batch_composition_source=rel(composition),
            notes="All saved GEMM records; repeat=gemm_batch*num_layers once. Phase checked against original batch composition. Service cycles are additive, not arrival-aware makespan or GPU runtime.")
        yield result


def new_records():
    for line, raw in enumerate(rows(SELECTED), 2):
        group = raw["dataset"], raw["cohort"]
        if group not in NEW_LABELS:
            continue
        result = dict(raw)
        result.update(workload_label=NEW_LABELS[group],
            paper_input_source_file=rel(SELECTED), paper_input_source_row=line,
            paper_input_group="new_six", prior_workload_id=raw["workload_id"],
            prior_shape_id=raw["shape_id"])
        yield result


def all_records():
    for spec in TRACE_SPECS:
        yield from trace_records(spec)
    yield from new_records()


def counter_key(row):
    return (row["phase"], row.get("source_op") or row["op"], *mnk(row))


def build(out):
    if out.exists():
        raise FileExistsError(f"Preserve existing input directory: {out}")
    sources = [fingerprint(SELECTED, "Frozen eight-cohort input; retain six new cohorts and verify duplicate Vidur cohorts")]
    for tag, *_ in TRACE_SPECS:
        sources.extend([fingerprint(TRACES / tag / "mnk_trace.csv", "Original paper complete GEMM trace"),
                        fingerprint(COMPOSITIONS / tag / "batch_composition.csv", "Original schedule composition used to validate request geometry and phase")])
    native_manifest_path = NATIVE / "run/run_manifest.json"
    native_manifest = json.loads(native_manifest_path.read_text())
    assert native_manifest["status"] == "complete"
    assert sha(NATIVE / "inputs.csv") == native_manifest["input"]["sha256"]
    native_shapes = {mnk(row) for row in rows(NATIVE / "inputs.csv")}
    sources.extend([fingerprint(NATIVE / "inputs.csv", "Native cache MNK coverage only"),
                    fingerprint(native_manifest_path, "Completed native run metadata; no assertion of identical main-figure memory model")])
    groups = {}
    fields = set()
    duplicate_original = defaultdict(Counter)
    operators = defaultdict(lambda: dict(rows=0, logical_gemms=0, logical_macs=0, shapes=set()))
    for row in all_records():
        fields.update(row)
        key = row["dataset"], row["cohort"]
        shape = mnk(row)
        repeat = int(row["repeat"])
        assert repeat > 0
        g = groups.setdefault(key, dict(dataset=key[0], cohort=key[1], model=row["model"],
            workload_label=row["workload_label"], trace_tag=row.get("trace_tag", ""),
            input_group=row["paper_input_group"], records=0, source_records=0,
            logical_gemms=0, logical_macs=0, shapes=set()))
        g["records"] += 1
        g["source_records"] += int(row["source_records"])
        g["logical_gemms"] += repeat
        g["logical_macs"] += repeat * shape[0] * shape[1] * shape[2]
        g["shapes"].add(shape)
        o = operators[key + (row["phase"], row["op"])]
        o["rows"] += 1
        o["logical_gemms"] += repeat
        o["logical_macs"] += repeat * shape[0] * shape[1] * shape[2]
        o["shapes"].add(shape)
        if row["dataset"] in ("llama2_vidur", "phi2_vidur"):
            duplicate_original[row["dataset"]][counter_key(row)] += repeat
    duplicate_selected = defaultdict(Counter)
    for row in rows(SELECTED):
        if row["dataset"] in duplicate_original:
            duplicate_selected[row["dataset"]][counter_key(row)] += int(row["repeat"])
    assert duplicate_original == duplicate_selected, "Vidur duplicate phase/op/MNK/repeat changed"
    assert len(groups) == 12
    shapes = sorted(set().union(*(g["shapes"] for g in groups.values())))
    shape_ids = {shape: index for index, shape in enumerate(shapes)}
    first = ["workload_id", "shape_id", "dataset", "cohort", "model", "workload_label", "phase", "op", "M", "N", "K", "repeat"]
    fields = first + sorted(fields - set(first))
    out.mkdir(parents=True)
    count = 0
    def output_records():
        nonlocal count
        for count, row in enumerate(all_records(), 1):
            row.update(workload_id=count - 1, shape_id=shape_ids[mnk(row)])
            yield row
    write_csv(out / "workloads.csv", fields, output_records())
    assert count == sum(g["records"] for g in groups.values())
    write_csv(out / "unique_shapes.csv", ["shape_id", "M", "N", "K"],
              (dict(shape_id=i, M=s[0], N=s[1], K=s[2]) for i, s in enumerate(shapes)))
    group_rows = []
    coverage = []
    for key, g in groups.items():
        group_rows.append({k: v for k, v in g.items() if k != "shapes"} | {"unique_shapes": len(g["shapes"])})
        coverage.append(dict(dataset=key[0], cohort=key[1], trace_tag=g["trace_tag"],
            unique_shapes=len(g["shapes"]), native_cached_shapes=len(g["shapes"] & native_shapes),
            native_missing_shapes=len(g["shapes"] - native_shapes)))
    write_csv(out / "dataset_summary.csv", list(group_rows[0]), group_rows)
    write_csv(out / "native_cache_coverage.csv", list(coverage[0]), coverage)
    operator_rows = [dict(dataset=k[0], cohort=k[1], phase=k[2], op=k[3],
        **{name: value for name, value in v.items() if name != "shapes"}, unique_shapes=len(v["shapes"]))
        for k, v in sorted(operators.items())]
    write_csv(out / "operator_summary.csv", list(operator_rows[0]), operator_rows)
    missing = [dict(shape_id=shape_ids[s], M=s[0], N=s[1], K=s[2],
        cohorts=";".join(g["dataset"] + "/" + g["cohort"] for g in groups.values() if s in g["shapes"]))
        for s in shapes if s not in native_shapes]
    write_csv(out / "native_cache_missing_shapes.csv", ["shape_id", "M", "N", "K", "cohorts"], missing)
    (out / "producer_snapshot").mkdir()
    shutil.copy2(Path(__file__), out / "producer_snapshot/build_workloads.py")
    # Independently read the emitted rows to prove transport and exact per-source coverage.
    transported = defaultdict(Counter)
    source_lines = Counter()
    for row in rows(out / "workloads.csv"):
        transported[row["dataset"], row["cohort"]][counter_key(row)] += int(row["repeat"])
        source_lines[row["paper_input_source_file"]] += 1
        assert mnk(row) == shapes[int(row["shape_id"])]
    expected = defaultdict(Counter)
    for row in all_records():
        expected[row["dataset"], row["cohort"]][counter_key(row)] += int(row["repeat"])
    assert expected == transported
    assert all(sha(ROOT / item["path"]) == item["sha256"] for item in sources)
    manifest = dict(status="complete", operation="Saved-input extraction and verification only; no simulation",
        producer=fingerprint(Path(__file__), "Reproducible input builder"), sources=sources,
        cohorts=group_rows, input_records=count, unique_shapes=len(shapes),
        native_cache=dict(unique_shapes=len(native_shapes), covered_shapes=len(set(shapes) & native_shapes),
            missing_shapes=len(missing), meaning="MNK set coverage only; numerical reuse requires separately matching model/config hashes"),
        duplicate_cohort_checks={name: dict(status="PASS", key="phase/source_op/M/N/K",
            exact_repeat_equal=True, logical_gemms=sum(counter.values()), distinct_keys=len(counter))
            for name, counter in duplicate_original.items()},
        preservation="Every original-six MNK CSV row is preserved once. New attention/reuse rows retain all original metadata and repeat. Llama2/Phi2 selected-eight copies are excluded only after exact weighted phase/op/MNK equality.",
        repeat_semantics="Use repeat once; Vidur repeat=gemm_batch*num_layers. source_records is provenance, not another multiplicity.",
        phase_semantics="Vidur request phase and attention query/context lengths checked against saved batch composition; dense batches retain mixed phases.",
        operation_names="Only op attn_qk/attn_pv normalized to QK/PV; source_op preserves the original spelling and all MNK values stay unchanged.",
        metric="Per-cohort weighted single-GEMM service-cycle sum; scheduled_at is preserved but no arrival idle or cross-GEMM overlap is applied by the input builder.",
        reuse_scope="Original four datasets attempted all 800 rows; 799 source-valid requests retained. Existing Multi-News index37 exclusion and duplicate-index semantics are unchanged.",
        checks=dict(output_phase_operator_mnk_weights="PASS", duplicate_vidur_inputs="PASS", source_hash_stability="PASS", source_line_counts=dict(source_lines)),
        outputs=[fingerprint(p, "Generated paper input") for p in sorted(out.rglob("*")) if p.is_file()])
    (out / "input_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "input_records", "unique_shapes", "native_cache", "duplicate_cohort_checks")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "out/tessera-revision/paper_revision/inputs")
    args = parser.parse_args()
    build(args.out.resolve())


if __name__ == "__main__":
    main()
