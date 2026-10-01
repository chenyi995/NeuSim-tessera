"""Copy the paper workloads into NeuSim and independently check input accounting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error("a fresh directory inside NeuSim is required")
    out.mkdir(parents=True)
    workspace = ROOT.parents[1]
    fission = workspace / "FissionSA"
    paper = workspace / "Tessera-HPCA2027-revision"
    source = fission / "out/tessera-revision/paper_revision/inputs"
    records = []

    def copy(src, relative, expected=None):
        src = Path(src)
        before = sha(src)
        if expected is not None:
            assert before == expected, src
        dst = out / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        assert not dst.exists(), dst
        shutil.copyfile(src, dst)
        assert sha(src) == sha(dst) == before, src
        records.append(dict(source=str(src.resolve()), copied=str(dst), sha256=before,
                            bytes=dst.stat().st_size))
        return dst

    # chenyi9: decision start — copy the existing paper workload, preserving
    # every original row and using repeat exactly once; do not reconstruct shapes.
    manifest = json.loads((source / "input_manifest.json").read_text())
    expected = {Path(r["path"]).relative_to(source.relative_to(fission)): r["sha256"]
                for r in manifest["outputs"]}
    for src in sorted(source.rglob("*")):
        if src.is_file():
            rel = src.relative_to(source)
            copy(src, Path("inputs") / rel, expected.get(rel))
    original_meta = copy(ROOT / "configs/chips/tessera_revision.json", "original_model_metadata.json")
    old = json.loads(original_meta.read_text())
    copied_sources = {}
    for item in manifest["sources"]:
        if "mnk_trace.csv" in item["path"] or "batch_composition.csv" in item["path"]:
            copied_sources[str((fission / item["path"]).resolve())] = copy(
                fission / item["path"], Path("trace_sources") / item["path"], item["sha256"])
    for path, record in old["sources"].items():
        src = Path(path)
        if src.name == "config.json" and src.is_relative_to(workspace / "tessera_repro"):
            copy(src, Path("model_sources") / src.relative_to(workspace), record["sha256"])
    refs = [copy(paper / "sec/04_design.tex", "references/paper_design.tex"),
            copy(paper / "sec/08_methodology.tex", "references/paper_methodology.tex"),
            copy(paper / "README_revision_workloads.md", "references/paper_workloads.md"),
            copy(workspace / "gemmini/partition/TIMING_AUDIT.md", "references/rtl_timing_audit.md")]
    groups_file = copy(fission / "Tessera-revision/experiments/paper_revision/workload_groups.json",
                       "workload_groups.json")
    for name in ("ablation_edp.csv", "service_speedup.csv", "service_totals.csv", "copy_manifest.json"):
        copy(paper / "fig/plotting/data/revision" / name, Path("paper_reference") / name)
    for grain in (32, 8):
        copy(fission / f"out/tessera-revision/paper_revision/ablation_g{grain}/comparisons.csv",
             f"paper_reference/comparisons_g{grain}.csv")
    groups = json.loads(groups_file.read_text())
    metadata = dict(models=old["models"],
                    traces={tag: str(copied_sources[str(Path(path).resolve())])
                            for tag, path in old["traces"].items()},
                    cohorts=manifest["cohorts"], workload_groups=groups,
                    tessera_references=[str(path) for path in refs],
                    source_model_metadata=str(original_meta),
                    accounting="Native NeuSim model parameters; original metadata supplies model geometry only.")
    meta_path = out / "workload_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    assert json.loads(meta_path.read_text()) == metadata
    # chenyi9: decision end

    # Codex: decision start — re-read the copied data and compare independently
    # with the producer summaries, trace MACs and paper cohort identities.
    shapes = {int(r["shape_id"]): tuple(int(r[k]) for k in ("M", "N", "K"))
              for r in rows(out / "inputs/unique_shapes.csv")}
    totals = defaultdict(Counter)
    seen = defaultdict(set)
    for r in rows(out / "inputs/workloads.csv"):
        key = r["dataset"], r["cohort"]
        sid = int(r["shape_id"])
        dims = tuple(int(r[k]) for k in ("M", "N", "K"))
        assert shapes[sid] == dims
        repeat = int(r["repeat"])
        assert repeat > 0
        totals[key]["records"] += 1
        totals[key]["logical_gemms"] += repeat
        totals[key]["logical_macs"] += repeat * dims[0] * dims[1] * dims[2]
        seen[key].add(sid)
    assert set(totals) == {(g["dataset"], g["cohort"]) for g in groups}
    summary = list(rows(out / "inputs/dataset_summary.csv"))
    for r in summary:
        key = r["dataset"], r["cohort"]
        for field in ("records", "logical_gemms", "logical_macs"):
            assert totals[key][field] == int(r[field]), (key, field)
        assert len(seen[key]) == int(r["unique_shapes"])
        if r["trace_tag"]:
            actual = Counter()
            for t in rows(metadata["traces"][r["trace_tag"]]):
                weight = int(t["gemm_batch"]) * int(t["num_layers"])
                actual["records"] += 1
                actual["logical_gemms"] += weight
                actual["logical_macs"] += weight * int(t["M"]) * int(t["N"]) * int(t["K"])
            assert actual == totals[key], r["trace_tag"]
    assert sum(r["records"] for r in totals.values()) == manifest["input_records"]
    assert len(shapes) == manifest["unique_shapes"]
    for r in records:
        assert sha(Path(r["source"])) == sha(Path(r["copied"])) == r["sha256"]
    copy(__file__, "prepare_tessera_paper.py")
    result = dict(status="PASS", created_utc=datetime.now(timezone.utc).isoformat(),
                  input_records=manifest["input_records"], unique_shapes=len(shapes),
                  cohorts=len(totals), full_model_traces=len(metadata["traces"]),
                  repeat_semantics=manifest["repeat_semantics"],
                  source_and_copy_hash_checks="PASS", weighted_cohort_readback="PASS",
                  trace_rows_and_macs="PASS", metadata_sha256=sha(meta_path), files=records)
    target = out / "copy_verification.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(target.read_text()) == result
    # Codex: decision end
    print(json.dumps({k: v for k, v in result.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
