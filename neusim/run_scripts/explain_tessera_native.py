"""Produce compact, read-back-verified interpretation from the complete comparison."""
import argparse
import itertools
import json
from pathlib import Path
import shutil

from neusim.run_scripts.compare_tessera_native import ROOT, rows, sha, write_csv
from neusim.run_scripts.summarize_tessera_native import table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error("use a new directory inside NeuSim")
    verify = json.loads((a.analysis / "verification.json").read_text())
    assert verify["status"] == "PASS"
    data = {}
    for name in ("ablations", "causes", "energy", "absolute", "legacy", "mapping"):
        path = a.analysis / (name + ".csv")
        assert sha(path) == verify["output_sha256"][path.name]
        data[name] = [{k: v if k == "workload" else float(v) for k, v in r.items()} for r in rows(path)]
    by = {name: {r["workload"]: r for r in records} for name, records in data.items()}
    compact = []
    for r in data["ablations"]:
        item = dict(workload=r["workload"])
        for variant in ("square", "independent32", "skew"):
            item[variant + " F/N EDP"] = f'{r["fission_"+variant+"_edp_ratio"]:.4f} / {r["native_frozen_"+variant+"_edp_ratio"]:.4f}'
        compact.append(item)
    energy = [dict(workload=r["workload"], SA=r["sa_percent"], VU=r["vu_percent"], SRAM=r["sram_percent"],
                   HBM=r["hbm_percent"], Other_with_ICI=r["other_percent"]+r["ici_percent"], static_share=r["static_percent"])
              for r in data["energy"]]
    common = [dict(workload=r["workload"], square_matched=by["ablations"][r["workload"]]["fission_square_edp_ratio"],
                   square_common=r["fission_common_square_edp"], independent_matched=by["ablations"][r["workload"]]["fission_independent32_edp_ratio"],
                   independent_common=r["fission_common_independent_edp"], fission_to_native_bytes=r["fission_to_native_HBM_bytes"])
              for r in data["causes"]]
    # Codex: decision start — derive every quoted percentage from retained CSVs.
    facts = dict(workload_records=verify["workload_records"], unique_shapes=verify["unique_shapes"],
                 shape_result_rows=verify["shape_result_rows"],
                 native_static_percent_range=[min(r["static_percent"] for r in data["energy"]), max(r["static_percent"] for r in data["energy"])],
                 native_sa_percent_range=[min(r["sa_percent"] for r in data["energy"]), max(r["sa_percent"] for r in data["energy"])],
                 native_frequency_time_ratio_range=[min(r["native_1GHz_to_stock_time"] for r in data["causes"]), max(r["native_1GHz_to_stock_time"] for r in data["causes"])])
    for workload in ("Llama2", "Phi-2", "CacheBlend SAMSum", "EPIC Multi-News"):
        r = by["legacy"][workload]
        facts[workload] = dict(
            old8_independent_slowdown_percent=100 * (r["legacy8_independent_speedup"] - 1),
            old32_independent_slowdown_percent=100 * (r["legacy32_independent_speedup"] - 1),
            old32_skew_slowdown_percent=100 * (r["legacy32_skew_speedup"] - 1),
            final_skew_slowdown_percent=100 * (by["ablations"][workload]["fission_skew_speedup"] - 1))
    examples = []
    # Find native-remapping discontinuities on retained real workload shapes.
    for sid, rs in itertools.groupby(rows(a.run / "shape_results.csv"), lambda r: r["shape_id"]):
        records = {r["variant"]: r for r in rs}
        fixed, free = records["native_frozen_full"], records["native_full"]
        if (int(fixed["fallback"]) == 0 and int(free["fallback"]) == 1
                and int(free["sa_cycles"]) > int(fixed["sa_cycles"])
                and int(free["time_ns"]) < int(fixed["time_ns"])):
            examples.append({"shape_id": int(sid), **{key: int(fixed[key]) for key in ("M", "N", "K")},
                "fixed_sa_cycles": int(fixed["sa_cycles"]), "remapped_sa_cycles": int(free["sa_cycles"]),
                "fixed_native_time_ns": int(fixed["time_ns"]), "remapped_native_time_ns": int(free["time_ns"]),
                "remapped_native_vu_before_power_ns": int(free["pre_vu_ns"])})
            if len(examples) == 3:
                break
    assert examples
    facts["fallback_examples"] = examples
    # Codex: decision end
    out.mkdir(parents=True)
    for label, records in (("ablation_summary", compact), ("energy_summary", energy), ("traffic_control", common)):
        write_csv(out / (label + ".csv"), records)
        for x, y in zip(records, rows(out / (label + ".csv")), strict=True):
            assert all(str(value) == y[key] for key, value in x.items())
    (out / "key_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    assert json.loads((out / "key_facts.json").read_text()) == facts
    text = ["# Why the simulator results differ", "",
        f'Compared all {facts["workload_records"]:,} saved records / {facts["unique_shapes"]:,} unique shapes, with original repeat weights.', "",
        "The reference was replayed and verified. NeuSim's primary paired run fixes the reference SA mappings "
        "and runs native VU, memory and energy accounting. The injected data are SA cycles only. "
        "These totals are collected GEMM service times, not application E2E measurements.", "",
        "## Main ablations", "F/N means Fission / native NeuSim at fixed revision mappings. "
        "Entries are baseline EDP divided by full Tessera EDP; values above one favor full Tessera.", "",
        table(compact, list(compact[0])), "",
        "Rectangular mapping: the reference's large EDP differences mostly depend on the external, "
        "architecture-specific HBM templates. The original compute-only differences are small on these "
        "weighted workloads. Native NeuSim uses the same host memory tiling across variants, so it does not "
        "represent partition-specific input reuse or partial-sum spills.", "",
        "Independent arrays: the same HBM explanation applies. Llama2/Phi-2 retain small timing gains. "
        "Some fixed-mapping native ratios are slightly below one because an EDP-optimal mapping under "
        "Fission's energy accounting need not be optimal under native NeuSim's accounting. The separately "
        "reoptimized forced-SA control restores small nonnegative gains, not the large traffic-template benefit.", "",
        "Skew: both decode groups use VU for every primary native call, and HBM bounds their execution. "
        "Forcing SA still leaves most drain increments hidden by the native HBM latency floor. The reference "
        "has a transfer-based startup/finish model, without this native per-operator latency floor. "
        "The reference fixes energy for skew; native energy is recomputed, so native EDP and speedup differ.", "",
        "## Direct traffic-template control", table(common, list(common[0])), "",
        "The common-flow columns are reruns of the reference with the SAME full-array HBM template for all "
        "architectures. Their much smaller EDP gaps locate the major cause inside the reference itself; "
        "the result does not depend on interpreting native power coefficients. Planaria's templates also "
        "count 64-bit C, initial C reads, intermediate writes, padding and thread replication; native "
        "NeuSim's GEMM tiler uses BF16 output and its own residency policy.", "",
        "## Native energy breakdown", table(energy, list(energy[0]), digits=2), "",
        "Each component includes its static and dynamic energy. The final static_share column overlaps "
        "the component columns; do not sum it with them. NoPG leaves substantial whole-chip static power, "
        "including unused ICI and other circuitry. These are borrowed native chip parameters. They do not "
        "validate Tessera silicon energy or detailed external DRAM energy. The earlier custom branch "
        "instead used useful operations times a constant, a small array-background term, and 32 pJ/B HBM.", "",
        "## Native fallback and frequency diagnostics", table(examples, list(examples[0]), digits=0), "",
        "These real shapes show a slower SA candidate producing a faster native result after crossing the "
        "SA > 4 x VU fallback threshold. With native EDP remapping, all full-Tessera calls use VU; those "
        "numbers cannot be read as array-architecture improvements. Native NONE DVFS also resets component "
        "frequency to 1.7 GHz, even with ChipConfig=1 GHz and DVFS disabled. The public-API 1 GHz control "
        "changes total time by the ratios in key_facts.json. Neither native behavior was patched.", "",
        "## Why the earlier custom runs also differ", "",
        "The previous collection was a different full-model operator graph, not this frozen GEMM set. "
        "Its grain was 8, while the final revision grain is 32. Its independent arrays were skew-free, "
        "while the reference uses WS32. It serializes explicit vector reductions, and its skew ablation "
        "changes bank release as well as final drain. The retained same-workload legacy8/legacy32 runs "
        "quantify these differences; key_facts.json includes the independent-array and skew controls. "
        "For example, the Multi-News legacy bank-release skew effect remains large after matching grain, "
        "so it is not explained by grain alone.", "",
        "A native Tessera implementation intended to evaluate link reuse must express partition-aware "
        "traffic and accumulation inside NeuSim's native models, with a documented engine-selection policy. "
        "This controlled replay explains the discrepancy and identifies the missing model terms.", ""]
    (out / "INTERPRETATION.md").write_text("\n".join(text))
    shutil.copyfile(Path(__file__), out / Path(__file__).name)
    (out / "verification.json").write_text(json.dumps(dict(status="PASS", source_analysis=str(a.analysis),
        source_verification_sha256=sha(a.analysis / "verification.json"),
        output_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
