"""Normalize an audited joint-EDP run to mono WS and redraw its figures.

Consumes existing totals and their independent audit; never runs the simulator.
Figure style reference: Tessera-HPCA-2026/fig/plotting/make_grainfloor_fig.py.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
MAIN = ("WS", "Planaria-32", "SOSA", "FlexSA", "SISA", "Tessera-32", "Tessera-8")
BOUNDARIES = (
    ("model_compute_e2e", "Model E2E"),
    ("single_operator_e2e_weighted_sum", "Invocation-weighted GEMM E2E"),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def save_json(path, data):
    with path.open("x") as stream:
        stream.write(json.dumps(data, indent=2) + "\n")
    assert json.loads(path.read_text()) == data


def save_csv(path, records):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    actual = rows(path)
    assert actual == [{k: str(v) for k, v in r.items()} for r in records], path


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=2e-10, abs_tol=1e-16), (actual, expected)


def key(row):
    return row["workload_id"], row["architecture"], int(row["bandwidth_bytes_per_second"])


def label(arch):
    if arch == "WS":
        return "Mono WS"
    return arch.replace("-independent_noskew", " w/o interconnect").replace("-square", " square only")


def figure_check(fig):
    """Check that all visible labels lie within the saved canvas."""
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inactive = set()
    for ax in fig.axes:
        for axis, bounds in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(bounds)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    inactive.update((tick.label1, tick.label2))
    checked = 0
    for text in fig.findobj(Text):
        if text in inactive or not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1 and box.x1 <= fig.bbox.x1 + 1 and box.y1 <= fig.bbox.y1 + 1, (text.get_text(), box)
        checked += 1
    return checked


def render(out, groups, high):
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # User decision start — retain bandwidth curves; use Fig. 10 bars for ablations.
    # Style source: make_grainfloor_fig.py, horizontal bars and min/max whiskers.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = dict(zip(MAIN, ("#999999", "#252525", "#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")))
    general = rows(out / "general_edp.csv")
    fig, axes = plt.subplots(3, 4, figsize=(15, 9))
    curve_checks = 0
    for ax, group in zip(axes.flat, groups):
        for arch in MAIN:
            series = sorted((r for r in general if r["workload_id"] == group["workload_id"] and r["architecture"] == arch),
                            key=lambda r: int(r["bandwidth_bytes_per_second"]))
            xs = [int(r["bandwidth_bytes_per_second"]) / 1e12 for r in series]
            ys = [float(r["edp_gain_vs_ws"]) for r in series]
            line, = ax.plot(xs, ys, color=colors[arch], label=label(arch), linewidth=1.8,
                            marker="o" if arch.startswith("Tessera") else None, markersize=3,
                            linestyle="--" if arch == "WS" else "-")
            assert list(line.get_xdata()) == xs and list(line.get_ydata()) == ys
            curve_checks += len(ys)
        boundary = dict(BOUNDARIES)[series[0]["boundary"]]
        ax.set_title(group["name"] + "\n" + boundary, fontsize=9)
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
        ax.set_xlabel("HBM bandwidth per chip (TB/s)")
        ax.set_ylabel("EDP gain over mono WS")
        ax.grid(alpha=.2, which="both")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(MAIN), loc="upper center", frameon=False)
    fig.tight_layout(rect=(.005, .005, .995, .95))
    text_checks = figure_check(fig)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"general_edp_bandwidth.{ext}", dpi=180)
    plt.close(fig)

    summaries = rows(out / "summary_edp.csv")
    bar_checks = 0
    for variant, name, title in (("square", "edp_nonsquare", "Non-square support"),
                                 ("independent_noskew", "edp_interconnect", "Array interconnection")):
        architectures = ("WS", "Tessera-32", f"Tessera-32-{variant}", "Tessera-8", f"Tessera-8-{variant}")
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 3.35))
        for ax, (boundary, boundary_title) in zip(axes, BOUNDARIES):
            selected = [next(r for r in summaries if r["boundary"] == boundary and r["architecture"] == arch
                             and int(r["bandwidth_bytes_per_second"]) == high) for arch in architectures]
            means = [float(r["geomean_edp_gain_vs_ws"]) for r in selected]
            lows = [float(r["minimum_edp_gain_vs_ws"]) for r in selected]
            highs = [float(r["maximum_edp_gain_vs_ws"]) for r in selected]
            y = list(range(len(architectures)))
            palette = ["#999999" if arch == "WS" else "#3b7cb5" if arch in MAIN else "#f0913b" for arch in architectures]
            bars = ax.barh(y, means, .62, color=palette, edgecolor="0.25", linewidth=.5, zorder=2)
            errors = [[mean - low for mean, low in zip(means, lows)],
                      [high_value - mean for mean, high_value in zip(means, highs)]]
            whiskers = ax.errorbar(means, y, xerr=errors, fmt="none", ecolor="0.25",
                                  elinewidth=.8, capsize=2.2, capthick=.8, zorder=3)
            for bar, mean in zip(bars, means):
                close(bar.get_width(), mean)
                bar_checks += 1
            for segment, low, high_value in zip(whiskers[2][0].get_segments(), lows, highs):
                close(segment[0][0], low)
                close(segment[1][0], high_value)
            xmax = max(highs)
            for yi, mean, high_value in zip(y, means, highs):
                ax.text(high_value + xmax * .025, yi, f"{mean:.2f}×", ha="left", va="center", fontsize=9)
            ax.set(yticks=y, yticklabels=[label(arch) for arch in architectures], xlim=(0, xmax * 1.20),
                   xlabel="Geomean EDP gain over mono WS (×)", title=boundary_title)
            ax.axvline(1, color="0.45", linestyle="--", linewidth=.8, zorder=1)
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle=":", linewidth=.5, color="0.8")
            ax.set_axisbelow(True)
        fig.suptitle(f"{title} at HBM4 ({high / 1e12:g} TB/s)", fontsize=11, y=.98)
        fig.text(.5, .02, "Bars: equal-workload geomean; whiskers: workload min–max.  Mono WS = 1; higher is better.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(.005, .07, .995, .92), w_pad=2)
        text_checks += figure_check(fig)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=180)
        plt.close(fig)
    # User decision end
    return dict(curve_points_checked=curve_checks, bars_checked=bar_checks, visible_text_bounds_checked=text_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-analysis", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--style-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run, previous, out = args.run.resolve(), args.previous_analysis.resolve(), args.out.resolve()
    assert out.is_relative_to(ROOT) and not out.exists()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:1])
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["status"] == "PASS" and not manifest["pilot"]
    audit = json.loads((previous / "verification.json").read_text())
    assert audit["status"] == "PASS" and not audit["pilot"]
    for name, expected in manifest["output_sha256"].items():
        assert sha(run / name) == expected, name
    for name, expected in json.loads((previous / "artifact_hashes.json").read_text()).items():
        assert sha(previous / name) == expected, name
    # The inherited audit must refer to this exact run and these exact totals.
    for name in ("manifest.json", "totals.csv"):
        assert audit["source_sha256"][str(run / name)] == sha(run / name), name
    group_original = str(Path(manifest["base"]) / "inputs_snapshot/workload_groups.json")
    assert sha(args.groups) == manifest["source_sha256"][group_original]
    groups = json.loads(args.groups.read_text())
    high = max(manifest["bandwidth_bytes_per_second"])
    raw = rows(run / "totals.csv")
    lookup = {key(r): r for r in raw}
    assert len(lookup) == len(raw)
    configs = json.loads((run / "configs.json").read_text())
    config_keys = {tuple(name.rsplit("@", 1)) for name in configs}
    for group in groups:
        actual = {(r["architecture"], r["bandwidth_bytes_per_second"]) for r in raw if r["workload_id"] == group["workload_id"]}
        assert actual == config_keys
    ws_configs = [c for name, c in configs.items() if name.startswith("WS@")]
    assert all(c["sa_dim"] == 128 and c["num_sa"] == 1 and c["tessera_parameters"]["baseline_architecture"] == "WS" for c in ws_configs)

    normalized = []
    # User decision start — every EDP gain uses the same workload/bandwidth mono WS.
    for r in raw:
        wid, arch, bw = key(r)
        ws = lookup[wid, "WS", bw]
        close(r["edp_J_s"], float(r["energy_J"]) * int(r["chips"]) * int(r["time_ns"]) * 1e-9)
        close(r["system_energy_J"], float(r["energy_J"]) * int(r["chips"]))
        close(r["seconds"], int(r["time_ns"]) * 1e-9)
        assert r["chips"] == ws["chips"] and r["boundary"] == ws["boundary"]
        normalized.append(dict(**r, baseline_architecture="WS", baseline_edp_J_s=ws["edp_J_s"],
                               edp_gain_vs_ws=float(ws["edp_J_s"]) / float(r["edp_J_s"])))
    # User decision end
    general = [r for r in normalized if r["architecture"] in MAIN]
    ablations = []
    for variant in ("square", "independent_noskew"):
        architectures = {"WS", "Tessera-32", "Tessera-8", f"Tessera-32-{variant}", f"Tessera-8-{variant}"}
        ablations.extend(dict(**r, ablation=variant) for r in normalized
                         if r["architecture"] in architectures and int(r["bandwidth_bytes_per_second"]) == high)

    buckets = defaultdict(list)
    for r in normalized:
        buckets[r["boundary"], r["architecture"], int(r["bandwidth_bytes_per_second"])].append(r)
    summaries = []
    for (boundary, arch, bw), data in sorted(buckets.items()):
        values = [r["edp_gain_vs_ws"] for r in data]
        mean = math.exp(math.fsum(map(math.log, values)) / len(values))
        smallest = min(data, key=lambda r: r["edp_gain_vs_ws"])
        largest = max(data, key=lambda r: r["edp_gain_vs_ws"])
        summaries.append(dict(boundary=boundary, architecture=arch, bandwidth_bytes_per_second=bw,
                              workload_count=len(values), geomean_edp_gain_vs_ws=mean,
                              minimum_edp_gain_vs_ws=min(values), maximum_edp_gain_vs_ws=max(values),
                              minimum_workload=smallest["workload_id"], maximum_workload=largest["workload_id"]))
    headlines = []
    for g in groups:
        r = dict(workload_id=g["workload_id"], workload=g["name"])
        for arch in (*MAIN, "Tessera-32-square", "Tessera-8-square", "Tessera-32-independent_noskew", "Tessera-8-independent_noskew"):
            r[arch + "_EDP_gain_vs_WS"] = next(x["edp_gain_vs_ws"] for x in normalized
                                               if key(x) == (g["workload_id"], arch, high))
        headlines.append(r)

    out.mkdir(parents=True)
    shutil.copy2(__file__, out / Path(__file__).name)
    assert sha(Path(__file__)) == sha(out / Path(__file__).name)
    for name, records in (("normalized_edp.csv", normalized), ("general_edp.csv", general), ("edp_ablations.csv", ablations),
                          ("summary_edp.csv", summaries), ("hbm4_headline.csv", headlines)):
        save_csv(out / name, records)
    shutil.copy2(previous / "energy_breakdown.csv", out / "energy_breakdown.csv")
    assert sha(out / "energy_breakdown.csv") == sha(previous / "energy_breakdown.csv")

    # Independent read-back: recompute from energy/time components rather than
    # reusing the normalized records or their saved EDP column.
    readback = {key(r): r for r in rows(run / "totals.csv")}
    checked_ratios = 0
    for name in ("normalized_edp.csv", "general_edp.csv", "edp_ablations.csv"):
        for r in rows(out / name):
            wid, arch, bw = key(r)
            a, b = readback[wid, "WS", bw], readback[wid, arch, bw]
            expected = (float(a["energy_J"]) * int(a["chips"]) * int(a["time_ns"])) / (float(b["energy_J"]) * int(b["chips"]) * int(b["time_ns"]))
            close(r["edp_gain_vs_ws"], expected)
            if arch == "WS":
                assert float(r["edp_gain_vs_ws"]) == 1.0
            checked_ratios += 1
    for r in rows(out / "summary_edp.csv"):
        values = []
        for raw_row in readback.values():
            if (raw_row["boundary"], raw_row["architecture"], raw_row["bandwidth_bytes_per_second"]) != (r["boundary"], r["architecture"], r["bandwidth_bytes_per_second"]):
                continue
            wid, _, bw = key(raw_row)
            values.append(float(readback[wid, "WS", bw]["edp_J_s"]) / float(raw_row["edp_J_s"]))
        close(r["geomean_edp_gain_vs_ws"], math.prod(values) ** (1 / len(values)))
        close(r["minimum_edp_gain_vs_ws"], min(values))
        close(r["maximum_edp_gain_vs_ws"], max(values))
        assert int(r["workload_count"]) == len(values)

    rendered = render(out, groups, high)
    report = ["# Joint EDP evaluation normalized to mono WS", "",
              f"Run: `{run}`. These are modeled NeuSim estimates. Every displayed gain is derived from this run's verified `totals.csv`.", "",
              "Every figure uses EDP gain = mono WS EDP / design EDP at the same workload and bandwidth. Mono WS is exactly 1; higher is better. The WS instance is the monolithic array recorded in configs.json. EDP is total system energy times total serial execution time, including the tensor-parallel chip count in energy.", "",
              "[General bandwidth comparison](general_edp_bandwidth.png) retains per-workload curves. [Non-square ablation](edp_nonsquare.png) and [interconnection ablation](edp_interconnect.png) use horizontal bars inspired by Tessera-HPCA-2026 Fig. 10. Each full design and each ablation is normalized to mono WS. Blue bars are full Tessera; orange bars are the ablated designs; gray is mono WS.", "",
              "A bar is the equal-workload geometric mean of the E2E gain ratios in its panel; its whisker is the minimum to maximum across those workloads, not statistical uncertainty. Model E2E and invocation-weighted GEMM E2E are shown in separate panels. Unlike the reference figure's layer aggregation, these bars aggregate the requested E2E workload results.", "",
              "This update reuses the existing independent raw-row/weight audit and changes normalization and presentation only. Joint per-GEMM SRAM/array selection, finite SRAM reuse, HBM energy and native timing/power are unchanged. Per-GEMM selection is not a global graph EDP search. Power remains rough common-coefficient modeling; physical link power is not separately calibrated.", "",
              "The original energy breakdown is preserved byte-for-byte in `energy_breakdown.csv`. Detailed normalized data for all saved configurations, including the skew diagnostic, are in `normalized_edp.csv`. `summary_edp.csv` contains geometric means and workload ranges.", "",
              "HBM4 gains (derived from the same totals):", "",
              "| Workload | Mono WS | Planaria-32 | SOSA | FlexSA | SISA | Tessera-32 | Tessera-8 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in headlines:
        report.append("| " + r["workload"] + " | " + " | ".join(f"{r[arch + '_EDP_gain_vs_WS']:.6f}" for arch in MAIN) + " |")
    report.extend(["", "HBM4 ablated-design gains over mono WS:", "",
                   "| Workload | T32 square only | T8 square only | T32 w/o interconnect | T8 w/o interconnect |",
                   "|---|---:|---:|---:|---:|"])
    controls = ("Tessera-32-square", "Tessera-8-square", "Tessera-32-independent_noskew", "Tessera-8-independent_noskew")
    for r in headlines:
        report.append("| " + r["workload"] + " | " + " | ".join(f"{r[arch + '_EDP_gain_vs_WS']:.6f}" for arch in controls) + " |")
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    sources = [run / "manifest.json", run / "totals.csv", run / "configs.json", previous / "verification.json",
               previous / "artifact_hashes.json", args.groups.resolve(), args.style_source.resolve(), Path(__file__).resolve()]
    result = dict(status="PASS", decision=f"{getpass.getuser()} ruled: normalize every EDP comparison to mono WS; retain bandwidth curves and use Fig. 10 horizontal bars for ablations.",
                  accounting="WS EDP / design EDP; same workload and bandwidth; equal-workload geometric means; min/max workload ranges",
                  raw_simulation_repeated=False, inherited_raw_audit=str(previous / "verification.json"),
                  raw_output_hashes_checked=len(manifest["output_sha256"]), readback_ratios_checked=checked_ratios,
                  summary_rows_checked=len(summaries), source_sha256={str(p): sha(p) for p in sources}, **rendered)
    save_json(out / "verification.json", result)
    save_json(out / "artifact_hashes.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
