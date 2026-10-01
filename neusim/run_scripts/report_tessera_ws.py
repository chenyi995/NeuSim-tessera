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
# User decision start — show only Tessera-8, named Tessera, in the current figures.
MAIN = ("WS", "Planaria-32", "SOSA", "FlexSA", "SISA", "Tessera-8")
# User decision end
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
    # User decision start — retain original configuration IDs but simplify display names.
    return arch.replace("Tessera-8", "Tessera", 1).replace("-independent_noskew", " w/o interconnect").replace("-square", " square only")
    # User decision end


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


def render(out, groups, high, reuse_main=None, ablation_variants=("square", "independent_noskew")):
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    # User decision start — retain bandwidth curves and compare individual workloads with vertical bars.
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    colors = dict(zip(MAIN, ("#999999", "#252525", "#0072B2", "#E69F00", "#009E73", "#D55E00")))
    curve_checks = 0
    text_checks = 0
    if reuse_main is not None:
        prior_hashes = json.loads((reuse_main / "artifact_hashes.json").read_text())
        assert sha(reuse_main / "general_edp.csv") == sha(out / "general_edp.csv")
        for ext in ("png", "pdf"):
            name = f"general_edp_bandwidth.{ext}"
            assert sha(reuse_main / name) == prior_hashes[name]
            shutil.copy2(reuse_main / name, out / name)
            assert sha(out / name) == prior_hashes[name]
    else:
        general = rows(out / "general_edp.csv")
        fig, axes = plt.subplots(3, 4, figsize=(15, 9))
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
            # chenyi9: decision start — present every sourced workload as E2E.
            ax.set_title(group["name"] + "\nE2E", fontsize=9)
            # chenyi9: decision end
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

    # chenyi9: decision start — the revised main-only report uses an ablation table.
    ablations = rows(out / "edp_ablations.csv") if ablation_variants else []
    # chenyi9: decision end
    bar_checks = 0
    for variant, name, title in (("square", "edp_nonsquare", "Non-square support"),
                                 ("independent_noskew", "edp_interconnect", "Array interconnection")):
        if variant not in ablation_variants:
            continue
        architectures = ("Tessera-8", f"Tessera-8-{variant}")
        # chenyi9: decision start — show all E2E workloads together in each ablation.
        fig, ax = plt.subplots(figsize=(14, 3.55))
        for ax in (ax,):
            panel_rows = [r for r in ablations if r["ablation"] == variant]
        # chenyi9: decision end
            panel_groups = [g for g in groups if any(r["workload_id"] == g["workload_id"] for r in panel_rows)]
            x = list(range(len(panel_groups)))
            ymax = max(float(r["edp_gain_vs_ws"]) for r in panel_rows)
            # User requested the 1x reference near the bottom. This is an
            # explicitly labeled truncated axis; all bars still store real gains.
            ymin = .9
            assert min(float(r["edp_gain_vs_ws"]) for r in panel_rows) > ymin
            for index, arch in enumerate(architectures):
                values = [float(next(r["edp_gain_vs_ws"] for r in panel_rows
                                     if r["workload_id"] == g["workload_id"] and r["architecture"] == arch))
                          for g in panel_groups]
                positions = [v + (index - .5) * .36 for v in x]
                bars = ax.bar(positions, values, width=.36, color=("#3b7cb5", "#f0913b")[index],
                              edgecolor="0.25", linewidth=.5, label=label(arch), zorder=2)
                for bar, value in zip(bars, values):
                    close(bar.get_height(), value)
                    bar_checks += 1
                ax.bar_label(bars, labels=[f"{v:.2f}" for v in values], fontsize=7, padding=3)
            ax.set(xticks=x, xticklabels=[g["label"] for g in panel_groups],
                   ylim=(ymin, ymax * 1.14), ylabel="EDP gain over mono WS (×)")
            ticks = [v for v in ax.get_yticks() if 1 < v < ax.get_ylim()[1]]
            ax.set_yticks([1, *ticks])
            ax.axhline(1, color="0.45", linestyle="--", linewidth=.8, zorder=3)
            ax.grid(axis="y", linestyle=":", linewidth=.5, color="0.8")
            ax.set_axisbelow(True)
            # Visible break marks on both sides disclose the omitted baseline.
            for side in (0, 1):
                ax.plot([side - .008, side + .008], [-.018, .018], transform=ax.transAxes,
                        color="0.25", linewidth=1, clip_on=False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .96), ncol=2, frameon=False)
        fig.suptitle(f"{title} at HBM4 ({high / 1e12:g} TB/s)", fontsize=11, y=.995)
        fig.text(.5, .015, "Per-workload E2E; mono WS = 1×.  Truncated y-axes start at 0.9×; higher is better.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(.005, .07, .995, .83), w_pad=2)
        text_checks += figure_check(fig)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{name}.{ext}", dpi=180)
        plt.close(fig)
    # User decision end
    return dict(curve_points_checked=curve_checks, main_figure_reused_byte_exact=reuse_main is not None,
                bars_checked=bar_checks, visible_text_bounds_checked=text_checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--previous-analysis", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--style-source", type=Path, required=True)
    parser.add_argument("--reuse-main", type=Path, help="Reuse checked main-figure files while revising only ablations")
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
        normalized.append(dict(**r, display_name=label(arch), baseline_architecture="WS", baseline_edp_J_s=ws["edp_J_s"],
                               edp_gain_vs_ws=float(ws["edp_J_s"]) / float(r["edp_J_s"])))
    # User decision end
    general = [r for r in normalized if r["architecture"] in MAIN]
    ablations = []
    for variant in ("square", "independent_noskew"):
        architectures = {"WS", "Tessera-8", f"Tessera-8-{variant}"}
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
        for arch in (*MAIN, "Tessera-8-square", "Tessera-8-independent_noskew"):
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

    rendered = render(out, groups, high, args.reuse_main.resolve() if args.reuse_main else None)
    report = ["# Joint EDP evaluation normalized to mono WS", "",
              f"Run: `{run}`. These are modeled NeuSim estimates. Every displayed gain is derived from this run's verified `totals.csv`.", "",
              "Tessera in the figures and headline tables denotes the Tessera-8 configuration. The main comparison and both ablations show only this Tessera granularity. CSV configuration IDs retain the original architecture names; `display_name` gives the figure label.", "",
              "Every figure uses EDP gain = mono WS EDP / design EDP at the same workload and bandwidth. Mono WS is exactly 1; higher is better. The WS instance is the monolithic array recorded in configs.json. EDP is total system energy times total serial execution time, including the tensor-parallel chip count in energy.", "",
              "[General bandwidth comparison](general_edp_bandwidth.png) retains per-workload curves. [Non-square ablation](edp_nonsquare.png) and [interconnection ablation](edp_interconnect.png) compare each workload with paired vertical bars. Blue bars are full Tessera; orange bars are the ablated designs. Every value remains normalized to the same workload's mono WS.", "",
              "Each pair reports that workload's E2E result; all sourced workloads share one panel in each ablation. Sources and execution scope remain in metadata. The linear y-axes start at 0.9x, with explicit break marks and a dashed mono WS = 1x line. This truncation reduces space below the baseline; bar labels give the actual normalized gains.", "",
              "This update reuses the existing independent raw-row/weight audit and changes normalization and presentation only. Joint per-GEMM SRAM/array selection, finite SRAM reuse, HBM energy and native timing/power are unchanged. Per-GEMM selection is not a global graph EDP search. Power remains rough common-coefficient modeling; physical link power is not separately calibrated.", "",
              "The original energy breakdown is preserved byte-for-byte in `energy_breakdown.csv`. Detailed normalized data for all saved configurations, including the skew diagnostic, are in `normalized_edp.csv`. `summary_edp.csv` contains geometric means and workload ranges.", "",
              "HBM4 gains (derived from the same totals):", "",
              "| Workload | Mono WS | Planaria-32 | SOSA | FlexSA | SISA | Tessera |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in headlines:
        report.append("| " + r["workload"] + " | " + " | ".join(f"{r[arch + '_EDP_gain_vs_WS']:.6f}" for arch in MAIN) + " |")
    report.extend(["", "HBM4 ablated-design gains over mono WS:", "",
                   "| Workload | Tessera square only | Tessera w/o interconnect |",
                   "|---|---:|---:|"])
    controls = ("Tessera-8-square", "Tessera-8-independent_noskew")
    for r in headlines:
        report.append("| " + r["workload"] + " | " + " | ".join(f"{r[arch + '_EDP_gain_vs_WS']:.6f}" for arch in controls) + " |")
    (out / "REPORT.md").write_text("\n".join(report) + "\n")
    sources = [run / "manifest.json", run / "totals.csv", run / "configs.json", previous / "verification.json",
               previous / "artifact_hashes.json", args.groups.resolve(), args.style_source.resolve(), Path(__file__).resolve()]
    if args.reuse_main:
        sources.extend(args.reuse_main.resolve() / name for name in
                       ("artifact_hashes.json", "general_edp.csv", "general_edp_bandwidth.png", "general_edp_bandwidth.pdf"))
    result = dict(status="PASS", decision=f"{getpass.getuser()} ruled: show paired vertical ablation bars for each workload, with the mono WS = 1 reference close to the bottom; retain Tessera-8 named Tessera.",
                  accounting="WS EDP / design EDP; same workload and bandwidth; each bar is one workload; y-axes start at 0.9x",
                  raw_simulation_repeated=False, inherited_raw_audit=str(previous / "verification.json"),
                  raw_output_hashes_checked=len(manifest["output_sha256"]), readback_ratios_checked=checked_ratios,
                  summary_rows_checked=len(summaries), source_sha256={str(p): sha(p) for p in sources}, **rendered)
    save_json(out / "verification.json", result)
    save_json(out / "artifact_hashes.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != "source_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
