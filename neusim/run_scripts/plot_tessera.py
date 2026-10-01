"""Export standalone figures from retained, read-back checked simulation CSVs."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def plot_results(run):
    run = Path(run).resolve()
    if not run.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("figures must remain inside NeuSim")
    out = run / "figures"
    out.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(out / "matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    with (run / "summary.csv").open() as f:
        rows = list(csv.DictReader(f))
    if any(r["accounting"] == "fixed_trace_cumulative_service" for r in rows):
        rows = [r for r in rows if r["accounting"] == "fixed_trace_cumulative_service"]
    cases = list(dict.fromkeys(r["case"] for r in rows))
    groups = {case: {r["variant"]: r for r in rows if r["case"] == case} for case in cases}
    columns = ("array_energy_J", "sram_energy_J", "hbm_energy_J", "vector_energy_J", "ici_energy_J", "background_energy_J")
    labels = ("Array", "SRAM", "HBM", "Vector", "Inter-chip", "Array background")
    # Codex: decision start — all plotted values derive from the saved principal table.
    values = {case: {"energy_fraction": {col: float(g["tessera"][col]) / float(g["tessera"]["energy_J"])
                                        for col in columns},
                     "ratios": {v: {key: float(r[key]) for key in ("latency_over_tessera", "energy_over_tessera", "edp_over_tessera")}
                                for v, r in g.items()}} for case, g in groups.items()}
    target = out / "plot_data.json"
    target.write_text(json.dumps(values, indent=2) + "\n")
    assert json.loads(target.read_text()) == values
    source = Path(__file__)
    (out / source.name).write_bytes(source.read_bytes())
    assert (out / source.name).read_bytes() == source.read_bytes()
    # Codex: decision end
    short = [s.replace("_conv300", "").replace("_batch0", "").replace("_batch1", "") for s in cases]
    x = np.arange(len(cases))
    width = max(7, len(cases) * 0.6)
    fig, ax = plt.subplots(figsize=(width, 4.8))
    bottom = np.zeros(len(cases))
    for col, label in zip(columns, labels):
        height = np.array([values[c]["energy_fraction"][col] for c in cases]) * 100
        ax.bar(x, height, bottom=bottom, label=label)
        bottom += height
    assert np.allclose(bottom, 100, atol=1e-10, rtol=0)
    ax.set(xticks=x, xticklabels=short, ylabel="Tessera modeled energy (%)", ylim=(0, 100))
    ax.tick_params(axis="x", labelrotation=40)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"energy_breakdown.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    comparisons = {"non_square": ("square", "square_fixed"),
                   "independent_arrays": ("independent",), "skew": ("skew",)}
    for name, variants in comparisons.items():
        fig, axes = plt.subplots(3, 1, figsize=(width, 8.5), sharex=True)
        for ax, key, label in zip(axes, ("latency_over_tessera", "energy_over_tessera", "edp_over_tessera"),
                                 ("Service time / Tessera", "Energy / Tessera", "EDP / Tessera")):
            bar_width = 0.8 / len(variants)
            for i, v in enumerate(variants):
                y = [values[c]["ratios"][v][key] for c in cases]
                ax.bar(x + (i - (len(variants)-1)/2) * bar_width, y, bar_width, label=v)
            ax.axhline(1, color="black", linewidth=0.7, linestyle="--")
            ax.set_ylabel(label)
            ax.legend(fontsize=8)
            if max(float(groups[c][v][key]) for c in cases for v in variants) > 10:
                ax.set_yscale("log")
        axes[-1].set(xticks=x, xticklabels=short)
        axes[-1].tick_params(axis="x", labelrotation=40)
        fig.suptitle("Analytical estimates; identical resources and workloads")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    (out / "checks.json").write_text(json.dumps(dict(plot_readback="PASS", energy_percent_sum="PASS",
        source=str(run / "summary.csv"), cases=len(cases)), indent=2) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    plot_results(parser.parse_args().run)
