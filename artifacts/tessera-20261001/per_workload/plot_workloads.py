"""Render two wide, linear-axis PPA plots per sourced workload from verified CSVs."""
import csv
from decimal import Decimal
import hashlib
import html
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / '20261001_ppa_two_tilings_v1'
OUT = HERE / 'figures'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def save_csv(path, records):
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    back = read_rows(path)
    assert back == [{key: str(value) for key, value in r.items()} for r in records]


def main():
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-1:])
    OUT.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator, ScalarFormatter
    from matplotlib.transforms import Bbox

    assert json.loads((SOURCE / 'completion.json').read_text())['status'] == 'PASS'
    labels = {'native_tail': 'Normal tiling + tail packing', 'joint_edp': 'EDP tiling'}
    short = {'native_tail': 'normal', 'joint_edp': 'edp'}
    data = {}
    inputs = [Path(__file__), SOURCE / 'completion.json']
    for mode in labels:
        folder = SOURCE / mode / 'analysis'
        verification = json.loads((folder / 'verification.json').read_text())
        assert verification['status'] == 'PASS'
        source_csv = folder / 'workload_metrics.csv'
        assert sha(source_csv) == verification['artifacts'][source_csv.name]
        records = read_rows(source_csv)
        assert len(records) == 156
        data[mode] = {(r['workload_id'], r['architecture']): r for r in records}
        inputs.extend((source_csv, folder / 'verification.json'))
    ids = list(dict.fromkeys(key[0] for key in data['native_tail']))
    assert len(ids) == 12 and set(data['native_tail']) == set(data['joint_edp'])
    families = [
        ('Tessera', '#519000', 'o', [f'Tessera-{g}' for g in (64, 32, 16, 8)]),
        ('Planaria', '#168bc0', 's', [f'Planaria-{g}' for g in (64, 32, 16, 8)]),
        ('Independent WS', '#626262', '^', ['WS'] + [f'WS-independent-{g}' for g in (64, 32, 16, 8)]),
    ]
    arches = [arch for _, _, _, values in families for arch in values]
    plotted = []
    for mode, rows in data.items():
        for wid in ids:
            reference = rows[wid, 'WS']
            for arch in arches:
                r = rows[wid, arch]
                assert r['useful_ops'] == reference['useful_ops']
                assert int(r['total_pes_per_chip']) == 128 * 128
                ops = Decimal(2) * Decimal(r['useful_macs']) * Decimal(r['chips'])
                x = ops / Decimal(r['seconds']) / Decimal(r['total_area_mm2']) / Decimal('1e9')
                y = ops / Decimal(r['system_energy_J']) / Decimal('1e9')
                assert math.isclose(float(x), float(r['performance_density_gops_per_mm2']), rel_tol=2e-12)
                assert math.isclose(float(y), float(r['energy_efficiency_gops_per_w']), rel_tol=2e-12)
                plotted.append(dict(tiling_policy=mode, **r))
    save_csv(OUT / 'all_points.csv', plotted)

    # chenyi9: decision start — wider plots make the existing bends easier to see.
    # Only the linear display aspect and margins change; all original points and
    # the physical grain order are retained, without interpolation or fitting.
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.labelsize': 9, 'axes.titlesize': 10,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    single_size = (6.1, 3.15)
    pair_size = (12.2, 3.35)
    # chenyi9: decision end
    handles = [Line2D([], [], color=color, marker=marker, markersize=4,
                      linewidth=1.3, label=name) for name, color, marker, _ in families]
    panels = []

    def panel(ax, mode, wid):
        annotations = []
        xx, yy = [], []
        for name, color, marker, names in families:
            records = [data[mode][wid, arch] for arch in names]
            x = [float(r['performance_density_gops_per_mm2']) for r in records]
            y = [float(r['energy_efficiency_gops_per_w']) for r in records]
            line, = ax.plot(x, y, color=color, marker=marker, markersize=4.2,
                            linewidth=1.35, label=name, zorder=3)
            assert list(line.get_xdata()) == x and list(line.get_ydata()) == y
            annotations.extend((r['grain'], a, b, color) for r, a, b in zip(records, x, y))
            xx.extend(x)
            yy.extend(y)
        xr, yr = max(xx) - min(xx), max(yy) - min(yy)
        assert xr > 0 and yr > 0
        ax.set_xlim(max(0, min(xx) - .14 * xr), max(xx) + .16 * xr)
        ax.set_ylim(max(0, min(yy) - .17 * yr), max(yy) + .19 * yr)
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.set_xlabel(r'E2E performance / area (GOPS/mm$^2$)', labelpad=4)
        ax.set_ylabel('Energy efficiency (GOPS/W)', labelpad=4)
        ax.set_title(labels[mode], pad=6)
        ax.grid(color='#d5d9de', linestyle=':', linewidth=.6, alpha=.9)
        ax.spines[['top', 'right']].set_visible(False)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            formatter = ScalarFormatter(useOffset=False)
            formatter.set_scientific(False)
            axis.set_major_formatter(formatter)
        ax.tick_params(length=3, width=.7)
        return annotations

    def label_points(fig, ax, points):
        """Place integer grain labels without covering markers or nearby labels."""
        renderer = fig.canvas.get_renderer()
        marker_boxes = []
        for _, x, y, _ in points:
            px, py = ax.transData.transform((x, y))
            marker_boxes.append(Bbox.from_extents(px - 4, py - 4, px + 4, py + 4))
        placed = []
        offsets = [(8, 7), (8, -10), (-8, 7), (-8, -10), (0, 13),
                   (0, -15), (14, 0), (-14, 0), (14, 17), (-14, 17),
                   (14, -20), (-14, -20), (24, 9), (-24, 9),
                   (24, -12), (-24, -12), (0, 24), (0, -26)]

        def overlap(a, b):
            return max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))

        for label, x, y, color in points:
            text = ax.annotate(label, (x, y), xytext=offsets[0], textcoords='offset points',
                               fontsize=7.7, color=color, zorder=5,
                               bbox=dict(boxstyle='round,pad=0.06', facecolor='white',
                                         edgecolor='none', alpha=.83))
            candidates = []
            for dx, dy in offsets:
                text.set_position((dx, dy))
                text.set_ha('left' if dx > 0 else 'right' if dx < 0 else 'center')
                text.set_va('bottom' if dy > 0 else 'top' if dy < 0 else 'center')
                box = text.get_window_extent(renderer).expanded(1.08, 1.12)
                outside = box.width * box.height - overlap(box, ax.bbox)
                collisions = sum(overlap(box, other) for other in placed)
                markers = sum(overlap(box, other) for other in marker_boxes)
                score = max(0, outside) * 1e7 + collisions * 1e5 + markers * 1e4 + dx * dx + dy * dy
                candidates.append((score, dx, dy, box))
            _, dx, dy, box = min(candidates, key=lambda v: v[0])
            text.set_position((dx, dy))
            text.set_ha('left' if dx > 0 else 'right' if dx < 0 else 'center')
            text.set_va('bottom' if dy > 0 else 'top' if dy < 0 else 'center')
            placed.append(box)

    def draw(wid, modes):
        paired = len(modes) == 2
        fig, axes = plt.subplots(1, len(modes), figsize=pair_size if paired else single_size,
                                  squeeze=False)
        fig.subplots_adjust(left=.063 if paired else .13, right=.985, bottom=.225,
                            top=.715, wspace=.25)
        title = data['native_tail'][wid, 'WS']['workload']
        fig.suptitle(title, y=.985, fontsize=11, fontweight='medium')
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .93),
                   ncol=3, frameon=False, fontsize=8.5, handlelength=2, columnspacing=2)
        fig.text(.5, .045, 'Equal 128×128 PEs · HBM4 · Point labels: grain / WS subarray side · Linear axes',
                 ha='center', va='center', fontsize=7, color='#555555')
        pending = []
        for ax, mode in zip(axes.flat, modes):
            pending.append((ax, panel(ax, mode, wid)))
        fig.canvas.draw()
        for ax, annotations in pending:
            label_points(fig, ax, annotations)
            assert ax.get_xscale() == ax.get_yscale() == 'linear'
            panels.append(dict(workload_id=wid, tiling_policy=modes[len(panels) % len(modes)] if not paired else
                               modes[list(axes.flat).index(ax)], paired=paired,
                               xlim=list(ax.get_xlim()), ylim=list(ax.get_ylim()),
                               axes_width_height_ratio=ax.bbox.width / ax.bbox.height))
        return fig

    pages = []
    with PdfPages(OUT / 'all_workloads.pdf') as book:
        for wid in ids:
            folder = OUT / wid
            folder.mkdir()
            save_csv(folder / 'data.csv', [r for r in plotted if r['workload_id'] == wid])
            for mode in labels:
                fig = draw(wid, [mode])
                for ext in ('png', 'pdf'):
                    fig.savefig(folder / (short[mode] + '.' + ext), dpi=200, facecolor='white')
                plt.close(fig)
            fig = draw(wid, list(labels))
            for ext in ('png', 'pdf'):
                fig.savefig(folder / ('comparison.' + ext), dpi=190, facecolor='white')
            book.savefig(fig, facecolor='white')
            plt.close(fig)
            pages.append(dict(workload_id=wid, workload=data['native_tail'][wid, 'WS']['workload']))

    lines = ['# Per-workload normal and EDP tiling figures', '',
             'Status: PASS. Each sourced workload has two separate figures and a side-by-side comparison. '
             'These plots use the completed analytical NeuSim results from `20261001_ppa_two_tilings_v1`; '
             'there is no new simulation or mapping selection.', '',
             '[All workloads, one comparison per PDF page](all_workloads.pdf) · [HTML gallery](index.html) · '
             '[Full-precision plotted data](all_points.csv)', '',
             'The horizontal axis is useful E2E GOPS divided by array-plus-SRAM area; the vertical axis is useful GOPS/W. '
             'Each workload is plotted separately in absolute units. Both axes are linear. '
             'Axis limits are chosen separately for each panel and are printed on the axes. '
             'The plot area is wider than before; neither nonlinear transforms nor fitted curves are used. '
             'Straight segments connect the recorded grain points in physical-grain order.', '',
             'Tessera and Planaria sweep grains 64, 32, 16, 8. Independent WS connects 1×128×128, 4×64×64, '
             '16×32×32, 64×16×16, and 256×8×8 arrays. Every configuration has 128×128 total PEs. '
             'Labels next to markers are the fission grain or independent WS subarray side. '
             'Tessera packing, recorded arrivals, finite SRAM, padded accesses and revision energy accounting '
             'remain as in the completed run. RTL contributes area estimates only.', '',
             'Normal tiling is the native NeuSim tiler plus full-array bulk execution and fractured tail packing. '
             'Its known small-K-tile choices remain unchanged. The other panel uses the completed joint EDP mappings.', '',
             '| Workload | Normal tiling | EDP tiling | Comparison | Data |', '|---|---|---|---|---|']
    cards = []
    for p in pages:
        wid, name = p['workload_id'], p['workload']
        lines.append(f'| {name} | [PNG]({wid}/normal.png) · [PDF]({wid}/normal.pdf) | '
                     f'[PNG]({wid}/edp.png) · [PDF]({wid}/edp.pdf) | '
                     f'[PNG]({wid}/comparison.png) · [PDF]({wid}/comparison.pdf) | [CSV]({wid}/data.csv) |')
        cards.append(f'<section><h2>{html.escape(name)}</h2><a href="{wid}/comparison.pdf">'
                     f'<img src="{wid}/comparison.png" alt="{html.escape(name)} normal and EDP tiling"></a>'
                     f'<p><a href="{wid}/normal.pdf">Normal PDF</a> · <a href="{wid}/edp.pdf">EDP PDF</a> · '
                     f'<a href="{wid}/data.csv">Full-precision data</a></p></section>')
    lines += ['', 'Validation: source CSV hashes match the completed run; every plotted coordinate is independently '
              'recomputed from useful MACs, elapsed time, system energy and total area. Written CSVs are read back '
              'and all line coordinates are checked against them. `../plot_workloads.py` reproduces this display.', '']
    doc = '\n'.join(lines)
    (OUT / 'README.md').write_text(doc)
    assert (OUT / 'README.md').read_text() == doc
    gallery = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    gallery += '<title>Per-workload PPA: normal and EDP tiling</title><style>body{max-width:1280px;margin:32px auto;padding:0 18px;font:16px/1.5 system-ui;color:#222;background:#f5f6f8}h1{font-size:25px}h2{font-size:19px}section{background:white;padding:18px;margin:22px 0;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}a{color:#176b9b}p{margin:8px 0}</style><body>'
    gallery += '<h1>Per-workload PPA: normal and EDP tiling</h1><p>Left: native tiling + tail packing. Right: joint EDP tiling. Linear axes; each panel has its own printed scale.</p>'
    gallery += '<p><a href="all_workloads.pdf">All comparisons (PDF)</a> · <a href="all_points.csv">All points (CSV)</a></p>'
    gallery += ''.join(cards) + '</body></html>'
    (OUT / 'index.html').write_text(gallery)
    assert (OUT / 'index.html').read_text() == gallery
    result = dict(status='PASS', workloads=len(ids), individual_figures=2 * len(ids),
                  comparison_figures=len(ids), pdf_pages=len(ids), plotted_records=len(plotted),
                  accounting='Existing full E2E simulation data; absolute linear axes; wider display only; no fitting.',
                  source_sha256={str(p): sha(p) for p in inputs}, panels=panels,
                  artifacts={str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()})
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    assert json.loads((OUT / 'verification.json').read_text()) == result
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'artifacts', 'panels')}, indent=2))


if __name__ == '__main__':
    main()
