"""Verify both completed figures and derive the final comparison from their CSVs."""
import csv,hashlib,json,math
from pathlib import Path

RUN=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))
def gm(values):
    values=list(values);return math.exp(math.fsum(math.log(x) for x in values)/len(values))


def main():
    completion=json.loads((RUN/'completion.json').read_text());assert completion['status']=='PASS'
    all_data={};all_pairs={};sources=[Path(__file__),RUN/'completion.json',RUN/'combined_verification.json']
    summaries=[]
    for mode in ('native_tail','joint_edp'):
        folder=RUN/mode
        for sub in ('analysis','workloads'):
            check=json.loads((folder/sub/'verification.json').read_text());assert check['status']=='PASS'
            for name,digest in check['artifacts'].items():assert sha(folder/sub/name)==digest
            for name,digest in check['source_sha256'].items():assert sha(Path(name))==digest
            sources.append(folder/sub/'verification.json')
        d={r['architecture']:r for r in rows(folder/'analysis/geomean.csv')};all_data[mode]=d
        p=rows(folder/'workloads/grain_comparisons.csv');all_pairs[mode]=p
        s=rows(folder/'analysis/workload_metrics.csv');assert len(s)==156
        for arch,summary in d.items():
            vv=[r for r in s if r['architecture']==arch]
            assert len(vv)==12
            for field in ('energy_efficiency_gops_per_w','performance_density_gops_per_mm2'):
                assert math.isclose(float(summary[field]),gm(float(r[field]) for r in vv),rel_tol=2e-12)
        wins=[r for r in p if r['both_plot_axes_improve']=='True']
        g32=[r for r in p if r['coarse_grain']=='32' and r['fine_grain']=='8']
        summaries.append(dict(mapping_policy=mode,simulated_workload_rows=len(s),
            both_axes_improving_pairs=wins,
            grain32_to_8_geomean=dict(e2e_speedup=gm(float(r['e2e_speedup']) for r in g32),
                energy_efficiency_gain=gm(float(r['energy_efficiency_gain']) for r in g32),
                performance_density_gain=gm(float(r['performance_density_gain']) for r in g32))))
        sources += [folder/'analysis/geomean.csv',folder/'analysis/workload_metrics.csv',folder/'workloads/grain_comparisons.csv']
    a,b=(json.loads((RUN/m/'main_costs/configs.json').read_text()) for m in ('native_tail','joint_edp'))
    assert set(a)==set(b)
    for arch,c in a.items():
        for field,value in c.items():
            if 'power' in field or field in ('sa_dim','num_sa','freq_GHz','vmem_size_MB','hbm_bw_GBps'):
                assert value==b[arch][field],(arch,field)
    for arch in all_data['native_tail']:
        assert all_data['native_tail'][arch]['area_mm2']==all_data['joint_edp'][arch]['area_mm2']
    lines=['# Native bulk/tail versus joint EDP tiling','',
        'Status: PASS. Both policies completed all thirteen equal-PE configurations and twelve sourced workloads. '
        'Results are analytical NeuSim simulations. All numerical summaries below are derived from the retained CSVs.', '',
        '- [Native tiling + tail packing](native_tail/analysis/energy_efficiency_vs_performance_area.pdf)',
        '- [Joint EDP tiling](joint_edp/analysis/energy_efficiency_vs_performance_area.pdf)',
        '- [Native per-workload plots and comparisons](native_tail/workloads/README.md)',
        '- [EDP per-workload plots and comparisons](joint_edp/workloads/README.md)', '',
        'Both figures plot energy efficiency against E2E performance per area. Each point is the equal-weight geometric '
        'mean of the twelve workloads. The independent WS line connects 1×128², 4×64², 16×32², 64×16², and 256×8² arrays; '
        'Tessera and Planaria each vary the fission grain over 64, 32, 16, and 8. Every configuration has the same total PE count.', '',
        'chenyi9 ruled: energy uses the corrected revision NeuSim accounting, without RTL power multipliers. '
        'The existing calibrated compute coefficients and native SRAM/HBM/static/VU/ICI/regulator terms are retained. '
        'Actual padded compute and SRAM reads are charged; HBM rereads follow finite SRAM reuse. '
        'RTL is used only for the requested area estimate: the matching 32-scale design is scaled to the equal-PE fabric. '
        'Independent WS area uses the equal-PE monolithic WS proxy.', '',
        'The native policy calls NeuSim find_best_tile_shape_for_matmul through the revision memory_tile wrapper. '
        'That wrapper reserves double-buffered operands and FP32 partial sums. Its producer-plane reservation is raised '
        'when the selected tile requires concurrent residual reductions, and the same native formula is called again. '
        'Within each SRAM tile, complete 128×128 K/N weight blocks execute first on the full connected array. '
        'Residual K/N strips and corners fracture to the configured grain. Mixed residual groups fill free physical regions '
        'and reuse retired regions. Each occupied grain cell remains reserved until its last planned local use; this is a '
        'conservative reservation for admission of another request, not an assumption of zero-cost instantaneous remapping. '
        'Independent WS retains its fixed subarray size.', '',
        'The EDP policy searches the common capacity-feasible divisor/dyadic SRAM tiles and each architecture’s legal '
        'geometries for minimum native E2E EDP per GEMM. It preserves the existing finite search rather than claiming a global '
        'whole-workload optimum. Both policies preserve native FlashAttention outer Br/Bc tiling and apply their array '
        'policy to resident QK/PV phases. Tessera enables within-round packing and asynchronous admission of independent '
        'adjacent requests; recorded arrivals, pure idle, dependencies and pinned live SRAM are retained.', '',
        '## Geometric means, normalized to monolithic WS within each policy','',
        '| Policy | Architecture | E2E speed | Energy efficiency | Performance/area |','|---|---|---:|---:|---:|']
    for mode,d in all_data.items():
        for arch,r in d.items():
            lines.append(f"| {mode} | {arch} | {float(r['e2e_speedup_vs_ws']):.2f}× | {float(r['energy_efficiency_vs_ws']):.2f}× | {float(r['performance_density_vs_ws']):.2f}× |")
    lines+=['','## Validation','',
        '- `common_verification/`: original baseline kernels, scalar/vector counters, native energy and revision power coefficients.',
        '- `native_verification_v2/`: literal weight-block counters, native tiler equality, physical non-overlap, bank-event timing, FlashAttention and asynchronous replay.',
        '- Both policy directories retain fresh operator profiles, complete replay checkpoints, configuration files and source hashes.',
        '- This script independently recomputes geometric means and checks identical power, capacity, bandwidth and area inputs between the policies.',
        '- The initial native-plan check is retained as FAIL in `native_verification/`; the missing engine metadata was corrected before either full run.',
        '- The earlier RTL-scaled and energy-only attempts remain separate; neither contributes points to these two figures.', '']
    with (RUN/'README.md').open('x') as f:f.write('\n'.join(lines))
    assert (RUN/'README.md').read_text()=='\n'.join(lines)
    final=dict(status='PASS',summaries=summaries,source_sha256={str(p):sha(p) for p in sources},readme_sha256=sha(RUN/'README.md'))
    with (RUN/'final_verification.json').open('x') as f:json.dump(final,f,indent=2)
    assert json.loads((RUN/'final_verification.json').read_text())==final
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
