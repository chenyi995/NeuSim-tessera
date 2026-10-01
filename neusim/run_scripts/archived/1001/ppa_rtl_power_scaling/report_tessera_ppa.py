"""Plot verified equal-PE native E2E results, with full-precision source tables."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

from neusim.run_scripts.prepare_feature_cnn_qos import rows,sha,save_csv,save_json
from neusim.run_scripts.run_tessera_ppa import ROOT,RTL_PLOT,RTL_REPORT,ENERGY,rtl_data


def gm(values):
    values=list(values)
    assert values and all(v>0 for v in values)
    return math.exp(math.fsum(math.log(v) for v in values)/len(values))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    run=a.run.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-1:])
    replay=run/'arrival_replay';costs=run/'main_costs'
    audit=json.loads((replay/'verification.json').read_text());assert audit['status']=='PASS'
    assert sha(replay/'totals.csv')==audit['artifacts']['totals.csv']
    manifest=json.loads((costs/'manifest.json').read_text());assert manifest['status']=='PASS'
    config_records=json.loads((costs/'ppa_configurations.json').read_text())
    configs=json.loads((costs/'configs.json').read_text())
    inputs=Path(manifest['base'])/'inputs_snapshot'
    groups=json.loads((inputs/'workload_groups.json').read_text())
    ids=[g['workload_id'] for g in groups]
    name_file=ROOT/'results/tessera/20260930_energy_corrected_v1/paper_data_2dp/workload_sources.csv'
    names={r['workload_id']:r['workload'] for r in rows(name_file)}
    assert set(ids)==set(names)
    values={(r['workload_id'],r['architecture']):r for r in rows(replay/'totals.csv')}
    raw={(r['workload_id'],r['architecture']):r for r in rows(costs/'totals.csv')}
    assert len(config_records)==11 and len(ids)==12
    detailed=[];summary=[];source=rtl_data()
    for c in config_records:
        arch=c['architecture'];config=configs[arch+'@2800000000000']
        assert config['sa_dim']**2*config['num_sa']==c['total_pes']==128**2
        expected_area=16*sum(source[k][c['rtl_index']] for k in ('A_FMA','A_LOG','A_SRAM'))
        assert math.isclose(c['area_mm2'],expected_area,rel_tol=1e-14)
        local=[]
        for wid in ids:
            r=values[wid,arch];baseline=values[wid,'WS'];original=raw[wid,arch]
            seconds=float(r['seconds']);energy=float(r['system_energy_J']);chips=int(r['chips'])
            macs=int(r['useful_macs']);useful_ops=2*macs*chips
            assert macs==int(baseline['useful_macs'])==int(original['useful_macs'])
            assert float(r['hbm_bytes'])==float(original['hbm_bytes'])
            assert float(r['charged_sa_macs'])>=macs
            assert seconds>0 and energy>0
            components=math.fsum(float(v) for k,v in r.items() if k.startswith(('static_energy_','dynamic_energy_')))
            assert math.isclose(components*chips,energy,rel_tol=2e-11)
            total_area=c['area_mm2']*chips
            throughput=useful_ops/seconds
            efficiency=useful_ops/energy
            density=throughput/total_area
            # Independent physical ceilings: one MAC per PE per cycle and
            # at least the calibrated Joules proxy for each useful operation.
            peak_ops=chips*2*config['sa_dim']**2*config['num_sa']*config['freq_GHz']*1e9
            assert throughput<=peak_ops*(1+1e-10),(wid,arch,throughput,peak_ops)
            assert efficiency<=1e12/c['compute_pj_per_op']*(1+1e-10),(wid,arch,efficiency)
            row=dict(workload_id=wid,workload=names[wid],architecture=arch,grain=c['grain'],
                chips=chips,total_pes_per_chip=c['total_pes'],area_mm2_per_chip=c['area_mm2'],
                total_area_mm2=total_area,seconds=seconds,system_energy_J=energy,
                useful_ops=useful_ops,throughput_gops=throughput/1e9,
                energy_efficiency_gops_per_w=efficiency/1e9,
                performance_density_gops_per_mm2=density/1e9,
                e2e_speedup_vs_ws=float(baseline['seconds'])/seconds,
                energy_efficiency_vs_ws=float(baseline['system_energy_J'])/energy,
                performance_density_vs_ws=float(baseline['seconds'])/seconds*config_records[0]['area_mm2']/c['area_mm2'],
                e2e_edp_gain_vs_ws=float(baseline['edp_J_s'])/float(r['edp_J_s']),
                average_system_power_W=energy/seconds,
                array_utilization=float(r['array_active_utilization']),
                e2e_array_utilization=float(r['e2e_array_utilization']),
                idle_fraction=float(r.get('idle_ns',0))/float(r['time_ns']),
                compute_energy_fraction=float(r['dynamic_energy_sa_J'])/float(r['energy_J']),
                sram_energy_fraction=(float(r['dynamic_energy_sram_J'])+float(r['static_energy_sram_J']))/float(r['energy_J']),
                hbm_energy_fraction=(float(r['dynamic_energy_hbm_J'])+float(r['static_energy_hbm_J']))/float(r['energy_J']),
                intrinsic_array_seconds=float(original['sa_ns'])/1e9,
                hbm_service_seconds=float(original['hbm_ns'])/1e9,
                vector_service_seconds=float(original['vu_ns'])/1e9,
                useful_macs=macs,charged_sa_macs=r['charged_sa_macs'],padding_sram_read_bytes=r['padding_sram_read_bytes'],
                dispatch=r['dispatch'])
            detailed.append(row);local.append(row)
        summary.append(dict(**c,workloads=len(local),
            **{key:gm(r[key] for r in local) for key in
               ('throughput_gops','energy_efficiency_gops_per_w','performance_density_gops_per_mm2',
                'e2e_speedup_vs_ws','energy_efficiency_vs_ws','performance_density_vs_ws','e2e_edp_gain_vs_ws')},
            mean_sram_energy_fraction=math.fsum(r['sram_energy_fraction'] for r in local)/len(local),
            mean_idle_fraction=math.fsum(r['idle_fraction'] for r in local)/len(local)))
    save_csv(out/'workload_metrics.csv',detailed);save_csv(out/'geomean.csv',summary)
    # Independently read CSVs back and recompute the plotted means and WS ratios.
    back=list(rows(out/'workload_metrics.csv'))
    for r in rows(out/'geomean.csv'):
        vv=[x for x in back if x['architecture']==r['architecture']]
        for key in ('energy_efficiency_gops_per_w','performance_density_gops_per_mm2',
                    'energy_efficiency_vs_ws','performance_density_vs_ws'):
            independent=math.prod(float(x[key])**(1/len(vv)) for x in vv)
            assert math.isclose(independent,float(r[key]),rel_tol=2e-12)
    assert len(back)==len(config_records)*len(ids)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    by={r['architecture']:r for r in summary}
    fig,ax=plt.subplots(figsize=(6.5,4.3))
    artists=[]
    for family,color,marker in [('Tessera','#519000','o'),('Planaria','#168bc0','s')]:
        vv=[by[f'{family}-{g}'] for g in (64,32,16,8)]
        xs=[v['performance_density_gops_per_mm2'] for v in vv]
        ys=[v['energy_efficiency_gops_per_w'] for v in vv]
        line,=ax.plot(xs,ys,color=color,marker=marker,markersize=6,linewidth=1.5,label=family)
        artists.append((line,vv))
        for i,v in enumerate(vv):
            offsets = ({64:(-7,10),32:(-8,-14),16:(-8,2),8:(5,-13)} if family=='Planaria'
                       else {64:(5,6),32:(5,6),16:(8,-5),8:(8,-4)})
            offset=offsets[v['grain']]
            ax.annotate('d='+str(v['grain']),xy=(xs[i],ys[i]),xytext=offset,
                textcoords='offset points',fontsize=9,color=color,
                ha='right' if offset[0]<0 else 'left')
    for arch,label,color,marker in [('WS',r'1 $\times$ WS 128$\times$128','#4b4b4b','D'),
        ('WS-independent-8',r'256 $\times$ WS 8$\times$8','#c87c00','^'),
        ('WS-independent-16',r'64 $\times$ WS 16$\times$16','#9467bd','v')]:
        r=by[arch]
        line,=ax.plot([r['performance_density_gops_per_mm2']],[r['energy_efficiency_gops_per_w']],
            marker=marker,color=color,linestyle='None',markersize=8,label=label)
        artists.append((line,[r]))
    ax.set_xlabel('E2E throughput / (array + SRAM area)\n'+r'(GOPS/mm$^2$)')
    ax.set_ylabel('E2E energy efficiency (GOPS/W)')
    ax.grid(True,linestyle=':',linewidth=.6,alpha=.5)
    ax.spines[['top','right']].set_visible(False)
    ax.margins(x=.13,y=.15)
    ax.legend(loc='best',fontsize=8.5,framealpha=.93)
    ax.set_title('Equal 128×128 PEs · HBM4 · geometric mean of 12 workloads',fontsize=10)
    fig.tight_layout()
    for line,vv in artists:
        assert list(line.get_xdata())==[v['performance_density_gops_per_mm2'] for v in vv]
        assert list(line.get_ydata())==[v['energy_efficiency_gops_per_w'] for v in vv]
    for extension in ('pdf','png','svg'):
        fig.savefig(out/f'energy_efficiency_vs_performance_area.{extension}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    lines=['# Equal-PE E2E performance density and energy efficiency','',
        'Status: PASS. Results are analytical NeuSim simulations. All configurations contain 128×128 PEs.',
        '', '## Method', '',
        'Each marker is the equal-weight geometric mean of the same twelve complete sourced workloads. '
        'Performance is useful matrix arithmetic operations divided by E2E elapsed time; one MAC is two operations. '
        'Energy efficiency is useful operations per joule. Tensor-parallel chips contribute both energy and area. '
        'The denominator is the scaled RTL array-plus-SRAM area, excluding controllers as in the source figure; '
        'it is not a measured complete accelerator die area.', '',
        'All designs use HBM4 (2.8 TB/s), 12 MiB finite live SRAM, 1 GHz, and the same native SRAM-tile/array EDP selector. '
        'HBM rereads and padded compute/SRAM accesses are charged. Recorded arrival idle and dependencies are retained. '
        'Tessera uses corrected within-round packing and asynchronous reuse of free physical regions for independent adjacent requests. '
        'Planaria retains its legal long compositions and conventional schedule. The independent WS banks use all their arrays in parallel.', '',
        'Area is derived from the matching 32×32 RTL tier multiplied by sixteen: RTL d=2,4,8,16 maps to d=8,16,32,64. '
        'The three WS configurations use the monolithic WS area scaled to the same PE count, as requested. '
        'Two connected curves represent the fission sweep; each fixed WS configuration is one marker.', '',
        'Compute uses the period-corrected Joules non-SRAM array proxy per executed arithmetic operation. '
        'This proxy includes active array logic and leakage and is not an isolated FMA measurement. '
        'The WS coefficient also applies to both independent WS configurations. '
        'The user-requested non-compute scaling is an estimate: native SRAM dynamic/static coefficients are multiplied by '
        'the RTL P_SRAM ratio to WS; the array static coefficient is multiplied by the RTL P_LOG ratio to WS. '
        'Common HBM, VU, ICI, and platform-background coefficients have scale one. '
        'No additional RTL dynamic-logic term is added on top of Joules. '
        'These ratios describe a rough power model at fixed SRAM capacity, not new gate-level power measurements.', '',
        '## Geometric means (derived; monolithic WS = 1)', '',
        '| Architecture | Area (mm²/chip) | E2E speed | Performance/area | Energy efficiency |',
        '|---|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['architecture']} | {r['area_mm2']:.2f} | {r['e2e_speedup_vs_ws']:.2f} | {r['performance_density_vs_ws']:.2f} | {r['energy_efficiency_vs_ws']:.2f} |")
    lines += ['', '## Inputs and validation', '',
        f'- Area and non-compute ratios: `{RTL_PLOT}`; synthesis provenance: `{RTL_REPORT}`.',
        f'- Compute calibration: `{ENERGY}`.',
        f'- Source workload names and citations: `{name_file}`.',
        '- `../verification/verification.json`: original FissionSA kernels, scalar/vector counters, native energy, and full-grid utilization checks.',
        '- `../main_costs/`: fresh EDP mappings, invocation weights, per-operator records, and configurations.',
        '- `../arrival_replay/`: arrival/dependency-preserving native spatial replay.',
        '- `workload_metrics.csv`: complete per-workload metrics, power components, padding and utilization.',
        '- `geomean.csv`: full-precision plotted coordinates and normalized comparisons.',
        '- Plot coordinates and all geometric means were read back and independently recomputed.', '']
    (out/'README.md').write_text('\n'.join(lines))
    source_paths=[Path(__file__),costs/'ppa_configurations.json',costs/'configs.json',costs/'totals.csv',
        replay/'verification.json',replay/'totals.csv',RTL_PLOT,RTL_REPORT,ENERGY,name_file]
    save_json(out/'verification.json',dict(status='PASS',configurations=len(summary),workloads=len(ids),
        workload_rows=len(detailed),source_sha256={str(p):sha(p) for p in source_paths},
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()},
        accounting='Equal-PE, equal-workload-weight geometric mean; executed compute uses Joules; non-compute uses explicit RTL-ratio proxies.'))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
