"""Plot all sourced workloads separately and retain every grain-pair comparison."""
import csv,hashlib,json,math,os,sys
from decimal import Decimal
from pathlib import Path

RUN=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):
    with path.open() as f:return list(csv.DictReader(f))


def save(path,data):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    back=rows(path);assert len(back)==len(data)
    assert all(all(str(v)==b[k] for k,v in a.items()) for a,b in zip(data,back))


def main():
    mode=sys.argv[1];assert mode in ('native_tail','joint_edp')
    folder=RUN/mode;report=folder/'analysis';out=folder/'workloads';out.mkdir(exist_ok=False)
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-1:])
    audit=json.loads((report/'verification.json').read_text());assert audit['status']=='PASS'
    for key,digest in audit['artifacts'].items():assert sha(report/key)==digest
    raw=rows(report/'workload_metrics.csv');ids=list(dict.fromkeys(r['workload_id'] for r in raw))
    lookup={(r['workload_id'],r['architecture']):r for r in raw}
    pairs=[]
    for wid in ids:
        for coarse,fine in ((64,32),(64,16),(64,8),(32,16),(32,8),(16,8)):
            c,f=(lookup[wid,f'Tessera-{g}'] for g in (coarse,fine))
            speed=float(c['seconds'])/float(f['seconds']);eff=float(c['system_energy_J'])/float(f['system_energy_J'])
            density=float(f['performance_density_gops_per_mm2'])/float(c['performance_density_gops_per_mm2'])
            pairs.append(dict(workload_id=wid,workload=f['workload'],coarse_grain=coarse,fine_grain=fine,
                e2e_speedup=speed,energy_reduction_pct=100*(1-1/eff),energy_efficiency_gain=eff,
                performance_density_gain=density,edp_gain=speed*eff,
                coarse_seconds=c['seconds'],fine_seconds=f['seconds'],coarse_energy_J=c['system_energy_J'],fine_energy_J=f['system_energy_J'],
                coarse_array_utilization=c['array_utilization'],fine_array_utilization=f['array_utilization'],
                coarse_intrinsic_array_seconds=c['intrinsic_array_seconds'],fine_intrinsic_array_seconds=f['intrinsic_array_seconds'],
                coarse_padding_macs=int(c['charged_sa_macs'])-int(c['useful_macs']),
                fine_padding_macs=int(f['charged_sa_macs'])-int(f['useful_macs']),
                both_plot_axes_improve=(density>1+1e-10 and eff>1+1e-10)))
    save(out/'grain_comparisons.csv',pairs)
    for r in rows(out/'grain_comparisons.csv'):
        c,f=(lookup[r['workload_id'],'Tessera-'+r[k]] for k in ('coarse_grain','fine_grain'))
        for field,col in (('e2e_speedup','seconds'),('energy_efficiency_gain','system_energy_J')):
            assert math.isclose(float(r[field]),float(Decimal(c[col])/Decimal(f[col])),rel_tol=2e-12)
    native={(r['workload_id'],r['architecture']):r for r in rows(folder/'arrival_replay/totals.csv')}
    energy=[];independent=[]
    for (wid,arch),r in native.items():
        fields={k:float(v)*int(r['chips']) for k,v in r.items() if k.startswith(('dynamic_energy_','static_energy_'))}
        assert math.isclose(sum(fields.values()),float(r['system_energy_J']),rel_tol=2e-12)
        energy.append(dict(workload_id=wid,architecture=arch,system_energy_J=r['system_energy_J'],**fields,
            sram_bytes_per_chip=r['sram_bytes'],hbm_bytes_per_chip=r['hbm_bytes'],reduction_ops_per_chip=r['reduction_ops']))
    for wid in ids:
        for g in (64,32,16,8):
            t,w=native[wid,f'Tessera-{g}'],native[wid,f'WS-independent-{g}']
            independent.append(dict(workload_id=wid,workload=lookup[wid,'WS']['workload'],grain=g,
                tessera_latency_change_pct=100*(float(t['seconds'])/float(w['seconds'])-1),
                tessera_energy_reduction_pct=100*(1-float(t['system_energy_J'])/float(w['system_energy_J'])),
                ws_to_tessera_sram_traffic=float(w['sram_bytes'])/float(t['sram_bytes']),
                ws_to_tessera_hbm_traffic=float(w['hbm_bytes'])/float(t['hbm_bytes'])))
    save(out/'energy_breakdown.csv',energy);save(out/'connected_vs_independent_ws.csv',independent)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    def panel(ax,wid):
        specs=[('Tessera','#519000','o',[f'Tessera-{g}' for g in (64,32,16,8)]),
            ('Planaria','#168bc0','s',[f'Planaria-{g}' for g in (64,32,16,8)]),
            ('Independent WS','#555555','^',['WS']+[f'WS-independent-{g}' for g in (64,32,16,8)])]
        for family,color,marker,arches in specs:
            vv=[lookup[wid,a] for a in arches]
            xs=[float(r['performance_density_vs_ws']) for r in vv];ys=[float(r['energy_efficiency_vs_ws']) for r in vv]
            line,=ax.plot(xs,ys,color=color,marker=marker,markersize=4,linewidth=1.1,label=family)
            assert list(line.get_xdata())==xs and list(line.get_ydata())==ys
            for r,x,y in zip(vv,xs,ys):
                ax.annotate(r['grain'],(x,y),xytext=(3,5 if family=='Tessera' else -10),textcoords='offset points',fontsize=7,color=color)
        ax.set_title(lookup[wid,'WS']['workload'],fontsize=9);ax.grid(alpha=.2);ax.margins(x=.15,y=.2)
    title='Native tiling + tail packing' if mode=='native_tail' else 'Joint EDP tiling'
    fig,axes=plt.subplots(3,4,figsize=(13,8.6),squeeze=False)
    for ax,wid in zip(axes.flat,ids):panel(ax,wid)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.962),frameon=False)
    fig.suptitle(title+' · equal 128×128 PEs · HBM4',y=.997,fontsize=12)
    fig.supxlabel('E2E performance / area (monolithic WS = 1)');fig.supylabel('Energy efficiency (monolithic WS = 1)')
    fig.tight_layout(rect=(.025,.025,1,.925),h_pad=2,w_pad=1.4)
    for ext in ('png','pdf'):fig.savefig(out/('per_workload.'+ext),dpi=200,bbox_inches='tight')
    plt.close(fig)
    for wid in ids:
        fig,ax=plt.subplots(figsize=(5.2,3.9));panel(ax,wid)
        ax.set_xlabel('E2E performance / area (WS = 1)');ax.set_ylabel('Energy efficiency (WS = 1)')
        ax.legend(fontsize=7);fig.tight_layout();fig.savefig(out/(wid+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    current=[r for r in pairs if r['coarse_grain']==32 and r['fine_grain']==8]
    gains=[r for r in pairs if r['both_plot_axes_improve']]
    lines=['# '+title+': separate workloads','',
        'All metrics are derived from completed analytical NeuSim simulations. The same 128×128 total PEs, '
        'finite SRAM, HBM4, revision energy coefficients, padded work, arrivals and dependencies apply to every design. '
        'The WS line connects 1×128², 4×64², 16×32², 64×16² and 256×8² independent arrays. '
        'WS area is the equal-PE monolithic proxy requested by chenyi9; RTL affects area only. '
        'Tessera enables both within-round packing and asynchronous use of free physical regions.', '',
        '## Tessera grain 32 to grain 8','',
        '| Workload | E2E speedup | Energy reduction | Performance/area gain |','|---|---:|---:|---:|']
    for r in current:lines.append(f"| {r['workload']} | {r['e2e_speedup']:.2f}× | {r['energy_reduction_pct']:.2f}% | {r['performance_density_gain']:.2f}× |")
    lines+=['','## All finer-grain pairs improving both plot axes','',
        'All pairs, including regressions, remain in `grain_comparisons.csv`.','',
        '| Workload | Coarse → fine | Energy-efficiency gain | Performance/area gain |','|---|---:|---:|---:|']
    for r in gains:lines.append(f"| {r['workload']} | {r['coarse_grain']} → {r['fine_grain']} | {r['energy_efficiency_gain']:.2f}× | {r['performance_density_gain']:.2f}× |")
    if not gains:lines.append('No pair improves both plotted axes under the requested area estimate.')
    lines+=['','## Files','',
        '- `../analysis/`: geometric-mean plot, full-precision workload metrics and methodology.',
        '- `per_workload.pdf` and twelve workload PNGs: every complete source workload separately.',
        '- `energy_breakdown.csv`: native compute, SRAM, HBM, vector, ICI, static/background and regulator-inclusive energy.',
        '- `connected_vs_independent_ws.csv`: performance, energy and memory traffic at every matched grain.',
        '- All written numbers are copied or computed by this retained script; CSVs and plot coordinates are read back.', '']
    (out/'README.md').write_text('\n'.join(lines));assert (out/'README.md').read_text()=='\n'.join(lines)
    summary=dict(status='PASS',mapping_policy=mode,workloads=len(ids),grain_pairs=len(pairs),finer_both_axes_cases=gains,
        source_sha256={str(p):sha(p) for p in [Path(__file__),report/'verification.json',report/'workload_metrics.csv',folder/'arrival_replay/totals.csv']},
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert json.loads((out/'verification.json').read_text())==summary
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_sha256','artifacts')},indent=2))


if __name__=='__main__':main()
