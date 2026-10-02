"""Create the requested annotation-only plot variant; preserve previous outputs."""
from pathlib import Path

here=Path(__file__).resolve().parent
source=(here/'plot_and_report.py').read_text()
script=source.split('    # Comparable-grain ratios')[0]

def replace(old,new):
    global script
    assert old in script,old
    script=script.replace(old,new)

replace("OUT=HERE/'figures'", "OUT=HERE/'figures_labeled'")
replace("    source=json.loads((HERE/'source_manifest.json').read_text())",
        "    source=json.loads((HERE/'source_manifest.json').read_text())\n    area=json.loads((HERE/'area.json').read_text())\n    label_records=[]")
replace("('Tessera','#519000'", "('Tessera (fission)','#519000'")
replace("('Planaria','#168bc0'", "('Planaria (fission)','#168bc0'")
replace("('Independent WS','#333333'", "('WS (independent arrays)','#333333'")
replace("            pts.extend((r['grain'],a,b,color) for r,a,b in zip(rr,x,y));xx.extend(x);yy.extend(y)",
        """            for r,a,b in zip(rr,x,y):
                g=int(r['grain']); meta=area[r['architecture']]
                assert meta['grain']==g and meta['subarrays']*g*g==meta['total_pes']
                label=(f'Fission {g}×{g}' if meta['family'] in ('Tessera','Planaria')
                       else f\"{meta['subarrays']} × ({g}×{g})\")
                pts.append((label,a,b,color))
                label_records.append(dict(workload_id=wid,tiling_policy=mode,
                    architecture=r['architecture'],label=label,x=a,y=b))
            xx.extend(x);yy.extend(y)""")
replace("(0,28),(0,-28)]", "(0,28),(0,-28),(0,40),(0,-40),(35,30),(-35,30),(35,-30),(-35,-30),(50,0),(-50,0)]")
replace("fontsize=7.5,color=color,", "fontsize=7,color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.45,alpha=.65,shrinkA=2,shrinkB=4),")
replace("Complete attention operator E2E; QK + softmax + PV + memory. Equal 128 x 128 PEs; labels = minimum array side.",
        "Attention operator E2E; equal 128×128 PEs. Fission labels = minimum subarray size; WS labels = array count × (rows × columns).")
replace("    for i,wid in enumerate(ids):\n        for j,mode in enumerate(modes):",
        "    overview_points=[]\n    for i,wid in enumerate(ids):\n        for j,mode in enumerate(modes):")
replace("            pts=draw(axes[i,j],wid,mode)",
        "            pts=draw(axes[i,j],wid,mode)\n            overview_points.append((axes[i,j],pts))")
replace("    fig.savefig(OUT/'overview.png',dpi=160);",
        "    fig.canvas.draw()\n    for ax,pts in overview_points:annotate(fig,ax,pts)\n    fig.savefig(OUT/'overview.png',dpi=160);")
script+='''    # Each pair and the overview must annotate every original point once.
    assert len(label_records)==2*len(rows)
    assert label_records[:len(rows)]==label_records[len(rows):]
    for r in label_records:
        original=data[r['workload_id'],r['tiling_policy'],r['architecture']]
        assert r['x']==float(original['performance_density_gops_per_mm2'])
        assert r['y']==float(original['energy_efficiency_gops_per_w'])
    save_csv(OUT/'point_labels.csv',label_records[:len(rows)])
    check=dict(status='PASS',points=len(rows),labels_per_view=len(rows),
        source_sha256=sha(HERE/'all_points.csv'),
        artifacts={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    assert check['source_sha256']==verification['output_sha256']['all_points.csv']
    with (OUT/'verification.json').open('x') as f:json.dump(check,f,indent=2)
    assert json.loads((OUT/'verification.json').read_text())==check
    print('PASS: all point labels agree with recorded architecture, grain, and array count.')

if __name__=='__main__':main()
'''
path=here/'plot_labeled.py'
with path.open('x') as f:f.write(script)
assert path.read_text()==script
