"""Plot unchanged simulator coordinates and report all exploratory candidates."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures_labeled'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save_csv(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with path.open() as f:
        assert list(csv.DictReader(f))==[{k:str(v) for k,v in r.items()} for r in rows]


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='PASS'
    for n,h in verification['output_sha256'].items():assert sha(HERE/n)==h
    source=json.loads((HERE/'source_manifest.json').read_text())
    area=json.loads((HERE/'area.json').read_text())
    label_records=[]
    with (HERE/'all_points.csv').open() as f: rows=list(csv.DictReader(f))
    data={(r['workload_id'],r['tiling_policy'],r['architecture']):r for r in rows}
    ids=[w['workload_id'] for w in source['workloads']]
    titles={w['workload_id']:w['workload'] for w in source['workloads']}
    modes=['native_tail','joint_edp']
    labels={'native_tail':'Normal tiling + tail packing','joint_edp':'EDP tiling'}
    families=[('Tessera (fission)','#519000','o',[f'Tessera-{g}' for g in (64,32,16,8)]),
              ('Planaria (fission)','#168bc0','s',[f'Planaria-{g}' for g in (64,32,16,8)]),
              ('WS (independent arrays)','#333333','^',['WS']+[f'WS-independent-{g}' for g in (64,32,16,8)])]
    for r in rows:
        ops=2*float(r['useful_macs'])
        assert math.isclose(float(r['performance_density_gops_per_mm2']),ops/float(r['seconds'])/float(r['area_mm2'])/1e9,rel_tol=2e-12)
        assert math.isclose(float(r['energy_efficiency_gops_per_w']),ops/float(r['energy_J'])/1e9,rel_tol=2e-12)
    OUT.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from matplotlib.transforms import Bbox
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'axes.titlesize':10,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42})
    handles=[Line2D([],[],color=c,marker=m,linewidth=1.35,markersize=4,label=n) for n,c,m,_ in families]

    def draw(ax,wid,mode):
        pts=[];xx=[];yy=[]
        for family,color,marker,arches in families:
            rr=[data[wid,mode,a] for a in arches]
            x=[float(r['performance_density_gops_per_mm2']) for r in rr]
            y=[float(r['energy_efficiency_gops_per_w']) for r in rr]
            line,=ax.plot(x,y,color=color,marker=marker,linewidth=1.35,markersize=4.2,zorder=3)
            assert list(line.get_xdata())==x and list(line.get_ydata())==y
            for r,a,b in zip(rr,x,y):
                g=int(r['grain']); meta=area[r['architecture']]
                assert meta['grain']==g and meta['subarrays']*g*g==meta['total_pes']
                label=(f'Fission {g}×{g}' if meta['family'] in ('Tessera','Planaria')
                       else f"{meta['subarrays']} × ({g}×{g})")
                pts.append((label,a,b,color))
                label_records.append(dict(workload_id=wid,tiling_policy=mode,
                    architecture=r['architecture'],label=label,x=a,y=b))
            xx.extend(x);yy.extend(y)
        xr=max(xx)-min(xx);yr=max(yy)-min(yy)
        ax.set_xlim(max(0,min(xx)-.12*xr),max(xx)+.17*xr)
        ax.set_ylim(max(0,min(yy)-.14*yr),max(yy)+.18*yr)
        ax.set_xlabel(r'Attention E2E performance / area (GOPS/mm$^2$)')
        ax.set_ylabel('Energy efficiency (GOPS/W)')
        ax.grid(color='#d5d9de',linestyle=':',linewidth=.6)
        ax.spines[['top','right']].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_title(labels[mode])
        return pts

    def annotate(fig,ax,points):
        renderer=fig.canvas.get_renderer();placed=[];markers=[]
        for _,x,y,_ in points:
            px,py=ax.transData.transform((x,y));markers.append(Bbox.from_extents(px-4,py-4,px+4,py+4))
        def overlap(a,b):return max(0,min(a.x1,b.x1)-max(a.x0,b.x0))*max(0,min(a.y1,b.y1)-max(a.y0,b.y0))
        offsets=[(8,7),(8,-10),(-8,7),(-8,-10),(0,14),(0,-15),(15,0),(-15,0),(16,18),(-16,18),(22,-16),(-22,-16),(30,9),(-30,9),(0,28),(0,-28),(0,40),(0,-40),(35,30),(-35,30),(35,-30),(-35,-30),(50,0),(-50,0)]
        for label,x,y,color in points:
            t=ax.annotate(label,(x,y),xytext=(0,0),textcoords='offset points',fontsize=7,color=color,arrowprops=dict(arrowstyle='-',color=color,lw=.45,alpha=.65,shrinkA=2,shrinkB=4),
                bbox=dict(boxstyle='round,pad=0.04',facecolor='white',edgecolor='none',alpha=.8))
            candidates=[]
            for dx,dy in offsets:
                t.set_position((dx,dy));t.set_ha('left' if dx>0 else 'right' if dx<0 else 'center');t.set_va('bottom' if dy>0 else 'top' if dy<0 else 'center')
                box=t.get_window_extent(renderer).expanded(1.08,1.12)
                score=max(0,box.width*box.height-overlap(box,ax.bbox))*1e7+sum(overlap(box,z) for z in placed)*1e5+sum(overlap(box,z) for z in markers)*1e4+dx*dx+dy*dy
                candidates.append((score,dx,dy,box))
            _,dx,dy,box=min(candidates,key=lambda z:z[0]);placed.append(box)
            t.set_position((dx,dy));t.set_ha('left' if dx>0 else 'right' if dx<0 else 'center');t.set_va('bottom' if dy>0 else 'top' if dy<0 else 'center')

    with PdfPages(OUT/'all_workloads.pdf') as pdf:
        for wid in ids:
            fig,axes=plt.subplots(1,2,figsize=(12.2,3.7))
            points=[draw(ax,wid,mode) for ax,mode in zip(axes,modes)]
            suffix=' (model-config-derived KV trajectory)' if wid.startswith('falcon') else ''
            fig.suptitle(titles[wid]+suffix,fontsize=11,y=.99)
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
            fig.text(.5,.018,'Attention operator E2E; equal 128×128 PEs. Fission labels = minimum subarray size; WS labels = array count × (rows × columns).',ha='center',fontsize=8,color='#555555')
            fig.subplots_adjust(left=.065,right=.986,bottom=.18,top=.78,wspace=.25)
            fig.canvas.draw()
            for ax,pts in zip(axes,points):annotate(fig,ax,pts)
            fig.savefig(OUT/(wid+'.pdf'));fig.savefig(OUT/(wid+'.png'),dpi=170);pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(len(ids),2,figsize=(12.2,12.4))
    overview_points=[]
    for i,wid in enumerate(ids):
        for j,mode in enumerate(modes):
            pts=draw(axes[i,j],wid,mode)
            overview_points.append((axes[i,j],pts))
            axes[i,j].set_title(titles[wid]+' | '+labels[mode],fontsize=10)
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
    fig.subplots_adjust(left=.065,right=.985,bottom=.05,top=.96,hspace=.53,wspace=.25)
    fig.canvas.draw()
    for ax,pts in overview_points:annotate(fig,ax,pts)
    fig.savefig(OUT/'overview.png',dpi=160);fig.savefig(OUT/'overview.pdf');plt.close(fig)

    # Each pair and the overview must annotate every original point once.
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
