"""Derive useful array utilization from audited E2E workload totals."""
import argparse
from datetime import datetime
from decimal import Decimal
import getpass
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
from neusim.run_scripts.report_tessera_ws import MAIN, close, key, label, rows, save_csv, save_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--readme',type=Path,required=True)
    args=parser.parse_args();run=args.run.resolve();out=args.out.resolve();readme=args.readme.resolve()
    assert not out.exists()
    manifest=json.loads((run/'manifest.json').read_text())
    audit=json.loads((run/'analysis/verification.json').read_text())
    assert manifest['status']==audit['status']=='PASS' and not manifest['pilot']
    for name in ('totals.csv','configs.json'):
        assert sha(run/name)==manifest['output_sha256'][name]
        assert sha(run/name)==audit['source_sha256'][str(run/name)]
    configs=json.loads((run/'configs.json').read_text())
    data=rows(run/'totals.csv');lookup={key(r):r for r in data}
    groups_path=Path(manifest['base'])/'inputs_snapshot/workload_groups.json'
    groups=json.loads(groups_path.read_text())
    records=[]
    for r in data:
        wid,arch,bw=key(r);c=configs[f'{arch}@{bw}']
        # All matrix work in this run is assigned to SA, including resident QK/PV.
        assert not c['use_vu_for_small_matmul'] and not c['enable_dvfs']
        pe=c['num_sa']*c['sa_dim']**2
        macs=int(r['useful_macs']);sa=int(r['sa_ns']);elapsed=int(r['time_ns']);freq=c['freq_GHz']
        assert 0<sa<=elapsed and 0<=macs<=pe*freq*sa
        # chenyi9: decision start — report array utilization alongside EDP.
        # Codex: use useful MACs over the full physical fabric's SA-cycle capacity.
        # Source: saved useful_macs/sa_ns and config num_sa, sa_dim, freq_GHz.
        util=macs/(pe*freq*sa)
        e2e=macs/(pe*freq*elapsed)
        # chenyi9: decision end
        records.append(dict(workload_id=wid,workload=r['workload'],architecture=arch,display_name=label(arch),
            bandwidth_bytes_per_second=bw,chips=int(r['chips']),physical_PEs_per_chip=pe,
            frequency_GHz=freq,useful_sa_macs_per_chip=macs,sa_time_ns=sa,e2e_time_ns=elapsed,
            array_utilization_pct=100*util,e2e_array_utilization_pct=100*e2e))
    out.mkdir(parents=True)
    shutil.copy2(__file__,out/Path(__file__).name);assert sha(Path(__file__))==sha(out/Path(__file__).name)
    save_csv(out/'array_utilization.csv',records)
    for r in rows(out/'array_utilization.csv'):
        raw=lookup[r['workload_id'],r['architecture'],int(r['bandwidth_bytes_per_second'])]
        c=configs[f"{r['architecture']}@{r['bandwidth_bytes_per_second']}"]
        # Independent Decimal read-back from original totals, not the derived table.
        capacity=Decimal(c['num_sa'])*Decimal(c['sa_dim'])**2*Decimal(str(c['freq_GHz']))
        for field,time in (('array_utilization_pct','sa_ns'),('e2e_array_utilization_pct','time_ns')):
            expected=Decimal(100)*Decimal(raw['useful_macs'])/(capacity*Decimal(raw[time]))
            close(r[field],expected)
        assert int(r['useful_sa_macs_per_chip'])==int(raw['useful_macs'])
        assert float(r['e2e_array_utilization_pct'])<=float(r['array_utilization_pct'])<=100
    high=max(manifest['bandwidth_bytes_per_second'])
    by_key={key(r):r for r in records};headline=[]
    for g in groups:
        r=dict(workload_id=g['workload_id'],workload=g['name'])
        for arch in MAIN:r[arch+'_array_utilization_pct']=by_key[g['workload_id'],arch,high]['array_utilization_pct']
        headline.append(r)
    save_csv(out/'hbm4_array_utilization.csv',headline)
    note=['',f'{getpass.getuser()} requested array utilization alongside EDP.','',
        '`Array utilization = useful SA MACs / (all physical PEs per chip × frequency_GHz × summed SA time_ns)`. '
        'This is a derived useful-work ratio over the array compute windows, including their fill/drain cycles '
        'and idle physical partitions. Padding zero MACs are excluded from the numerator. Counts and times '
        'are accumulated using workload invocation weights before division; per-operator percentages are not averaged. '
        'Tensor-parallel chip factors cancel between useful work and total hardware capacity.','',
        'The CSV also labels `e2e_array_utilization_pct`, whose denominator uses total workload time and therefore '
        'includes memory and other waiting. The table below reports array compute-window utilization.','',
        f'HBM4 array utilization, derived from `{run}` at {high/1e12:g} TB/s:', '',
        '| E2E workload | '+' | '.join(label(a) for a in MAIN)+' |','|---|'+'---:|'*len(MAIN)]
    for r in rows(out/'hbm4_array_utilization.csv'):
        note.append('| '+r['workload']+' | '+' | '.join(f"{float(r[a+'_array_utilization_pct']):.2f}%" for a in MAIN)+' |')
    relative=out.relative_to(readme.parent)
    note+=['',f'Full-precision values for all bandwidths: [array_utilization.csv]({relative}/array_utilization.csv). '
        'These values describe the existing architecture-energy main run; the subsequently discussed padded-SRAM '
        'read-energy correction is not applied in that run.','']
    old=readme.read_text();text=old+'\n'.join(note)
    day=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%m%d')
    archive=readme.parent/'archived'/day/'array_utilization'/readme.name
    assert not archive.exists();archive.parent.mkdir(parents=True)
    shutil.copy2(readme,archive);assert sha(readme)==sha(archive)
    readme.write_text(text);assert readme.read_text()==text
    result=dict(status='PASS',simulation_repeated=False,rows_checked=len(records),
        accounting='Useful SA MACs / full physical PE-cycle capacity; weighted totals before division.',
        source_sha256={str(p):sha(p) for p in (run/'manifest.json',run/'totals.csv',run/'configs.json',
            run/'analysis/verification.json',groups_path,Path(__file__))},
        readme_archive=str(archive),readme_archive_sha256=sha(archive),readme_sha256=sha(readme))
    save_json(out/'verification.json',result)
    # Retain the previous publication record before updating its README hash.
    publication=run.parent/'publication_verification.json'
    previous=json.loads(publication.read_text())
    target=publication.parent/'archived'/day/'array_utilization'/publication.name
    assert not target.exists();target.parent.mkdir(parents=True)
    shutil.copy2(publication,target);assert sha(publication)==sha(target)
    previous['source_sha256'][str(readme)]=sha(readme)
    previous.update(array_utilization_audit=str(out/'verification.json'),
                    array_utilization_audit_sha256=sha(out/'verification.json'),
                    prior_publication_archive=str(target),prior_publication_archive_sha256=sha(target))
    publication.write_text(json.dumps(previous,indent=2)+'\n')
    assert json.loads(publication.read_text())==previous
    save_json(out/'artifact_hashes.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(status='PASS',readme=str(readme),rows_checked=len(records),hbm4=headline),indent=2))


if __name__=='__main__':main()
