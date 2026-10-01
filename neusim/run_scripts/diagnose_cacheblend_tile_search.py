"""Reproduce saved mappings and test a legal omitted SRAM tile; no model edits."""
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.npusim.backend.tessera_partitioned import evaluate_candidate, geometries
from neusim.npusim.backend.util import get_factors

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/tessera/20260930_async_full_v1'
PAPER=ROOT.parent.parent/'Tessera-HPCA2027-revision'
OUT=RUN/'cacheblend_tile_search_diagnostic_v1'


def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:1])
    OUT.mkdir(exist_ok=False)
    source=PAPER/'fig/plotting/data/cacheblend_connectivity_20260930_v2/diagnostic.json'
    saved=json.loads(source.read_text())
    configs=json.loads((RUN/'main_costs/configs.json').read_text())
    results=[]
    for example in saved['examples']:
        shape=tuple(int(example['shape'][k]) for k in ('M','N','K'))
        m,n,k=shape
        row=dict(cohort=example['cohort'],shape=shape,m_candidates=list(map(int,get_factors(m))),variants=[])
        for record in example['rows']:
            arch=record['architecture']
            chip=ChipConfig.model_validate(configs[arch+'@'+record['bandwidth_bytes_per_second']])
            mapping=json.loads(record['mapping_json'])
            jobs=[('saved_mapping',tuple(mapping['geometry']),tuple(mapping['memory_tile']))]
            if 'independent' in arch:
                # Diagnostic candidate only, not a fitted model constant:
                # keep the saved producer count and fill the physical slots via N parallelism;
                # stream a region-width multiple of M instead of the full prime-length stream.
                g=tuple(mapping['geometry']);grain=int(chip.tessera_parameters['grain'])
                slots=chip.sa_dim**2//(g[-1]*g[-2])
                tile=(chip.sa_dim,slots//g[3]*grain,chip.sa_dim)
                assert tile[0] not in row['m_candidates']
                jobs.append(('omitted_nondisivor_tile',g,tile))
            for tag,g,tile in jobs:
                assert g in set(geometries(m,n,k,chip))
                with contextlib.redirect_stdout(io.StringIO()):
                    op=evaluate_candidate(1,m,n,k,'DT_BFLOAT16',g,chip,forced_tile=tile)
                s=op.stats;d=s.tessera_details
                assert d['useful_macs']==m*n*k
                assert d['charged_sa_macs']>=m*n*k
                assert d['peak_live_bytes']<=chip.vmem_size_MB*1024**2
                if tag=='saved_mapping':
                    assert s.execution_time_ns==int(record['time_ns'])
                    assert math.isclose(s.total_energy_J,float(record['energy_J']),rel_tol=1e-12)
                    assert d['sram_bytes']==int(record['sram_bytes'])
                row['variants'].append(dict(tag=tag,architecture=arch,geometry=g,tile=tile,
                    time_ns=s.execution_time_ns,sa_ns=s.sa_time_ns,energy_J=s.total_energy_J,
                    edp_J_ns=s.total_energy_J*s.execution_time_ns,hbm_bytes=d['hbm_bytes'],
                    sram_bytes=d['sram_bytes'],peak_live_bytes=d['peak_live_bytes'],
                    charged_sa_macs=d['charged_sa_macs'],rounds=d['rounds']))
        original=next(v for v in row['variants'] if v['tag']=='saved_mapping' and 'independent' in v['architecture'])
        alt=next(v for v in row['variants'] if v['tag']!='saved_mapping')
        row['counterexample_improves_edp']=alt['edp_J_ns']<original['edp_J_ns']
        row['counterexample_improves_latency']=alt['time_ns']<original['time_ns']
        results.append(row)
    output=dict(status='PASS',results=results,conclusion='The existing divisor-only search omits capacity-feasible nondivisor tiles. This is a search-space diagnostic, not a revised workload result.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,RUN/'main_costs/configs.json',Path(__file__)]})
    (OUT/'diagnostic.json').write_text(json.dumps(output,indent=2))
    assert json.loads((OUT/'diagnostic.json').read_text())==json.loads(json.dumps(output))
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
