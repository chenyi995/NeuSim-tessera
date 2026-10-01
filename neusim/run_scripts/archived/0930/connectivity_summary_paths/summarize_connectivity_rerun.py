"""Publish a verified before/after connectivity comparison and reproduction notes."""
import argparse
from decimal import Decimal as D
import json
from pathlib import Path
from neusim.run_scripts.prepare_feature_cnn_qos import ROOT,rows,sha,save_csv,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False)
    replay=a.run/'arrival_replay_parallel_v2'
    audit=json.loads((replay/'verification.json').read_text());assert audit['status']=='PASS'
    assert sha(replay/'totals.csv')==audit['artifacts']['totals.csv']
    new={(r['workload_id'],r['architecture']):r for r in rows(replay/'totals.csv')}
    oldroot=ROOT/'results/tessera/20260930_async_full_v1/arrival_replay_v3'
    old={(r['workload_id'],r['architecture']):r for r in rows(oldroot/'totals.csv')
         if r['bandwidth_bytes_per_second']=='2800000000000'}
    labels=ROOT.parents[1]/'Tessera-HPCA2027-revision/fig/plotting/data/neusim_interconnect_delay_20260930/ablation_contribution.csv'
    results=[]
    for label in rows(labels):
        wid=label['workload_id'];c=new[wid,'Tessera-8'];d=new[wid,'Tessera-8-independent_noskew']
        oc=old[wid,'Tessera-8'];od=old[wid,'Tessera-8-independent_noskew']
        results.append(dict(workload=label['workload'],workload_id=wid,
            latency_increase_pct=100*(D(c['time_ns'])/D(d['time_ns'])-1),
            energy_reduction_pct=100*(1-D(c['energy_J'])/D(d['energy_J'])),
            edp_reduction_pct=100*(1-D(c['edp_J_s'])/D(d['edp_J_s'])),
            connected_latency_ms=D(c['time_ns'])/D(10**6),disconnected_latency_ms=D(d['time_ns'])/D(10**6),
            connected_system_energy_J=c['system_energy_J'],disconnected_system_energy_J=d['system_energy_J'],
            connected_array_utilization_pct=100*D(c['array_active_utilization']),
            disconnected_array_utilization_pct=100*D(d['array_active_utilization']),
            old_latency_increase_pct=100*(D(oc['time_ns'])/D(od['time_ns'])-1),
            old_energy_reduction_pct=100*(1-D(oc['energy_J'])/D(od['energy_J']))))
    save_csv(a.out/'comparison.csv',results)
    # save_csv independently reads back every value without reducing precision.
    config=json.loads((a.run/'main_costs/configs.json').read_text())['Tessera-8@2800000000000']
    checks=json.loads((a.run/'verification_v2/verification.json').read_text())
    weighted=json.loads((a.run/'cost_audit/verification.json').read_text())
    lines=['# Matched connectivity rerun','',
        'Current results: `comparison.csv`. Positive latency increase means the connected design is slower; positive energy and EDP reductions favor connectivity.',
        '',f"Both designs have {config['sa_dim']} x {config['sa_dim']} PEs, grain {config['tessera_parameters']['grain']}, {config['vmem_size_MB']} MiB SRAM, and identical native energy coefficients.",
        'The common joint search retains every native divisor tile and adds dyadic extents; full and residual tiles are charged and replayed explicitly. This is a finite candidate search, not a guarantee of optimality over every integer extent.',
        'HBM4 bandwidth, original arrivals (including initial idle), operator dependencies, finite SRAM, shared HBM/vector service, and padded compute/SRAM energy are retained. Both designs enable the same asynchronous admission policy.',
        '', 'The old divisor-only results are preserved in `../20260930_async_full_v1`. The diagnostic counterexamples are preserved there in `cacheblend_tile_search_diagnostic_v1`. The rerun changes both sides of the connectivity pair.',
        '', 'Validation:',
        f"- {checks['scalar_counter_checks']} vector/scalar counter comparisons; {checks['native_energy_checks']} native-energy checks; {checks['literal_schedule_checks']} literal schedule comparisons.",
        f"- {weighted['raw_rows']} raw records and {weighted['weighted_checks']} weighted-total checks; {weighted['edp_nonregression_checks']} GEMM EDP comparisons against the original candidates.",
        f"- {audit['records']} complete E2E records; completed checkpoints reused only after source-hash checks.",
        '', '| Workload | Latency increase | Energy reduction | EDP reduction |',
        '|---|---:|---:|---:|']
    for r in results:lines.append(f"| {r['workload']} | {r['latency_increase_pct']:+.2f}% | {r['energy_reduction_pct']:.2f}% | {r['edp_reduction_pct']:.2f}% |")
    lines+=['','Reproduction:','```sh',
        '.venv/bin/python -m neusim.run_scripts.verify_tessera_remainders --out <fresh-check-dir>',
        '.venv/bin/python -m neusim.run_scripts.run_tessera_joint --base results/tessera/archived/0929/20260929_paper_sram_per_pe --verification <fresh-check-dir>/verification.json --interconnect --workers 56 --memory-gb 196 --out <fresh-cost-dir>',
        '.venv/bin/python -m neusim.run_scripts.replay_tessera_arrivals index --costs <fresh-cost-dir> --out <fresh-store-dir> --workers 1 --memory-gb 16',
        '.venv/bin/python -m neusim.run_scripts.replay_tessera_arrivals run --costs <fresh-cost-dir> --store <fresh-store-dir> --programs results/tessera/20260930_async_full_v1/trace_programs_v2 --out <fresh-replay-dir> --workers 3 --memory-gb 96',
        '```','The completed run uses `arrival_replay_parallel_v2` to parallelize independent workloads with the same replay routine; `arrival_replay` retains interrupted serial-job checkpoints. All predecessor attempts and logs remain append-only.','']
    (a.out/'README.md').write_text('\n'.join(lines))
    refs=[replay/'verification.json',replay/'totals.csv',oldroot/'totals.csv',labels,Path(__file__)]
    save_json(a.out/'verification.json',dict(status='PASS',workloads=len(results),
        source_sha256={str(p.resolve()):sha(p) for p in refs},artifacts={p.name:sha(p) for p in a.out.iterdir() if p.is_file()}))
    print('\n'.join(lines[14:14+len(results)+3]))


if __name__=='__main__':main()
