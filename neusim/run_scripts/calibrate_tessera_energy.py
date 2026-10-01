"""Derive the array coefficients from the archived 700 MHz Joules plateau."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent.parent / 'SA-rtl/tessera-hpca/array_joules_v5'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    out = parser.parse_args().out
    runner = SOURCE / 'run_array_gemm_power_v5.sh'
    tb = SOURCE / 'tb/tb_array_gemm_power_tessera0.sv'
    plot = SOURCE / 'plot_power_timeline_all.py'
    period = float(re.search(r'CLK_PERIOD_NS=([\d.]+)', runner.read_text())[1])
    side = int(re.search(r'localparam int D = (\d+)', tb.read_text())[1])
    assert period == 2 * float(re.search(r'always #([\d.]+)', tb.read_text())[1])
    threshold = float(re.search(r'thr = ([\d.]+) \* peak', plot.read_text())[1])
    sources = [runner, tb, plot, Path(__file__)]
    records = {}
    # chenyi9: decision start -- correct the missing period; retain source precision.
    for arch, tag in [('WS', 'ws2_32'), ('Tessera', 'tessera0_32'), ('Planaria', 'planaria2_direct_32')]:
        path = SOURCE / 'results' / tag / f'arr_{tag}_gemm_timeline.csv'
        with path.open() as f:
            rows = list(csv.DictReader(f))
        assert all(float(r['memory_mW']) == 0 for r in rows)
        power = [float(r['total_mW']) - float(r['memory_mW']) for r in rows]
        power[0] = power[1]
        plateau = [v for v in power if v >= threshold * max(power)]
        mean = math.fsum(plateau) / len(plateau)
        pj = mean * period / (2 * side**2)
        independent = (mean / 1000) / ((2 * side**2) / (period * 1e-9)) * 1e12
        assert math.isclose(pj, independent, rel_tol=1e-14)
        records[arch] = dict(pj_per_op=pj, display_pj_per_op=f'{pj:.3f}',
            plateau_mW=mean, plateau_frames=len(plateau), source_csv=str(path))
        sources.append(path)
    # chenyi9: decision end
    result = dict(status='PASS', clock_period_ns=period, ops_per_cycle=2*side**2,
        designs=records, operations_per_MAC=2,
        accounting='Period-corrected non-SRAM total plateau per arithmetic op; used as the existing rough NeuSim activity-energy proxy. Native static power and regulator accounting are retained. This is not an isolated FMA or separately calibrated dynamic-only measurement.',
        proxy_groups={'SISA': 'Planaria', 'SOSA': 'Planaria', 'FlexSA': 'Planaria'},
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    assert json.loads(out.read_text()) == result
    print(json.dumps(records, indent=2))

if __name__ == '__main__':
    main()
