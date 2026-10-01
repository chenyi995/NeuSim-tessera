"""Verify, restore and rerun the published Tessera workload artifact."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'artifacts/tessera-20261001'
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'


def sha(path, compressed=False):
    h = hashlib.sha256()
    with (gzip.open(path, 'rb') if compressed else path.open('rb')) as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = json.loads((ARTIFACT / 'manifest.json').read_text())
    assert manifest['status'] == 'PASS'
    for row in manifest['files']:
        path = ARTIFACT / row['file']
        assert path.resolve().is_relative_to(ARTIFACT.resolve())
        assert sha(path) == row['stored_sha256'], path
        assert sha(path, row['compression'] == 'gzip') == row['sha256'], path
    print('PASS artifact files:', len(manifest['files']), flush=True)
    return manifest


def restore(out):
    manifest = verify()
    out.mkdir(parents=True, exist_ok=False)
    for row in manifest['files']:
        zipped = row['compression'] == 'gzip'
        relative = row['file'][:-3] if zipped else row['file']
        target = out / relative
        assert target.resolve().is_relative_to(out.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        src = ARTIFACT / row['file']
        with (gzip.open(src, 'rb') if zipped else src.open('rb')) as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer, 1 << 20)
        assert sha(target) == row['sha256'], target
        assert target.stat().st_size == row['bytes']
    # Codex: decision start -- expose the original runner's input layout portably.
    base = out / 'base'
    base.mkdir()
    (base / 'inputs_snapshot').symlink_to('../inputs/snapshot', target_is_directory=True)
    (base / 'speed').symlink_to('../inputs/graphs', target_is_directory=True)
    (base / 'baseline_verification').mkdir()
    (base / 'baseline_verification/bandwidth_grid.json').symlink_to('../../inputs/bandwidth_grid.json')
    # Codex: decision end
    (out / 'restore_verification.json').write_text(json.dumps(dict(status='PASS', files=len(manifest['files']),
        artifact_manifest_sha256=sha(ARTIFACT / 'manifest.json')), indent=2) + '\n')
    print('PASS restored:', out, flush=True)


def profile(args):
    from neusim.configs.chips.ChipConfig import ChipConfig
    from neusim.run_scripts import run_tessera_joint as runner, run_tessera_ppa as ppa
    source = args.restored / 'ppa' / args.policy / 'main_costs'
    configs = json.loads((source / 'configs.json').read_text())
    area = json.loads((source / 'ppa_configurations.json').read_text())
    assert json.loads((args.restored / 'restore_verification.json').read_text())['status'] == 'PASS'
    assert all(c['tessera_parameters']['mapping_policy'] == args.policy for c in configs.values())

    def saved_configs(bandwidth, mapping):
        # chenyi9: preserve every saved physical and energy parameter for reruns.
        assert mapping == args.policy
        result = {}
        for key, value in configs.items():
            arch, bw = key.rsplit('@', 1)
            assert int(bw) == bandwidth
            result[arch, int(bw)] = ChipConfig.model_validate(value)
        return result, area

    ppa.configurations = saved_configs
    gate = args.out.with_name(args.out.name + '_source_check.json')
    sources = list((ROOT / 'neusim/npusim').rglob('*.py'))
    sources = [p for p in sources if 'archived' not in p.parts]
    sources += [Path(__file__), source / 'configs.json', source / 'ppa_configurations.json']
    gate.parent.mkdir(parents=True, exist_ok=True)
    with gate.open('x') as f:
        json.dump(dict(status='PASS', source_sha256={str(p):sha(p) for p in sources},
                       scope='Input/source identity for this rerun; functional tests are reported separately.'), f, indent=2)
    sys.argv = ['run_tessera_joint', '--base', str(args.restored / 'base'),
                '--verification', str(gate), '--equal-pe-ppa', '--ppa-tiling', args.policy,
                '--workers', str(args.workers), '--memory-gb', str(args.memory_gb), '--out', str(args.out)]
    if args.limit:
        sys.argv += ['--limit', str(args.limit)]
    runner.main()


def replay(args):
    import resource
    from neusim.run_scripts import replay_tessera_arrivals as runner
    from neusim.run_scripts.prepare_feature_cnn_qos import save_csv, save_json
    assert 1 <= args.workers <= 64 and 0 < args.memory_gb <= 200
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.workers])
    cap = int(args.memory_gb * 1e9) // (args.workers + 1)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.costs or args.restored / 'ppa' / args.policy / 'main_costs'
    costs = args.out / 'costs'
    costs.mkdir()
    for name in ('operator_costs.csv', 'configs.json', 'totals.csv'):
        (costs / name).symlink_to((source / name).resolve())
    metadata = json.loads((source / 'manifest.json').read_text())
    assert not metadata['pilot'], 'Arrival replay requires the complete operator profile.'
    metadata['base'] = str(args.restored / 'base')
    metadata['portable_source_manifest'] = str(source / 'manifest.json')
    save_json(costs / 'manifest.json', metadata)
    store = args.out / 'store'
    store.mkdir()
    runner.index_costs(costs, store)
    with (source / 'configs.json').open() as f:
        configs = json.load(f)
    # Exact jobs from the published figure: Tessera async; other arrays serial.
    from concurrent.futures import ProcessPoolExecutor
    output = args.out / 'arrival_replay'
    output.mkdir()
    jobs = [(key.rsplit('@',1)[0], int(key.rsplit('@',1)[1]),
             'region_async' if key.startswith('Tessera') else 'serial') for key in configs]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=runner.initialize_run,
            initargs=(str(costs), str(store), str(args.restored / 'inputs/arrivals'), str(output))) as pool:
        stems = list(pool.map(runner.run_job, jobs))
    totals = []
    for stem in stems:
        totals.extend(json.loads((output / (stem+'.json')).read_text())['totals'])
    keys = list(dict.fromkeys(k for r in totals for k in r))
    save_csv(output / 'totals.csv', [{k:r.get(k,0) for k in keys} for r in totals])
    save_json(output / 'verification.json', dict(status='PASS', records=len(totals),
        artifacts={'totals.csv':sha(output / 'totals.csv')}, jobs=jobs))
    print('PASS replay records:', len(totals))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('verify')
    r = sub.add_parser('restore')
    r.add_argument('--out', type=Path, required=True)
    for name in ('profile', 'replay'):
        r = sub.add_parser(name)
        r.add_argument('--restored', type=Path, required=True)
        r.add_argument('--policy', choices=('native_tail','joint_edp'), required=True)
        r.add_argument('--out', type=Path, required=True)
        r.add_argument('--workers', type=int, default=16)
        r.add_argument('--memory-gb', type=float, default=100)
        if name == 'profile':
            r.add_argument('--limit', type=int, default=0, help='Explicit incomplete smoke test only.')
        else:
            r.add_argument('--costs', type=Path, help='New complete profile; omit to replay included costs.')
    args = p.parse_args()
    for name in ('out', 'restored', 'costs'):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.command == 'verify': verify()
    elif args.command == 'restore': restore(args.out)
    elif args.command == 'profile': profile(args)
    else: replay(args)


if __name__ == '__main__':
    main()
