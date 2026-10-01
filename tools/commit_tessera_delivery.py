"""Create the explicitly requested topic commits and the final push record."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True)


def commit(paths,message):
    if not paths:
        return
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    diff=git('diff','--cached','--no-ext-diff')
    assert not re.search('[\u4e00-\u9fff]',diff),'Non-English staged source'
    subprocess.run(['git','commit','-q','-m',message],cwd=ROOT,check=True)
    print(git('log','-1','--format=%h %s').strip(),flush=True)


def main():
    assert git('remote','get-url','origin').strip()=='git@github.com:chenyi995/NeuSim-tessera.git'
    start=git('rev-parse','HEAD').strip()
    assert start==git('rev-parse','origin/main').strip()
    paths=set(git('diff','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    topics=[
        ('Add partition-aware Tessera and baseline timing and energy models',lambda p:p.startswith(('neusim/configs/','configs/chips/tessera','neusim/npusim/')) and '/tests/' not in p),
        ('Preserve reference allocation functions and sourced test inputs',lambda p:p.startswith('references/')),
        ('Add array and scheduler checks with archived tests excluded',lambda p:'/tests/' in p or p=='pyproject.toml'),
        ('Add workload profiling mapping and request replay tools',lambda p:p.startswith('neusim/run_scripts/')),
        ('Add lossless artifact export restoration and reproduction commands',lambda p:p.startswith('tools/')),
        ('Publish the latest two-policy workload inputs and complete result data',lambda p:p.startswith('artifacts/')),
        ('Document Tessera reproduction energy units and the current result set',lambda p:p in ('README.md','.gitignore') or p.startswith('docs/') and p.endswith('.md')),
    ]
    committed=[]
    for message,match in topics:
        selected=sorted(p for p in paths if match(p))
        commit(selected,message)
        committed.append(dict(sha=git('rev-parse','HEAD').strip(),subject=message,files=len(selected)))
        paths.difference_update(selected)
    assert paths=={'docs/archived/f1b797032d9e6a6eae5f3822123f230d.png'},paths
    # The user's discussion image is already local history, not simulator source.
    exclude=ROOT/'.git/info/exclude'
    with exclude.open('a') as f:f.write('\n/docs/archived/f1b797032d9e6a6eae5f3822123f230d.png\n')
    checks=json.loads((ROOT/'results/tessera/20261001_delivery_checks/readback_verification.json').read_text())
    archive=json.loads((ROOT/'results/tessera/archived/1001/previous_results/archive_manifest.json').read_text())
    manifest=json.loads((ROOT/'artifacts/tessera-20261001/manifest.json').read_text())
    log=Path('/tmp/neusim-tessera-pytest-v4-20261001.log').read_text()
    result=re.search(r'\d+ passed, \d+ skipped in [\d.]+s',log)[0]
    account=subprocess.check_output(['whoami'],text=True).strip()
    path=ROOT/'docs/session/daily'/f'{account}-20261001-neusim-tessera-publication.md'
    path.parent.mkdir(parents=True,exist_ok=True)
    lines=['# NeuSim Tessera publication', '',
        f'{account} requested publishing the NeuSim changes, workload inputs and results to `chenyi995/NeuSim-tessera`, split into topic commits. '
        'The subsequent ruling limits published performance results to the run behind the supplied all_workloads.pdf; earlier runs are archived locally.', '',
        f'Base: `main` at `{start}`, also the destination main before delivery. '
        'Origin now points to the requested fork; the previous origin remains as upstream. '
        'Rebase onto origin/main has no divergent upstream changes to replay.', '',
        '## Commits', '', '| Commit | Change | Files |','|---|---|---:|']
    lines += [f"| `{r['sha']}` | {r['subject']} | {r['files']} |" for r in committed]
    lines += ['', '## Validation and data scope', '',
        f'- Active tests: `{result}` using `python -m pytest -o addopts=\'\' -q`.',
        '- The first test invocation lacked pytest-cov. The next collected archived tests; collection now excludes them. '
        'A source-input path failure after archiving was repaired using hash-checked reference inputs, without weakening the test.',
        f"- Artifact: {len(manifest['files'])} source files; {manifest['stored_bytes']} stored bytes. Every file was read/decompressed and SHA256 compared to its source.",
        f"- Portable smoke checks: {json.dumps(checks['portable_smoke_records'])}; every field matched the original complete operator profiles.",
        f"- Plot coordinates independently recomputed for {checks['plotted_records']} records; published PDF SHA256 `{checks['published_pdf_sha256']}` equals the user-specified PDF.",
        f"- Archived {len(archive['moved'])} preceding top-level paths under `results/tessera/archived/1001/previous_results/`; all pre-existing older archives remain.",
        '- Current result sources remain under `results/tessera/20261001_ppa_two_tilings_v1` and `results/tessera/20261001_ppa_workload_pairs_v1`. '
        'The published artifact includes complete profiles and reproducible inputs, but omits rebuildable SQLite caches.',
        '- Validation output is local under `results/tessera/20261001_delivery_checks`; original command logs remain under `/tmp/neusim-tessera-*20261001.log`.',
        '- Array coefficients and SRAM units are stated in the artifact and reproduction README. No N28 SRAM candidate was substituted; no performance data was changed.',
        '- The remaining user-requested task is the isolated single-array GEMM/GEMV tiling diagnostic, to be performed after this push. '
        'The native tiler may select a short reduction tile that is underpriced by its invariant compute estimate. '
        'The published results retain this known modeling/selection interaction.', '']
    path.write_text('\n'.join(lines))
    assert path.read_text()=='\n'.join(lines)
    commit([str(path.relative_to(ROOT))],'Record the NeuSim Tessera artifact publication')
    print('READY',git('rev-parse','HEAD').strip(),flush=True)


if __name__=='__main__':main()
