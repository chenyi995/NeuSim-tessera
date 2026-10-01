"""Write the delivery documentation and copy exact reference test inputs."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def main():
    sources = []
    def copy(source, target):
        source, target = Path(source), ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(source, target)
        h = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == h
        sources.append(dict(source=str(source),file=str(target.relative_to(ROOT)),sha256=h))
    for name in ('scheduler.py','generator.py'):
        copy(PROJECT / 'planaria.code/scheduler' / name, 'references/planaria/' + name)
    copy(PROJECT / 'planaria.code/LICENSE', 'references/planaria/LICENSE')
    copy(PROJECT / 'FissionSA/multitenant/run_abc_qos.py', 'references/fissionsa/run_abc_qos.py')
    base = ROOT / 'results/tessera/archived/1001/previous_results/20260929_feature_cnn_qos/inputs_v1'
    for name in ('cnn_layers.csv','sources/configs.json'):
        copy(base / name, 'references/cnn_inputs/' + name)
    (ROOT/'references/manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    assert json.loads((ROOT/'references/manifest.json').read_text()) == sources
    (ROOT/'references/README.md').write_text(
        '# Reference code and test inputs\n\n'
        'Exact source copies for the retained scheduler tests and CNN input adapter. '
        'These files are inputs, not published performance results. '
        '`manifest.json` records original locations and SHA256 digests. '
        'The Planaria source retains its own license. FissionSA scenario assignments are parsed '
        'from its source without importing or running that simulator.\n')
    artifact = ROOT/'artifacts/tessera-20261001/README.md'
    energy = artifact.read_text().split('## Current energy coefficients\n\n')[1].split('\n## Results')[0]
    doc = '''# Tessera extension and current experiment

This fork preserves NeuSim's default backend and adds a partition-aware array backend. The published result is the complete per-workload comparison in [all_workloads.pdf](../artifacts/tessera-20261001/per_workload/figures/all_workloads.pdf). Its inputs, operator costs, arrivals, configurations, energy counters and plotting scripts are included in the [artifact](../artifacts/tessera-20261001/README.md). Earlier experiment outputs have been archived locally and are not published as current results.

## Implementation

| Files | Responsibility |
|---|---|
| `neusim/npusim/backend/tessera_partitioned.py` | Finite SRAM, regional geometry, padding, partial sums, array timing and within-round packing |
| `tessera_baselines.py` | Legal baseline geometries and array/traffic counters |
| `tessera_joint.py` | Joint SRAM-tile and array mapping using native E2E EDP |
| `tessera_native_tail.py` | Native factor tiling, whole-array bulk work and fracturable residual packing |
| `power_model.py`, `ChipConfig.py` | Native component energy and explicit calibrated array coefficients |
| `neusim/run_scripts/replay_tessera_arrivals.py`, `tessera_request_dispatch.py` | Recorded arrivals, dependencies, physical regions and shared component service |
| `tools/tessera_artifact.py` | Self-contained input restoration, operator profiling and arrival replay |

Backend filenames without directories in the table are under `neusim/npusim/backend/`. Historical timing-only and custom energy adapters remain in source for provenance; the published configuration selects `tessera_partitioned`. Archived code is not imported or collected as tests.

## Energy units and scope

ENERGY_TABLE

The compute coefficient is the non-SRAM Joules plateau multiplied by the measured clock period and divided by arithmetic operations per cycle. The raw timelines and extracted source parameters are in `artifacts/tessera-20261001/calibration/`. Full precision remains in `configs/chips/tessera_joules_energy.json`; display rounding does not change the simulation.

SRAM dynamic energy is the native dynamic-power coefficient multiplied by transferred bytes divided by native reference bandwidth. The ideal per-PE SRAM service assumption changes service time, not this per-byte cost. Input fills, array operand reads, padded reads, partial writes and reduction reads/writes all contribute. HBM reloads follow finite-capacity tiling. HBM power remains NeuSim's transfer/controller/PHY model, not a calibrated HBM4 DRAM-core/refresh model. Static energy and regulator losses are separate native terms. The Joules array coefficient includes total non-SRAM plateau power, so retaining native static power makes this a rough activity proxy rather than a clean dynamic-only calibration.

## Mapping and E2E boundary

Normal tiling calls the native factor search with the revision's double-buffer and FP32 partial-output capacity. Connected designs execute complete weight blocks on the full array and pack residual work at the supported grain. Joint EDP tiling evaluates capacity-feasible divisor/dyadic SRAM tiles and legal array mappings against native component delay and total energy. This is a finite per-GEMM search, not a global scheduling optimum. Both policies retain native FlashAttention outer tiling.

The native tiler ranks traffic using a candidate-invariant compute estimate. A very short reduction tile can therefore look inexpensive during selection but incur repeated padded array work during execution. The observed large normal/EDP gap is retained and documented in the artifact's diagnostic, rather than represented as an architectural gain.

Tessera enables within-round packing and asynchronous admission of independent adjacent requests. Arrival idle, batch boundaries, autoregressive dependencies, pinned partials and shared HBM/vector service remain. Other arrays use serial request replay. Workloads without independent request metadata retain the complete sourced operator invocation sum. These are analytical E2E workload estimates, not RTL cycle traces or host scheduler measurements.

## Reproduction

Install the package following the upstream README. From the repository root:

```bash
python tools/tessera_artifact.py verify
python tools/tessera_artifact.py restore --out results/tessera/restored
python tools/tessera_artifact.py profile --restored results/tessera/restored \\
  --policy joint_edp --workers 16 --memory-gb 100 --out results/tessera/new-edp-costs
python tools/tessera_artifact.py replay --restored results/tessera/restored \\
  --policy joint_edp --costs results/tessera/new-edp-costs --workers 16 --memory-gb 100 \\
  --out results/tessera/new-edp-replay
```

Use `native_tail` for the other policy. Omit `--costs` to replay the bundled complete operator profiles. Profiling and replay require fresh output paths. `--limit` is an explicit incomplete smoke test, never a complete workload result. The resource arguments control workers and address-space limits; numerical libraries use one thread. Source-machine absolute paths in frozen manifests are provenance, while the portable adapter uses restored input paths and saved physical configurations.

Run active tests with `python -m pytest`; where pytest-cov is not installed, use `python -m pytest -o addopts=''`. Historical performance sweeps are not automatically launched by tests. Reference scheduler code and sourced CNN test inputs are included under `references/`; the latest PPA workflow does not need sibling repositories.
'''.replace('ENERGY_TABLE',energy.strip())
    (ROOT/'docs/tessera_reproduction.md').write_text(doc)
    readme=ROOT/'README.md'
    backup=ROOT/'docs/archived/1001/README.upstream.md'
    backup.parent.mkdir(parents=True,exist_ok=True)
    copy(readme, str(backup.relative_to(ROOT)))
    intro='''# NeuSim with Tessera

This fork adds Tessera mapping, equal-PE baselines, finite-SRAM energy accounting and request replay to NeuSim. The default upstream backend remains available.

- [Tessera changes, energy coefficients and reproduction](docs/tessera_reproduction.md)
- [Current results: one two-panel figure per workload](artifacts/tessera-20261001/per_workload/figures/all_workloads.pdf)
- [Complete current artifact and workload sources](artifacts/tessera-20261001/README.md)

The current artifact corresponds only to the requested `all_workloads.pdf` run. Earlier performance results are archived locally. Array coefficients are in pJ per multiply or add; SRAM coefficients are in pJ per transferred bit, not per arithmetic op. The detailed accounting and limitations are documented above.

---

'''
    readme.write_text(intro+backup.read_text())
    assert readme.read_text().endswith(backup.read_text())
    print('PASS documentation and exact reference copies')


if __name__=='__main__':
    main()
