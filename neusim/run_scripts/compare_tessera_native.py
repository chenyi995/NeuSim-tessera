"""Paired revision workload replay through native NeuSim, with isolated controls.

The only architecture injection is frozen SA cycles. All native VU, memory,
overlap and power functions are executed. This is an operator-level comparison,
not a full application trace or a physical implementation of Tessera routing.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import csv
from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("out/tessera-revision/fissionsa_selected8_hbm4/sram_12mb")
REPLAY = Path("out/tessera-revision/planaria_hbm_replay")
ARCHS = ("fission32_rect", "fission32_square", "independent_ws32")
FLOW = dict(zip(ARCHS, ("native_full", "native_square", "native_independent")))
SHORT = dict(zip(ARCHS, ("full", "square", "independent32")))
COMPONENTS = ("sa", "vu", "sram", "hbm", "ici", "other")
ENERGY_FIELDS = [f"{kind}_energy_{component}_J" for kind in ("dynamic", "static") for component in COMPONENTS]
METRICS = ["time_ns", "energy_J", "hbm_bytes", "sa_ns", "vu_ns", "sram_ns", "hbm_ns",
           "pre_time_ns", "pre_sa_ns", "pre_vu_ns", "pre_sram_ns", "pre_hbm_ns",
           "sa_cycles", "fallback", "sa_bound", "vu_bound", "sram_bound", "hbm_bound",
           "static_J", "dynamic_J", "fission_existing_J", "fission_hbm_J"] + ENERGY_FIELDS
_SK = None
_CFG = None
_LOCAL = None


def rows(path):
    with Path(path).open(newline="") as handle:
        yield from csv.DictReader(handle)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, records):
    records = iter(records)
    first = next(records)
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(records)


class Sink:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


def initialize(snapshot):
    global _SK, _CFG, _LOCAL
    sys.dont_write_bytecode = True
    from neusim.configs.chips.ChipConfig import ChipConfig
    from neusim.npusim.backend import npusim_lib
    # Codex: decision start — memoize only this pure native tiling function;
    # its inputs/return value are unchanged, and no simulator source is patched.
    npusim_lib.find_best_tile_shape_for_matmul = lru_cache(maxsize=4096)(
        npusim_lib.find_best_tile_shape_for_matmul)
    _CFG = ChipConfig(num_sa=1, sa_dim=128, freq_GHz=1, vmem_size_MB=12,
                      hbm_bw_GBps=2_800_000_000_000 / 1024**3,
                      array_backend="tessera_native_replay")
    from neusim.npusim.backend.tessera import TesseraConfig
    _LOCAL = TesseraConfig(**json.loads((ROOT / "configs/chips/tessera_revision.json").read_text())["parameters"])
    path = Path(snapshot) / "Tessera-revision/experiments/planaria_hbm_ablation/run_skew_ablation.py"
    spec = importlib.util.spec_from_file_location("frozen_skew_for_native_comparison", path)
    _SK = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_SK)
    # Codex: decision end


def native(m, n, k, cycles, *, force_sa=False, base_frequency=False):
    from neusim.npusim.frontend.llm_ops_lib import create_einsum_op
    from neusim.npusim.frontend.op_analysis_lib import fill_operators_execution_info
    from neusim.npusim.frontend import power_analysis_lib
    from neusim.npusim.frontend.Operator import DVFSConfig
    config = _CFG.model_copy(update={"use_vu_for_small_matmul": not force_sa,
                                     "array_backend": "default" if cycles is None else "tessera_native_replay"})
    op = create_einsum_op([m, k], [k, n], "MK;KN->MN")
    if cycles is not None:
        op.tessera_spec = dict(kind="native_sa_replay", B=1, M=m, N=n, K=k,
                              pe_count=16384, sa_cycles=cycles)
    with redirect_stdout(Sink()):
        op = fill_operators_execution_info([op], config, analyze_energy=False)[0]
        pre = op.stats.model_copy(deep=True)
        if base_frequency:
            # Explicit public API control; retain all native power calculations.
            # The primary run leaves native NONE-policy 1.7 GHz untouched.
            power_analysis_lib.configure_dvfs_for_op(op, config, DVFSConfig())
            for component in ("sa", "vu", "sram", "hbm", "ici"):
                getattr(op, "dvfs_" + component).frequency_GHz = config.freq_GHz
            power_analysis_lib.analyze_operator_energy(op, config, set_dvfs_config_for_op=False)
        else:
            power_analysis_lib.analyze_operator_energy(op, config)
    s = op.stats
    assert s.flop_count == 2 * m * n * k
    assert s.execution_time_ns == max(s.sa_time_ns, s.vu_time_ns, s.memory_time_ns, s.vmem_time_ns)
    assert math.isclose(s.total_energy_J, sum(getattr(s, key) for key in ENERGY_FIELDS), rel_tol=1e-12)
    active = [(s.sa_time_ns, "sa"), (s.vu_time_ns, "vu"),
              (s.memory_time_ns, "hbm"), (s.vmem_time_ns, "sram")]
    bound = max(active, key=lambda pair: pair[0])[1]
    result = dict(time_ns=s.execution_time_ns, energy_J=s.total_energy_J,
                  hbm_bytes=s.memory_traffic_bytes, sa_ns=s.sa_time_ns, vu_ns=s.vu_time_ns,
                  sram_ns=s.vmem_time_ns, hbm_ns=s.memory_time_ns,
                  pre_time_ns=pre.execution_time_ns, pre_sa_ns=pre.sa_time_ns,
                  pre_vu_ns=pre.vu_time_ns, pre_sram_ns=pre.vmem_time_ns,
                  pre_hbm_ns=pre.memory_time_ns, sa_cycles=cycles or 0,
                  fallback=int(pre.sa_time_ns == 0), static_J=s.static_energy_J,
                  dynamic_J=s.dynamic_energy_J, fission_existing_J=0, fission_hbm_J=0)
    result.update({component + "_bound": int(bound == component) for component in ("sa", "vu", "sram", "hbm")})
    result.update({key: getattr(s, key) for key in ENERGY_FIELDS})
    return result


def tail(candidate):
    m, n, k, kt, nt, cm, cn = (int(candidate[key]) for key in
                               ("M", "N", "K", "kt", "nt", "chunk_rows", "chunk_cols"))
    square = candidate["physical_square"] == "True"
    return sum(mc * nc * _SK.final_tail(nn, k, kt, nt, square)[0]
               for _, mc in _SK.parts(m, cm) for nn, nc in _SK.parts(n, cn))


def mapping(row):
    return tuple(row[k] for k in ("kt", "nt", "physical_square"))


def evaluate(job):
    sid, candidates, frozen, archived_skew = job
    shape = tuple(int(candidates[0][key]) for key in ("M", "N", "K"))
    m, n, k = shape
    assert len(candidates) == 28
    memo = {}

    def run(cycles, force_sa=False, base_frequency=False):
        key = cycles, force_sa, base_frequency
        if key not in memo:
            memo[key] = native(m, n, k, cycles, force_sa=force_sa, base_frequency=base_frequency)
        return memo[key]

    selected, scores = [], []

    def keep(label, stats, candidate=None, extra=0):
        row = dict(shape_id=sid, M=m, N=n, K=k, variant=label,
                   kt=candidate["kt"] if candidate else "",
                   nt=candidate["nt"] if candidate else "",
                   physical_square=candidate["physical_square"] if candidate else "",
                   skew_tail_cycles=extra)
        row.update(stats)
        selected.append(row)

    keep("native_stock", run(None))
    chosen, forced_chosen = {}, {}
    for arch in ARCHS:
        subset = [r for r in candidates if r["architecture"] == arch]
        # Native EDP is re-evaluated for exactly the same candidate set.
        ranked = []
        for candidate in subset:
            cost = run(int(candidate["compute_cycles"]))
            score = cost["energy_J"] * cost["time_ns"]
            ranked.append((score, candidate, cost))
            scores.append(dict(shape_id=sid, architecture=arch, kt=candidate["kt"], nt=candidate["nt"],
                               physical_square=candidate["physical_square"], sa_cycles=candidate["compute_cycles"],
                               time_ns=cost["time_ns"], energy_J=cost["energy_J"], edp_J_ns=score))
        _, best, cost = min(ranked, key=lambda x: x[0])
        chosen[arch] = best
        keep("native_" + SHORT[arch], cost, best)
        forced_costs = [(candidate, run(int(candidate["compute_cycles"]), force_sa=True)) for candidate in subset]
        forced_best, forced_cost = min(forced_costs, key=lambda pair: pair[1]["energy_J"] * pair[1]["time_ns"])
        forced_chosen[arch] = forced_best
        keep("native_forceSA_" + SHORT[arch], forced_cost, forced_best)
        fixed = frozen[(arch, FLOW[arch], "overlap")]
        assert any(mapping(c) == mapping(fixed) and c["compute_cycles"] == fixed["compute_cycles"] for c in subset)
        keep("native_frozen_" + SHORT[arch], run(int(fixed["compute_cycles"])), fixed)
        keep("native_1GHz_" + SHORT[arch], run(int(best["compute_cycles"]), base_frequency=True), best)
        for prefix, flow, timing in (("fission_", FLOW[arch], "overlap"),
                                     ("fission_common_", "native_full", "overlap"),
                                     ("fission_serial_", FLOW[arch], "serial")):
            f = frozen[(arch, flow, timing)]
            stat = dict.fromkeys(METRICS, 0)
            stat.update(time_ns=int(f["total_cycles"]), energy_J=float(f["energy_pj"]) * 1e-12,
                        hbm_bytes=int(f["hbm_read_bytes"]) + int(f["hbm_write_bytes"]),
                        sa_cycles=int(f["compute_cycles"]), fission_existing_J=float(f["energy_existing_pj"]) * 1e-12,
                        fission_hbm_J=float(f["energy_hbm_pj"]) * 1e-12)
            keep(prefix + SHORT[arch], stat, f)
    best = chosen[ARCHS[0]]
    fixed = frozen[(ARCHS[0], FLOW[ARCHS[0]], "overlap")]
    extra, fixed_extra = tail(best), int(archived_skew["skew_tail_cycles"])
    assert tail(fixed) == fixed_extra
    keep("native_skew", run(int(best["compute_cycles"]) + extra), best, extra)
    keep("native_1GHz_skew", run(int(best["compute_cycles"]) + extra, base_frequency=True), best, extra)
    keep("native_frozen_skew", run(int(fixed["compute_cycles"]) + fixed_extra), fixed, fixed_extra)
    forced_best = forced_chosen[ARCHS[0]]
    forced_extra = tail(forced_best)
    keep("native_forceSA_skew", run(int(forced_best["compute_cycles"]) + forced_extra, force_sa=True), forced_best, forced_extra)
    f = next(row for row in selected if row["variant"] == "fission_full")
    stat = {key: f[key] for key in METRICS}
    stat.update(time_ns=int(archived_skew["skew_total_cycles"]), sa_cycles=int(archived_skew["skew_compute_cycles"]))
    keep("fission_skew", stat, fixed, fixed_extra)
    # Codex: decision start — reproduce the previous custom model on the SAME
    # isolated inputs, separately labeled; never substitute its energy into native runs.
    from neusim.npusim.backend.tessera import gemm_cost
    for grain in (8, 32):
        config = replace(_LOCAL, grain=grain)
        chosen_local = None
        for variant, short in (("tessera", "full"), ("square", "square"),
                               ("independent", "independent"), ("skew", "skew")):
            reads, writes = 2 * (m * k + k * n), 2 * m * n
            local = gemm_cost(m, n, k, 1, variant, config,
                              (chosen_local.kt, chosen_local.nt) if variant == "skew" else None,
                              transfer_bytes=reads, write_bytes=writes)
            if variant == "tessera":
                chosen_local = local
            candidate = dict(kt=local.kt, nt=local.nt, physical_square=str(variant == "square"))
            keep(f"native_local{grain}_{short}", run(local.array_cycles), candidate)
            stat = dict.fromkeys(METRICS, 0)
            stat.update(time_ns=local.seconds * 1e9, energy_J=local.energy_J,
                        hbm_bytes=reads + writes + local.hbm_reload_bytes, sa_cycles=local.array_cycles,
                        static_J=local.background_energy_J,
                        dynamic_J=local.energy_J - local.background_energy_J,
                        dynamic_energy_sa_J=local.array_energy_J,
                        dynamic_energy_sram_J=local.sram_energy_J,
                        dynamic_energy_vu_J=local.vector_energy_J,
                        dynamic_energy_hbm_J=local.hbm_energy_J,
                        static_energy_sa_J=local.background_energy_J)
            keep(f"legacy{grain}_{short}", stat, candidate)
    # Codex: decision end
    assert len({r["hbm_bytes"] for r in selected if r["variant"].startswith("native")}) == 1
    return selected, scores


def groups(path):
    for sid, it in itertools.groupby(rows(path), key=lambda row: int(row["shape_id"])):
        yield sid, list(it)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--replayed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="Pilot only; never reported as full workload")
    args = parser.parse_args()
    out, snapshot = args.out.resolve(), args.archive.resolve() / "snapshot"
    if out.exists() or not out.is_relative_to(ROOT) or not 1 <= args.workers <= 16:
        parser.error("use a new output directory inside NeuSim and 1..16 workers")
    out.mkdir(parents=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.workers])
    cap = 100_000_000_000 // (args.workers + 1)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    start = time.monotonic()
    replay_manifest = json.loads((args.replayed / "reproduction_manifest.json").read_text())
    assert replay_manifest["status"] == "PASS" and replay_manifest["mode"] == "replay"
    sources = [snapshot / BASE / "inputs/workloads.csv", snapshot / BASE / "inputs/unique_shapes.csv",
               snapshot / REPLAY / "run/candidate_inputs.csv", args.replayed / "best_results.csv",
               args.replayed / "skew_shape_results.csv"]
    provenance = {str(p): sha(p) for p in sources}
    for name in ("best_results.csv", "skew_shape_results.csv"):
        assert provenance[str(args.replayed / name)] == replay_manifest["output_sha256"][name]
    for path in sources[:3]:
        assert provenance[str(path)] == replay_manifest["source_sha256"][str(path.relative_to(snapshot))]
    source_dir = out / "source"
    source_dir.mkdir()
    for relative in ("run_scripts/compare_tessera_native.py", "npusim/backend/tessera_native.py",
                     "npusim/backend/npusim_lib.py", "npusim/frontend/op_analysis_lib.py",
                     "configs/chips/ChipConfig.py", "npusim/backend/power_model.py",
                     "npusim/backend/dvfs_policy_lib.py", "npusim/frontend/power_analysis_lib.py",
                     "npusim/backend/tessera.py"):
        path = ROOT / "neusim" / relative
        shutil.copyfile(path, source_dir / relative.replace("/", "__"))
        provenance[str(path)] = sha(path)
    provenance[str(ROOT / "configs/chips/tessera_revision.json")] = sha(ROOT / "configs/chips/tessera_revision.json")
    manifest = dict(status="running", workers=args.workers, limit=args.limit, source_sha256=provenance,
                    affinity=sorted(os.sched_getaffinity(0)), per_process_address_space_bytes=cap,
                    invocation=sys.argv, scope="Frozen SA-cycle replay in native NeuSim; weighted isolated GEMMs")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    frozen = defaultdict(dict)
    for row in rows(sources[3]):
        arch, flow, timing = row["architecture"], row["flow_profile"], row["timing"]
        if arch in ARCHS and ((flow == FLOW[arch]) or (flow == "native_full" and timing == "overlap")):
            frozen[int(row["shape_id"])][arch, flow, timing] = row
    skew = {int(r["shape_id"]): r for r in rows(sources[4])}
    initialize(snapshot)
    (out / "chip_config.json").write_text(_CFG.model_dump_json(indent=2) + "\n")
    jobs = ((sid, candidates, frozen[sid], skew[sid]) for sid, candidates in groups(sources[2]))
    if args.limit:
        jobs = itertools.islice(jobs, args.limit)
    count = 0
    with (out / "shape_results.csv").open("w", newline="") as sf, (out / "candidate_scores.csv").open("w", newline="") as cf:
        sw = cw = None
        # Bound queued jobs as well as worker memory.
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
            while block := list(itertools.islice(jobs, 128)):
                for selected, scores in pool.map(evaluate, block, chunksize=4):
                    if sw is None:
                        sw = csv.DictWriter(sf, fieldnames=list(selected[0])); sw.writeheader()
                        cw = csv.DictWriter(cf, fieldnames=list(scores[0])); cw.writeheader()
                    sw.writerows(selected); cw.writerows(scores)
                    count += 1
                sf.flush(); cf.flush()
                print(f"Native replay {count}/{args.limit or len(skew)} shapes; {time.monotonic() - start:.1f} s", flush=True)
    assert count == (args.limit or len(skew))
    manifest.update(status="PASS", unique_shapes=count, elapsed_seconds=time.monotonic()-start,
                    output_sha256={p.name: sha(p) for p in (out / "shape_results.csv", out / "candidate_scores.csv")})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("status", "unique_shapes", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Preserve partial outputs and record failures, including pilot failures.
        if "--out" in sys.argv:
            path = Path(sys.argv[sys.argv.index("--out") + 1]) / "manifest.json"
            if path.exists():
                manifest = json.loads(path.read_text())
                if manifest.get("status") == "running":
                    manifest.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
                    path.write_text(json.dumps(manifest, indent=2) + "\n")
        raise
