"""Read-only replay of the final revision archive, bounded by the user budget."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("workers must fit the authorized 16-CPU budget")
    root = Path(__file__).resolve().parents[2]
    out = args.out.resolve()
    if not out.is_relative_to(root) or out.exists():
        parser.error("output must be a new directory inside NeuSim")
    out.mkdir(parents=True)
    # Codex: decision start — bound the entire process tree, not each worker to 100 GB.
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.workers])
    cap = 100_000_000_000 // (args.workers + 1)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    script = args.archive.resolve() / "tools/reproduce.py"
    command = [sys.executable, "-B", str(script), "--mode", "replay", "--workers",
               str(args.workers), "--out", str(out / "fission_replay")]
    record = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat(),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "per_process_address_space": cap,
              "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}
    (out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    with (out / "fission_replay.log").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    record.update(returncode=result.returncode, ended_utc=datetime.now(timezone.utc).isoformat())
    (out / "launch.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    # Codex: decision end
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
