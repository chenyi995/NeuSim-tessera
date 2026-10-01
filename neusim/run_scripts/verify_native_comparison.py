"""Final source/output immutability and test-log audit for the comparison."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from neusim.run_scripts.compare_tessera_native import ROOT, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = a.out.resolve()
    if out.exists() or not out.is_relative_to(ROOT):
        p.error("output must be a new file inside NeuSim")
    replay = json.loads((a.comparison / "fission_replay/reproduction_manifest.json").read_text())
    native = json.loads((a.comparison / "native_full_with_forceSA/manifest.json").read_text())
    checks = 0
    for name, expected in replay["source_sha256"].items():
        assert sha(a.archive / "snapshot" / name) == expected, name
        checks += 1
    for name, expected in native["source_sha256"].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        assert sha(path) == expected, name
        checks += 1
    for folder, manifest in (("fission_replay", replay), ("native_full_with_forceSA", native)):
        assert manifest["status"] == "PASS"
        for name, expected in manifest["output_sha256"].items():
            assert sha(a.comparison / folder / name) == expected, name
            checks += 1
    log = a.comparison / "backend_tests_with_threshold.log"
    tests = re.search(r"Ran (\d+) tests", log.read_text())
    assert tests and "\nOK\n" in log.read_text()
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    result = dict(status="PASS", source_and_output_hash_checks=checks,
                  backend_tests=int(tests[1]), test_log_sha256=sha(log),
                  archive_sources_unchanged=True, native_sources_match_run_snapshot=True,
                  replay_checks=replay["checks"], unique_shapes=native["unique_shapes"])
    out.write_text(json.dumps(result, indent=2) + "\n")
    assert json.loads(out.read_text()) == result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
