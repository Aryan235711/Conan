"""Run every test script in this folder and report a combined result.

The test files are standalone scripts (they execute at import time), so
``python -m unittest discover`` finds zero tests.  This runner executes each
``test_*.py`` in its own process with the project root on PYTHONPATH and
fails if any script exits non-zero.

Run:  python3 tests/run_all.py          (add -v to see each script's output)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
ROOT = TESTS_DIR.parent


def main() -> int:
    verbose = "-v" in sys.argv[1:]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    failures: list[str] = []
    for script in sorted(TESTS_DIR.glob("test_*.py")):
        start = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        elapsed = time.perf_counter() - start
        ok = proc.returncode == 0
        print(f"{'PASS' if ok else 'FAIL'}  {script.name:<28} {elapsed:6.2f}s")
        if verbose or not ok:
            print(proc.stdout[-4000:])
            if proc.stderr:
                print(proc.stderr[-4000:], file=sys.stderr)
        if not ok:
            failures.append(script.name)

    print()
    if failures:
        print(f"{len(failures)} test script(s) failed: {', '.join(failures)}")
        return 1
    print("All test scripts passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
