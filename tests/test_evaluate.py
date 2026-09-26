"""Tests for the evaluation harness (no network, no model).

Run:  python3 tests/test_evaluate.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detective_engine import evaluate as ev  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.generator import generate  # noqa: E402

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}" + (f" — {detail}" if detail else ""))


raws = generate(60, seed=31)
cases = [CaseDefinition.from_dict(r) for r in raws]


def run_agent(agent) -> list:
    return [score_answer(c, agent(c, build_prompt(c), r)) for c, r in zip(cases, raws)]


print("\n=== EV-01: Baseline agents ===")
solver = run_agent(ev.agent_solver)
check("solver is perfect on generated cases", all(r.passed and r.reward > 0.999 for r in solver))
uniform = run_agent(ev.agent_uniform)
check("uniform never passes", not any(r.passed for r in uniform))
check("uniform mean reward is low", sum(r.reward for r in uniform) / len(uniform) < 0.15)
rand = run_agent(ev.make_agent_random(0))
check("random never passes", not any(r.passed for r in rand))
check("all baselines emit valid format", all(r.format_ok for r in solver + uniform + rand))

print("\n=== EV-02: Wilson interval ===")
lo, hi = ev.wilson(0, 10)
check("0/10 lower bound is 0", lo == 0.0 and 0.2 < hi < 0.35, f"{lo:.3f}-{hi:.3f}")
lo, hi = ev.wilson(50, 100)
check("50/100 is symmetric around 0.5", abs((lo + hi) / 2 - 0.5) < 1e-9 and 0.39 < lo < 0.41)
check("n=0 is safe", ev.wilson(0, 0) == (0.0, 0.0))

print("\n=== EV-03: Report aggregation ===")
with tempfile.TemporaryDirectory() as tmp:
    d = Path(tmp) / "agentX"
    d.mkdir()
    recs = [{"agent": "agentX", "case_id": f"c{i}", "correct": i % 2 == 0, "reward": 0.5, "passed": False,
             "format_ok": True, "components": {"correctness": float(i % 2 == 0)}} for i in range(10)]
    (d / "test_id.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    table = ev.report(Path(tmp))
    check("report has a row for the run", "| test_id | agentX | 10 | 0.50" in table, table)

print("\n" + "=" * 50)
print(f"  EVALUATE: {passed} passed, {failed} failed")
print("=" * 50)
if failed:
    sys.exit(1)
