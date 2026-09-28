"""Tests for the reliability-suite builder (benchmarks/reliability/build.py).

Uses a small built-in case so it runs in CI, where the private case sources
are not present. If local sources exist, it also checks they all build.

Run:  python3 tests/test_reliability.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks" / "reliability"))

import build  # noqa: E402

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


def S(label: str) -> dict:
    return {"text": f"{label}, the suspect, killed Vic Tim.", "label": label}


BASE = {
    "id": "T-1", "tier": "A", "verification": "solver", "family": "timeline",
    "title": "Test", "summary": "A test case.", "hidden_truth": "Cy Dow did it.",
    "evidence": [
        {"text": "Vic Tim was found at 07:00. At 07:00 the body temperature was 29.0°C; a body cools about 1°C per hour.",
         "facts": [{"kind": "discovery", "data": {"time": "07:00"}},
                   {"kind": "body_temp", "data": {"temp": 29.0, "discovery": "07:00"}}]},
        {"text": "No forced entry. Keys are held only by Ann Lee, Bo Fry and Cy Dow.",
         "facts": [{"kind": "no_forced_entry", "data": {}},
                   {"kind": "keyholders", "data": {"names": ["Ann Lee", "Bo Fry", "Cy Dow"]}}]},
        {"text": "CCTV shows Ann Lee at the pub from 21:30 to 00:30.",
         "facts": [{"kind": "alibi", "data": {"name": "Ann Lee", "start": "21:30", "end": "00:30"}}]},
        {"text": "Bo Fry was at work from 22:00 to 00:10.",
         "facts": [{"kind": "alibi", "data": {"name": "Bo Fry", "start": "22:00", "end": "00:10"}}]},
        {"text": "Cy Dow was at the gym from 21:00 to 22:30.",
         "facts": [{"kind": "alibi", "data": {"name": "Cy Dow", "start": "21:00", "end": "22:30"}}]},
        {"text": "It rained that night.", "herring": True},
    ],
    "scenarios": [S("Ann Lee"), S("Bo Fry"), S("Cy Dow")],
    "intended": "Cy Dow",
}


def expect_error(name: str, src: dict, needle: str) -> None:
    try:
        build.build_solver_case(src)
        check(name, False, "no error raised")
    except build.BuildError as exc:
        check(name, needle in str(exc), str(exc))


print("\n=== RB-01: A valid case builds with a computed key ===")
case = build.build_solver_case(copy.deepcopy(BASE))
k = case["answer_key"]
labels = [s.split(",")[0] for s in case["scenarios"]]
check("true scenario is the intended culprit", labels[int(k["true_scenario"][1:]) - 1] == "Cy Dow")
check("time window computed (22:00 to 00:00)", k["time_window"]["earliest"] == "22:00" and k["time_window"]["latest"] == "00:00", str(k["time_window"]))
# All three suspects hold keys, so the access paragraph (E2) decides nothing and is not key.
check("window and covering alibis are key; non-discriminating access is not",
      set(k["key_evidence"]) == {"E1", "E3", "E4"}, str(k["key_evidence"]))
check("red herring recorded", k["red_herrings"] == ["E6"])

print("\n=== RB-02: Authoring mistakes are caught ===")
bad = copy.deepcopy(BASE)
bad["evidence"][2]["text"] = "CCTV shows Ann Lee at the pub all evening."
expect_error("prose that omits a fact's times is rejected", bad, "does not state")
bad = copy.deepcopy(BASE)
bad["intended"] = "Bo Fry"
expect_error("wrong intended culprit is rejected", bad, "solver finds")
bad = copy.deepcopy(BASE)
bad["evidence"][3]["herring"] = True
expect_error("a 'red herring' that changes the answer is rejected", bad, "changes the answer")

print("\n=== RB-03: Local suite (if present) builds cleanly ===")
if any(build.SRC.glob("*.json")):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        rc = build.main()
    check("every local source file builds", rc == 0)
else:
    print("  (no local sources; skipped)")

print("\n" + "=" * 50)
print(f"  RELIABILITY BUILDER: {passed} passed, {failed} failed")
print("=" * 50)
if failed:
    sys.exit(1)
