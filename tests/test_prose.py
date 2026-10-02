"""Tests for prose cases (detective_engine/prose.py) and their traces.

Checks that every prose case:
  - states each fact's names, times and places in its paragraph text;
  - has one answer, confirmed by an independent minute-by-minute solve for
    the timeline family;
  - scores 1.0 for the verified answer;
and that the two phrasing banks never share a template, that the trace
writer's reading pass and conclusion agree with the answer key, and that an
alibi extension paragraph always follows the record it corrects.

Run:  python3 tests/test_prose.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from detective_engine import prose  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402
from detective_engine.evaluate import agent_solver  # noqa: E402
import make_sft_data  # noqa: E402

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


def timeline_culprits(case: dict) -> set[str]:
    """Independent solve: every minute the evidence allows, every suspect."""
    facts = case["generator"]["facts"]

    def possible(t: int) -> bool:
        for f in facts:
            d = f["data"]
            if f["kind"] == "last_alive" and t < d["time"]:
                return False
            if f["kind"] == "discovery" and t > d["time"]:
                return False
            if f["kind"] == "body_temp" and abs(t - (d["discovery"] - (37.0 - d["temp"]) * 60)) > 60:
                return False
        return True

    minutes = [t for t in range(24 * 60) if possible(t)]
    kinds = {f["kind"] for f in facts}
    holders = next((set(f["data"]["names"]) for f in facts if f["kind"] == "keyholders"), None)
    labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in case["scenarios"]]
    out = set()
    for s in labels:
        if s == "accident":
            if "homicide" not in kinds:
                out.add(s)
            continue
        if "no_forced_entry" in kinds and holders is not None and s not in holders:
            continue
        alibis = [f["data"] for f in facts if f["kind"] == "alibi" and f["data"]["name"] == s]
        if any(not any(a["start"] <= t <= a["end"] for a in alibis) for t in minutes):
            out.add(s)
    return out


print("Prose cases")
cases = []
for fam, levels in (("timeline", (1, 2, 3)), ("liar", (1, 2, 3)), ("composite", (2, 3))):
    banks = ([("train", "A"), ("test", "B"), ("wide", "A")] + ([("wide2", "A")] if prose.LLM_BANK.exists() else [])
             + ([("wide3", "A")] if prose.LLM_BANK_V2.exists() else []))
    for bank, pool in banks:
        cs = prose.generate(25, 301, fam, levels, pool, bank, f"T{fam[0]}{bank[0]}")
        cases += cs
        check(f"{fam}/{bank}: 25 cases build", len(cs) == 25)
        multi = sum(any(sum(1 for f in c["generator"]["facts"] if f["para"] == i) > 1
                        for i in range(1, len(c["evidence"]) + 1)) for c in cs)
        check(f"{fam}/{bank}: most cases have a multi-fact paragraph", multi >= 15, f"{multi}/25")

bad_text = bad_score = bad_indep = 0
for c in cases:
    for f in c["generator"]["facts"]:
        text = c["evidence"][f["para"] - 1]
        d = f["data"]
        for k in ("name", "witness", "suspect", "speaker", "place"):
            if isinstance(d.get(k), str) and d[k] not in text:
                bad_text += 1
    case = CaseDefinition.from_dict(c)
    r = score_answer(case, agent_solver(case, build_prompt(case), c))
    bad_score += r.reward < 0.999
    if c["generator"]["family"] == "timeline":
        truth = c["answer_key"]["true_scenario"]
        labels = ["accident" if s.endswith("was an accident.") else s.split(",")[0] for s in c["scenarios"]]
        bad_indep += timeline_culprits(c) != {labels[int(truth[1:]) - 1]}
check("every fact's names and places appear in its paragraph", bad_text == 0, str(bad_text))
check("the verified answer scores 1.0 on every case", bad_score == 0, str(bad_score))
check("independent minute-by-minute solve agrees on every timeline case", bad_indep == 0, str(bad_indep))

def _extension_ok(c: dict) -> bool:
    """An alibi that extends another (same person and start, later end) sits in
    the paragraph right after the one it extends."""
    al = [f for f in c["generator"]["facts"] if f["kind"] == "alibi"]
    for f in al:
        for g in al:
            if (g is not f and g["data"]["name"] == f["data"]["name"] and g["data"]["start"] == f["data"]["start"]
                    and g["data"]["end"] > f["data"]["end"] and g["para"] != f["para"] + 1):
                return False
    return True


ext = [c for c in cases if c["generator"]["family"] == "timeline"]
check("an alibi extension always follows the record it corrects", all(_extension_ok(c) for c in ext))

print("\nPhrasing banks")
overlap = [k for k, v in prose.BANK.items() if set(v["train"]) & set(v["test"])]
check("train and test banks share no template", not overlap, str(overlap))
train_text = " ".join(e for c in cases if c["generator"]["bank"] == "train" for e in c["evidence"])
leaks = []
for k, v in prose.BANK.items():
    for tpl in v["test"]:
        chunk = max(re.split(r"\{[a-z]+\}", tpl), key=len).strip()
        if len(chunk) >= 15 and chunk in train_text:
            leaks.append(chunk)
check("no test-bank phrase appears in train-bank cases", not leaks, str(leaks[:3]))


def _grams(words: list[str], n: int = 4) -> set[tuple]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _runs(tpl: str) -> list[list[str]]:
    return [re.findall(r"[a-z0-9°'\-]+", chunk.lower()) for chunk in re.split(r"\{[a-z_]+\}", tpl)]


test_g = {g for v in prose.BANK.values() for tpl in v["test"] for w in _runs(tpl) for g in _grams(w)}
train_g = {g for v in prose.BANK.values() for tpl in v["train"] for w in _runs(tpl) for g in _grams(w)}
banned = test_g - train_g
wide_cases = []
for fam, levels in (("timeline", (2, 3)), ("liar", (2, 3)), ("composite", (2, 3))):
    wide_cases += prose.generate(60, 401, fam, levels, "A", "wide", "W")
if prose.LLM_BANK.exists():
    wide_cases += [c for c in cases if c["generator"]["bank"] in ("wide2", "wide3")]
hits = {g for c in wide_cases for e in c["evidence"]
        for g in _grams(re.findall(r"[a-z0-9°'\-]+", e.lower())) & banned}
check("no four-word run of a test template appears in wide-bank text", not hits, str(sorted(hits)[:3]))


def _shape(e: str) -> str:
    e = re.sub(r"\d\d:\d\d", "T", e)
    return re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "N", e)


wide_shapes = {_shape(e) for c in wide_cases for e in c["evidence"]}
train_shapes = {_shape(e) for c in cases if c["generator"]["bank"] == "train" for e in c["evidence"]}
check("the wide bank is far more varied than the train bank", len(wide_shapes) > 2 * len(train_shapes),
      f"{len(wide_shapes)} vs {len(train_shapes)}")

print("\nTraces")
make_sft_data.COMPACT = make_sft_data.TRACK = make_sft_data.VERBOSE_COMPARE = True
make_sft_data.INDEX_RECORDS = make_sft_data.READ = True
bad_answer = bad_concl = bad_read = 0
for c in cases:
    tr = make_sft_data.write_trace(c)
    ans = json.loads(tr.split("```json")[1].split("```")[0])
    k = c["answer_key"]
    bad_answer += ans["most_likely"] != k["true_scenario"] or ans["key_evidence"] != k["key_evidence"]
    fam = c["generator"]["family"]
    names = [s[: -len(" is lying.")] if fam == "liar" else s.split(",")[0] for s in c["scenarios"]]
    truth = names[int(k["true_scenario"][1:]) - 1]
    if fam == "liar":
        m = re.search(r"Only (.+?)'s statement is contradicted", tr)
        bad_concl += not m or m.group(1) != truth
    else:
        m = re.findall(r"Still possible: (.+)\.", tr)
        bad_concl += not m or m[-1] != f"{k['true_scenario']} {truth}"
    read = tr.split("\n\n")[0]
    cited = set(re.findall(r"^(E\d+):", read, re.M)) | set(re.findall(r"E\d+", " ".join(
        l for l in read.split("\n") if l.startswith(("Motives only", "No fact")))))
    bad_read += cited != {f"E{i}" for i in range(1, len(c["evidence"]) + 1)}
check("trace answer block equals the answer key", bad_answer == 0, str(bad_answer))
check("trace conclusion names the true scenario", bad_concl == 0, str(bad_concl))
check("reading pass accounts for every paragraph exactly", bad_read == 0, str(bad_read))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
