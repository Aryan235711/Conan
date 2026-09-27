"""Build supervised warm-up data: step-by-step reasoning traces from the solver.

GRPO needs reward variance inside each group of samples.  A small model
that never emits a valid final answer gets reward 0 everywhere and learns
nothing, so training starts with a short supervised pass on worked
solutions.  Each trace is written from the generator's structured facts in
the same order a careful detective would work:

    1. fix the time window from physical evidence
    2. rule out an accident if the evidence allows
    3. apply access (forced entry, key holders)
    4. check which verified alibis cover the whole window
    5. name the red herrings and why they change nothing
    6. conclude, then emit the final JSON answer

Only the train split is used; test splits never appear in SFT data.

    python3 training/make_sft_data.py --out data/generated/sft_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt  # noqa: E402
from detective_engine.evaluate import agent_solver  # noqa: E402
from detective_engine.generator import BODY_TEMP, COOLING_PER_HOUR, ESTIMATE_TOLERANCE, Fact, death_window, fmt  # noqa: E402

sys.path.insert(0, str(ROOT / "training"))
from coverage_text import explain_coverage  # noqa: E402


def write_liar_trace(raw: dict) -> str:
    """Trace for the witness-consistency family: find the statement a record contradicts."""
    case = CaseDefinition.from_dict(raw)
    facts = raw["generator"]["facts"]
    eid = {i: f"E{i}" for i in range(1, len(facts) + 1)}
    names = [s[: -len(" is lying.")] for s in case.scenarios]
    sid = {n: f"S{i}" for i, n in enumerate(names, 1)}
    records = [(i, tuple(a)) for i, f in enumerate(facts, 1) if f["kind"] == "record" for a in f["data"]["atoms"]]
    lines = ["Step 1 - Records are reliable, so each fixes where one person was at one time."]
    for i, (p, tm, l, _) in records:
        lines.append(f"{eid[i]}: {p} was at {l} at {fmt(tm)}.")
    lines.append("\nStep 2 - Check what each statement implies against the records.")
    liar, clash = None, None
    for i, f in enumerate(facts, 1):
        if f["kind"] != "statement":
            continue
        for p, tm, l, _ in (tuple(a) for a in f["data"]["atoms"]):
            for j, (rp, rt, rl, _) in records:
                if (rp, rt) == (p, tm) and rl != l:
                    liar, clash = f["data"]["speaker"], (i, j, p, tm, l, rl)
    i, j, p, tm, l, rl = clash
    who = "they were" if p == liar else f"{p} was"
    lines.append(f"{eid[i]}: {liar}'s statement means {who} at {l} at {fmt(tm)}. "
                 f"{eid[j]} shows {p} at {rl} at {fmt(tm)}. These cannot both be true, so {liar}'s statement is false.")
    lines.append("Every other statement fits the records and the other statements, so exactly one witness is lying.")
    herr = [i for i, f in enumerate(facts, 1) if f["kind"] in ("demeanour", "salient")]
    lines.append("\nStep 3 - Red herrings.")
    for i in herr:
        lines.append(f"{eid[i]}: describes behaviour or background, not where anyone was, so it proves nothing.")
    answer = agent_solver(case, "", raw)
    lines.append(f"\nConclusion: {liar} ({sid[liar]}) is lying.")
    lines.append("\n```json\n" + json.dumps(answer, indent=2) + "\n```")
    return "\n".join(lines)


def write_composite_trace(raw: dict) -> str:
    """Trace for the combined family: window, lying witness, voided alibi, access, alibis."""
    case = CaseDefinition.from_dict(raw)
    facts = [Fact(f["kind"], f["text"], f["data"], f["role"]) for f in raw["generator"]["facts"]]
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}
    label_of = {}
    for i, s in enumerate(case.scenarios, 1):
        label_of["accident" if s.endswith("was an accident.") else s.split(",")[0].strip()] = f"S{i}"
    lo, hi = death_window(facts)
    by = lambda k: [f for f in facts if f.kind == k]  # noqa: E731
    lines = ["Step 1 - Time of death."]
    for f in by("body_temp"):
        drop = BODY_TEMP - f.data["temp"]; hours = drop / COOLING_PER_HOUR
        est = round(f.data["discovery"] - hours * 60)
        lines.append(f"{eid[id(f)]}: a drop of {drop:.1f}°C is about {hours:.1f} hours, so death was around {fmt(est)}, "
                     f"within an hour: {fmt(est - ESTIMATE_TOLERANCE)} to {fmt(est + ESTIMATE_TOLERANCE)}.")
    for f in by("last_alive"):
        lines.append(f"{eid[id(f)]}: the victim was alive at {fmt(f.data['time'])}.")
    lines.append(f"So the death window is {fmt(lo)} to {fmt(hi)}.")
    for f in by("homicide"):
        lines.append(f"{eid[id(f)]} rules out an accident ({label_of['accident']}).")

    lines.append("\nStep 2 - Find the lying witness by checking each alibi claim against the records.")
    liar = None
    for c in by("witness_alibi"):
        d = c.data
        for r in by("record"):
            rd = r.data
            if rd["name"] in (d["witness"], d["suspect"]) and d["start"] <= rd["time"] <= d["end"] and rd["place"] != d["place"]:
                liar = d["witness"]
                lines.append(f"{eid[id(c)]} says {d['witness']} and {d['suspect']} were at {d['place']} from "
                             f"{fmt(d['start'])} to {fmt(d['end'])}, but {eid[id(r)]} shows {rd['name']} at {rd['place']} "
                             f"at {fmt(rd['time'])}. So {d['witness']} is lying and this alibi does not count.")
    lines.append("Every other alibi claim fits the records.")

    lines.append("\nStep 3 - Access.")
    kh = by("keyholders")[0]
    keyholders = set(kh.data["names"])
    lines.append(f"{eid[id(by('no_forced_entry')[0])]}: no forced entry, so the killer used a key. {eid[id(kh)]} lists the key holders.")
    for name, sid in label_of.items():
        if name != "accident" and name not in keyholders:
            lines.append(f"{name} ({sid}) has no key and is ruled out.")

    lines.append(f"\nStep 4 - Truthful alibis against the window {fmt(lo)} to {fmt(hi)}.")
    culprit = None
    for c in by("witness_alibi"):
        d = c.data
        if d["witness"] == liar:
            culprit = d["suspect"]
            lines.append(f"{eid[id(c)]}: {d['suspect']}'s alibi came from the liar, so {d['suspect']} is not cleared.")
        else:
            lines.append(f"{eid[id(c)]}: {d['witness']} vouches for {d['suspect']} from {fmt(d['start'])} to {fmt(d['end'])}.")
            lines += explain_coverage(d["start"], d["end"], lo, hi, who=f"{d['suspect']}'s alibi")
            if d["start"] <= lo and d["end"] >= hi:
                lines.append(f"So {d['suspect']} ({label_of[d['suspect']]}) is ruled out.")
            else:
                lines.append(f"So this alibi does not clear {d['suspect']}.")

    herr = [f for f in facts if f.role == "herring"]
    lines.append("\nStep 5 - Red herrings.")
    for f in herr:
        lines.append(f"{eid[id(f)]}: " + ("a motive for someone already excluded; motive alone proves nothing."
                                          if f.kind == "motive" else "behaviour or background, not evidence of who or when."))
    answer = agent_solver(case, "", raw)
    lines.append(f"\nConclusion: {culprit} ({answer['most_likely']}) had a key and no truthful alibi for the whole window.")
    lines.append("\n```json\n" + json.dumps(answer, indent=2) + "\n```")
    return "\n".join(lines)


def write_trace(raw: dict) -> str:
    if raw["generator"].get("family") == "composite":
        return write_composite_trace(raw)
    if raw["generator"].get("family") == "liar":
        return write_liar_trace(raw)
    case = CaseDefinition.from_dict(raw)
    facts = [Fact(f["kind"], f["text"], f["data"], f["role"]) for f in raw["generator"]["facts"]]
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}
    label_of = {}
    for i, s in enumerate(case.scenarios, 1):
        label_of["accident" if s.endswith("was an accident.") else s.split(",")[0].strip()] = f"S{i}"
    lo, hi = death_window(facts)
    by_kind: dict[str, list[Fact]] = {}
    for f in facts:
        by_kind.setdefault(f.kind, []).append(f)
    lines: list[str] = []

    # 1. time window
    lines.append("Step 1 - Time of death.")
    for f in by_kind.get("body_temp", []):
        drop = BODY_TEMP - f.data["temp"]
        hours = drop / COOLING_PER_HOUR
        est = round(f.data["discovery"] - hours * 60)
        lines.append(f"{eid[id(f)]}: {f.data['temp']:.1f}°C at {fmt(f.data['discovery'])} is a drop of {drop:.1f}°C, "
                     f"about {hours:.1f} hours of cooling, so death was around {fmt(est)}, within about an hour: "
                     f"{fmt(est - ESTIMATE_TOLERANCE)} to {fmt(est + ESTIMATE_TOLERANCE)}.")
    for f in by_kind.get("last_alive", []):
        lines.append(f"{eid[id(f)]}: the victim was alive at {fmt(f.data['time'])}.")
    lines.append(f"So the death window is {fmt(lo)} to {fmt(hi)}.")

    # 2. accident
    if "accident" in label_of:
        lines.append("\nStep 2 - Accident?")
        h = by_kind.get("homicide", [])
        if h:
            lines.append(f"{eid[id(h[0])]} describes an injury no accident explains, so {label_of['accident']} is ruled out.")

    # 3. access
    lines.append("\nStep 3 - Access.")
    keyholders = set()
    nfe, kh = by_kind.get("no_forced_entry", []), by_kind.get("keyholders", [])
    if nfe and kh:
        keyholders = set(kh[0].data["names"])
        lines.append(f"{eid[id(nfe[0])]}: no forced entry, so the killer used a key. {eid[id(kh[0])]} lists who holds keys.")
        for name, sid in label_of.items():
            if name != "accident" and name not in keyholders:
                lines.append(f"{name} ({sid}) has no key and is ruled out.")

    # 4. alibis
    lines.append(f"\nStep 4 - Alibis against the window {fmt(lo)} to {fmt(hi)}.")
    for f in by_kind.get("alibi", []):
        name = f.data["name"]
        if keyholders and name not in keyholders:
            continue
        a, b = f.data["start"], f.data["end"]
        lines.append(f"{eid[id(f)]}: {name} is verified from {fmt(a)} to {fmt(b)}.")
        lines += explain_coverage(a, b, lo, hi, who=f"{name}'s alibi")
        if a <= lo and b >= hi:
            lines.append(f"So {name} ({label_of[name]}) is ruled out.")
        else:
            lines.append(f"So this alibi does not clear {name}.")
    for f in by_kind.get("testimony", []):
        name = f.data["name"]
        if not keyholders or name in keyholders:
            lines.append(f"{eid[id(f)]}: {name}'s account is unverified testimony, so it does not clear them.")

    # 5. red herrings
    herrings = [f for f in facts if f.role == "herring"]
    if herrings:
        lines.append("\nStep 5 - Red herrings.")
        for f in herrings:
            if f.kind == "motive":
                lines.append(f"{eid[id(f)]}: a motive for {f.data['name']}, who is already excluded; motive alone proves nothing.")
            else:
                lines.append(f"{eid[id(f)]}: has an innocent explanation given in the same line.")

    # 6. conclusion
    answer = agent_solver(case, "", raw)
    top = answer["most_likely"]
    culprit = next(n for n, s in label_of.items() if s == top)
    lines.append(f"\nConclusion: only {culprit} ({top}) had a key and no verified alibi for the whole window.")
    lines.append("\n```json\n" + json.dumps(answer, indent=2) + "\n```")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write SFT traces from the train split.")
    ap.add_argument("--train", default=str(ROOT / "data" / "generated" / "train.jsonl"),
                    help="one or more comma-separated JSONL files, e.g. train.jsonl,liar_train.jsonl")
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "sft_train.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    raws = [json.loads(l) for path in args.train.split(",") for l in open(path, encoding="utf-8") if l.strip()]
    if args.limit:
        raws = raws[:args.limit]
    with open(args.out, "w", encoding="utf-8") as f:
        for raw in raws:
            case = CaseDefinition.from_dict(raw)
            f.write(json.dumps({
                "prompt": [{"role": "user", "content": build_prompt(case)}],
                "completion": [{"role": "assistant", "content": write_trace(raw)}],
                "case_id": raw["id"], "level": raw["generator"]["level"],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(raws)} traces to {args.out}")


if __name__ == "__main__":
    main()
