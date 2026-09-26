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


def write_trace(raw: dict) -> str:
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
        if a <= lo and b >= hi:
            lines.append(f"{eid[id(f)]}: {name} is verified from {fmt(a)} to {fmt(b)}, covering the whole window, "
                         f"so {label_of[name]} is ruled out.")
        else:
            lines.append(f"{eid[id(f)]}: {name} is verified only from {fmt(a)} to {fmt(b)}, which does not cover "
                         f"the whole window, so it does not clear them.")
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
    ap.add_argument("--train", default=str(ROOT / "data" / "generated" / "train.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "sft_train.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    raws = [json.loads(l) for l in open(args.train, encoding="utf-8") if l.strip()]
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
