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
from coverage_text import explain_coverage, explain_coverage_compact, night  # noqa: E402

# Compact traces (--compact): state the midnight rule once, list red herrings
# by ID only, and write the final JSON on one line. Keeps hard cases under
# about 1024 output tokens so small models finish their answer.
COMPACT = False
# State tracking (--track): list the suspects still possible after each step,
# so the conclusion reads off the one left instead of relying on memory.
TRACK = False
# Verbose comparison inside compact traces (--verbose-compare): names each value
# ("the alibi starts 22:50, the window opens 23:05"). SFT v4 showed the terse
# comparison leads to copying errors; the verbose one scored 100% on the probe.
VERBOSE_COMPARE = False
# Index records by person in lying-witness traces (--index-records): each claim
# check quotes that person's short list of records, so the lookup is local.
# SFT v7 failed mostly at finding the right record among 15-20 lines.
INDEX_RECORDS = False
# Reading pass (--read): before reasoning, list what each evidence paragraph
# states in one canonical line. Prose paragraphs hold several facts, and v7.1
# failed the prose reliability cases mostly by misreading them.
READ = False


def _fact_ids(raw: dict) -> tuple[list[Fact], dict[int, str]]:
    """Facts plus the evidence ID each one comes from. Prose cases group several
    facts per paragraph and record it as "para"; generated cases have one each."""
    fl = raw["generator"]["facts"]
    facts = [Fact(f["kind"], f.get("text", ""), f["data"], f["role"]) for f in fl]
    return facts, {id(f): f"E{d.get('para', i)}" for i, (f, d) in enumerate(zip(facts, fl), 1)}


def _read_fact(f: Fact) -> str:
    d = f.data
    k = f.kind
    if k == "discovery":
        return f"body found at {fmt(d['time'])}"
    if k == "body_temp":
        return f"body at {d['temp']:.1f}°C at {fmt(d['discovery'])}"
    if k == "last_alive":
        return f"victim alive at {fmt(d['time'])}"
    if k == "homicide":
        return "an injury no accident explains"
    if k == "no_forced_entry":
        return "no forced entry, so a key was used"
    if k == "keyholders":
        return "key holders: " + ", ".join(d["names"])
    if k == "alibi":
        return f"{d['name']} verified from {fmt(d['start'])} to {fmt(d['end'])}"
    if k == "testimony":
        return f"{d['name']}'s own account, unverified"
    if k == "point":
        return f"{d['name']} at {d['place']} at {fmt(d['time'])}, a single moment"
    if k == "motive":
        return f"a motive for {d['name']}"
    if k == "witness_alibi":
        return (f"{d['witness']} says they were with {d['suspect']} at {d['place']} "
                f"from {fmt(d['start'])} to {fmt(d['end'])}")
    if k == "record" and "atoms" in d:
        p, tm, l, _ = d["atoms"][0]
        return f"record: {p} at {l} at {fmt(tm)}"
    if k == "record":
        return f"record: {d['name']} at {d['place']} at {fmt(d['time'])}"
    if k == "statement":
        atoms = d["atoms"]
        if len(atoms) == 1:
            return f"{d['speaker']} says they were at {atoms[0][2]} at {fmt(atoms[0][1])}"
        return f"{d['speaker']} says they and {atoms[1][0]} were at {atoms[0][2]} at {fmt(atoms[0][1])}"
    if k == "rule":
        return "the rule: one witness lies, records are reliable"
    return "background only, no fact about who or when"


def _read_pass(raw: dict) -> list[str]:
    """One line per evidence paragraph: what it states, in canonical form."""
    facts, eid = _fact_ids(raw)
    per: dict[str, list[str]] = {}
    for f in facts:
        per.setdefault(eid[id(f)], []).append(_read_fact(f))
    fam = raw["generator"].get("family")
    lines = ["Step 0 - What each paragraph states."]
    motives, background = [], []
    for i in range(1, len(raw["evidence"]) + 1):
        e = f"E{i}"
        if e in per and all(x.startswith("a motive for ") for x in per[e]):
            motives.append(f"{e} " + ", ".join(x[len("a motive for "):] for x in per[e]))
        elif e in per and not all(x.startswith("background only") for x in per[e]):
            lines.append(f"{e}: " + "; ".join(x for x in per[e] if not x.startswith("background only")) + ".")
        elif i == 1 and fam in ("liar", "composite"):
            lines.append(f"{e}: the rule: one witness lies, records are reliable.")
        else:
            background.append(e)
    # Paragraphs that state no fact about who or when are grouped to keep traces short.
    if motives:
        lines.append("Motives only: " + "; ".join(motives) + ".")
    if background:
        lines.append("No fact about who or when: " + ", ".join(background) + ".")
    return lines + [""]


def _still(remaining: list[tuple[str, str]], label: str = "Still possible") -> str:
    return f"{label}: " + ", ".join(f"{sid} {name}" for sid, name in remaining) + "."


def _midnight_note(lo: int, hi: int) -> str:
    return (f"Times after midnight get 24 added to the hour so they stay in order; "
            f"the window in these terms is {night(lo)} to {night(hi)}.")


def _answer_block(answer: dict) -> str:
    if COMPACT:
        return "\n```json\n" + json.dumps(answer, separators=(", ", ": ")) + "\n```"
    return "\n```json\n" + json.dumps(answer, indent=2) + "\n```"


def _alive_lines(alive: list[Fact], eid: dict[int, str]) -> list[str]:
    out = [f"{eid[id(f)]}: the victim was alive at {fmt(f.data['time'])}." for f in alive]
    if len(alive) > 1:
        latest = max(alive, key=lambda f: f.data["time"])
        out.append(f"The latest sign of life counts: {fmt(latest.data['time'])} ({eid[id(latest)]}).")
    return out


def write_liar_trace(raw: dict) -> str:
    """Trace for the witness-consistency family: find the statement a record contradicts."""
    case = CaseDefinition.from_dict(raw)
    facts = raw["generator"]["facts"]
    eid = {i: f"E{f.get('para', i)}" for i, f in enumerate(facts, 1)}
    names = [s[: -len(" is lying.")] for s in case.scenarios]
    sid = {n: f"S{i}" for i, n in enumerate(names, 1)}
    records = [(i, tuple(a)) for i, f in enumerate(facts, 1) if f["kind"] == "record" for a in f["data"]["atoms"]]
    lines = (_read_pass(raw) if READ else []) + ["Step 1 - Records are reliable, so each fixes where one person was at one time."]
    for i, (p, tm, l, _) in records:
        lines.append(f"{eid[i]}: {p} was at {l} at {fmt(tm)}.")
    lines.append("\nStep 2 - Check what each statement implies against the records.")
    liar, clash = None, None
    if VERBOSE_COMPARE:
        # Every statement is expanded into the places it implies, and each is
        # checked against a record for the same person and time, if one exists.
        rec_at = {(rp, rt): (j, rl) for j, (rp, rt, rl, _) in records}
        by_person: dict[str, list] = {}
        for j, (rp, rt, rl, _) in records:
            by_person.setdefault(rp, []).append((rt, j, rl))
        if INDEX_RECORDS:
            lines.append("Records by person:")
            for rp in sorted(by_person):
                lines.append(f"{rp}: " + "; ".join(f"{eid[j]} {rl} at {fmt(rt)}" for rt, j, rl in sorted(by_person[rp])) + ".")
        for i, f in enumerate(facts, 1):
            if f["kind"] != "statement":
                continue
            spk = f["data"]["speaker"]
            false_here = False
            for p, tm, l, _ in (tuple(a) for a in f["data"]["atoms"]):
                who = f"{spk} was" if p == spk else f"{p} was"
                line = f"{eid[i]}: {spk}'s statement means {who} at {l} at {fmt(tm)}. "
                if INDEX_RECORDS:
                    mine = sorted(by_person.get(p, []))
                    line += (f"Records for {p}: " + "; ".join(f"{eid[j]} {rl} at {fmt(rt)}" for rt, j, rl in mine) + ". "
                             if mine else f"Records for {p}: none. ")
                if (p, tm) in rec_at:
                    j, rl = rec_at[(p, tm)]
                    if rl == l:
                        line += f"{eid[j]} agrees."
                    else:
                        line += f"{eid[j]} shows {p} at {rl} at {fmt(tm)}; different place, so this is false."
                        false_here = True
                else:
                    line += f"No record for {p} at {fmt(tm)}, so nothing contradicts it."
                lines.append(line)
            if false_here:
                liar = spk
        lines.append(f"Only {liar}'s statement is contradicted by a record, so {liar} is lying and everyone else fits.")
        lines.append("\nStep 3 - Red herrings: " + ", ".join(raw["answer_key"]["red_herrings"])
                     + " (behaviour or background, not where anyone was).")
        answer = agent_solver(case, "", raw)
        lines.append(f"\nConclusion: {liar} ({sid[liar]}) is lying.")
        lines.append(_answer_block(answer))
        return "\n".join(lines)
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
    lines.append(_answer_block(answer))
    return "\n".join(lines)


def write_composite_trace(raw: dict) -> str:
    """Trace for the combined family: window, lying witness, voided alibi, access, alibis."""
    case = CaseDefinition.from_dict(raw)
    facts, eid = _fact_ids(raw)
    label_of = {}
    for i, s in enumerate(case.scenarios, 1):
        label_of["accident" if s.endswith("was an accident.") else s.split(",")[0].strip()] = f"S{i}"
    lo, hi = death_window(facts)
    by = lambda k: [f for f in facts if f.kind == k]  # noqa: E731
    lines = (_read_pass(raw) if READ else []) + ["Step 1 - Time of death."]
    for f in by("body_temp"):
        drop = BODY_TEMP - f.data["temp"]; hours = drop / COOLING_PER_HOUR
        est = round(f.data["discovery"] - hours * 60)
        lines.append(f"{eid[id(f)]}: a drop of {drop:.1f}°C is about {hours:.1f} hours, so death was around {fmt(est)}, "
                     f"within an hour: {fmt(est - ESTIMATE_TOLERANCE)} to {fmt(est + ESTIMATE_TOLERANCE)}.")
    lines += _alive_lines(by("last_alive"), eid)
    lines.append(f"So the death window is {fmt(lo)} to {fmt(hi)}.")
    if COMPACT:
        lines.append(_midnight_note(lo, hi))
    for f in by("homicide"):
        lines.append(f"{eid[id(f)]} rules out an accident ({label_of['accident']}).")

    lines.append("\nStep 2 - Find the lying witness by checking each alibi claim against the records.")
    liar = None
    if VERBOSE_COMPARE:
        # Every claim is checked against every record about the witness or the
        # person they vouch for, so finding the liar is shown, not asserted.
        for c in by("witness_alibi"):
            d = c.data
            lines.append(f"{eid[id(c)]}: {d['witness']} says they were with {d['suspect']} at {d['place']} "
                         f"from {fmt(d['start'])} to {fmt(d['end'])}.")
            if INDEX_RECORDS:
                for who in (d["witness"], d["suspect"]):
                    mine = sorted((r.data["time"], eid[id(r)], r.data["place"]) for r in by("record") if r.data["name"] == who)
                    lines.append(f"Records for {who}: " + ("; ".join(f"{e} {pl} at {fmt(tm)}" for tm, e, pl in mine)
                                                           if mine else "none") + ".")
            false_claim = False
            for r in by("record"):
                rd = r.data
                if rd["name"] not in (d["witness"], d["suspect"]):
                    continue
                t0, a0, b0 = rd["time"], d["start"], d["end"]
                head = f"{eid[id(r)]}: {rd['name']} at {rd['place']} at {fmt(t0)}. "
                if t0 < a0:
                    lines.append(head + f"{night(t0)} < {night(a0)}, so before the claimed time; it does not test the claim.")
                elif t0 > b0:
                    lines.append(head + f"{night(t0)} > {night(b0)}, so after the claimed time; it does not test the claim.")
                elif rd["place"] == d["place"]:
                    lines.append(head + f"{night(a0)} <= {night(t0)} <= {night(b0)}, so inside the claimed time; same place, so it fits.")
                else:
                    lines.append(head + f"{night(a0)} <= {night(t0)} <= {night(b0)}, so inside the claimed time; "
                                 f"different place ({rd['place']} vs {d['place']}), so the claim is false.")
                    false_claim = True
            if false_claim:
                liar = d["witness"]
                lines.append(f"So {d['witness']} is lying and this alibi does not count.")
            else:
                lines.append(f"So {d['witness']}'s claim fits the records.")
    for c in ([] if VERBOSE_COMPARE else by("witness_alibi")):
        d = c.data
        for r in by("record"):
            rd = r.data
            if rd["name"] in (d["witness"], d["suspect"]) and d["start"] <= rd["time"] <= d["end"] and rd["place"] != d["place"]:
                liar = d["witness"]
                lines.append(f"{eid[id(c)]} says {d['witness']} and {d['suspect']} were at {d['place']} from "
                             f"{fmt(d['start'])} to {fmt(d['end'])}, but {eid[id(r)]} shows {rd['name']} at {rd['place']} "
                             f"at {fmt(rd['time'])}. So {d['witness']} is lying and this alibi does not count.")
    if not VERBOSE_COMPARE:
        lines.append("Every other alibi claim fits the records.")

    lines.append("\nStep 3 - Access.")
    remaining = sorted(((s, n) for n, s in label_of.items() if n != "accident"), key=lambda x: int(x[0][1:]))
    if TRACK:
        lines.append(_still(remaining, "Suspects"))
    kh = by("keyholders")[0]
    keyholders = set(kh.data["names"])
    lines.append(f"{eid[id(by('no_forced_entry')[0])]}: no forced entry, so the killer used a key. {eid[id(kh)]} lists the key holders.")
    for name, sid in label_of.items():
        if name != "accident" and name not in keyholders:
            lines.append(f"{name} ({sid}) has no key and is ruled out.")
    remaining = [(s, n) for s, n in remaining if n in keyholders]
    if TRACK:
        lines.append(_still(remaining))

    lines.append(f"\nStep 4 - Truthful alibis against the window {fmt(lo)} to {fmt(hi)}.")
    culprit = None
    for c in by("witness_alibi"):
        d = c.data
        if d["witness"] == liar:
            culprit = d["suspect"]
            lines.append(f"{eid[id(c)]}: {d['suspect']}'s alibi came from the liar, so {d['suspect']} is not cleared.")
        else:
            lines.append(f"{eid[id(c)]}: {d['witness']} vouches for {d['suspect']} from {fmt(d['start'])} to {fmt(d['end'])}.")
            lines += (explain_coverage_compact(d["start"], d["end"], lo, hi, rewrite=False) if COMPACT and not VERBOSE_COMPARE
                      else explain_coverage(d["start"], d["end"], lo, hi, who=f"{d['suspect']}'s alibi"))
            if d["start"] <= lo and d["end"] >= hi:
                lines.append(f"So {d['suspect']} ({label_of[d['suspect']]}) is ruled out.")
            else:
                lines.append(f"So this alibi does not clear {d['suspect']}.")

    cleared = {c.data["suspect"] for c in by("witness_alibi")
               if c.data["witness"] != liar and c.data["start"] <= lo and c.data["end"] >= hi}
    remaining = [(s, n) for s, n in remaining if n not in cleared]
    if TRACK:
        lines.append(_still(remaining))

    herr = [f for f in facts if f.role == "herring"]
    if COMPACT:
        lines.append("\nStep 5 - Red herrings: " + ", ".join(raw["answer_key"]["red_herrings"])
                     + " (motives of people already excluded, behaviour, or background noise).")
    else:
        lines.append("\nStep 5 - Red herrings.")
        for f in herr:
            lines.append(f"{eid[id(f)]}: " + ("a motive for someone already excluded; motive alone proves nothing."
                                              if f.kind == "motive" else "behaviour or background, not evidence of who or when."))
    answer = agent_solver(case, "", raw)
    if TRACK:
        lines.append(f"\nConclusion: only {answer['most_likely']} {culprit} is left, so {culprit} ({answer['most_likely']}) is the culprit.")
    else:
        lines.append(f"\nConclusion: {culprit} ({answer['most_likely']}) had a key and no truthful alibi for the whole window.")
    lines.append(_answer_block(answer))
    return "\n".join(lines)


def write_trace(raw: dict) -> str:
    if raw["generator"].get("family") == "composite":
        return write_composite_trace(raw)
    if raw["generator"].get("family") == "liar":
        return write_liar_trace(raw)
    case = CaseDefinition.from_dict(raw)
    facts, eid = _fact_ids(raw)
    label_of = {}
    for i, s in enumerate(case.scenarios, 1):
        label_of["accident" if s.endswith("was an accident.") else s.split(",")[0].strip()] = f"S{i}"
    lo, hi = death_window(facts)
    by_kind: dict[str, list[Fact]] = {}
    for f in facts:
        by_kind.setdefault(f.kind, []).append(f)
    lines: list[str] = _read_pass(raw) if READ else []

    # 1. time window
    lines.append("Step 1 - Time of death.")
    for f in by_kind.get("body_temp", []):
        drop = BODY_TEMP - f.data["temp"]
        hours = drop / COOLING_PER_HOUR
        est = round(f.data["discovery"] - hours * 60)
        lines.append(f"{eid[id(f)]}: {f.data['temp']:.1f}°C at {fmt(f.data['discovery'])} is a drop of {drop:.1f}°C, "
                     f"about {hours:.1f} hours of cooling, so death was around {fmt(est)}, within about an hour: "
                     f"{fmt(est - ESTIMATE_TOLERANCE)} to {fmt(est + ESTIMATE_TOLERANCE)}.")
    lines += _alive_lines(by_kind.get("last_alive", []), eid)
    lines.append(f"So the death window is {fmt(lo)} to {fmt(hi)}.")
    if COMPACT:
        lines.append(_midnight_note(lo, hi))

    # 2. accident
    if "accident" in label_of:
        lines.append("\nStep 2 - Accident?")
        h = by_kind.get("homicide", [])
        if h:
            lines.append(f"{eid[id(h[0])]} describes an injury no accident explains, so {label_of['accident']} is ruled out.")

    # 3. access
    lines.append("\nStep 3 - Access.")
    remaining = sorted(((s, n) for n, s in label_of.items() if n != "accident"), key=lambda x: int(x[0][1:]))
    if TRACK:
        lines.append(_still(remaining, "Suspects"))
    keyholders = set()
    nfe, kh = by_kind.get("no_forced_entry", []), by_kind.get("keyholders", [])
    if nfe and kh:
        keyholders = set(kh[0].data["names"])
        lines.append(f"{eid[id(nfe[0])]}: no forced entry, so the killer used a key. {eid[id(kh[0])]} lists who holds keys.")
        for name, sid in label_of.items():
            if name != "accident" and name not in keyholders:
                lines.append(f"{name} ({sid}) has no key and is ruled out.")
        remaining = [(s, n) for s, n in remaining if n in keyholders]
    if TRACK:
        lines.append(_still(remaining))

    # 4. alibis
    lines.append(f"\nStep 4 - Alibis against the window {fmt(lo)} to {fmt(hi)}.")
    done: set[str] = set()
    for f in by_kind.get("alibi", []):
        name = f.data["name"]
        if keyholders and name not in keyholders:
            continue
        if name in done:
            lines.append(f"{eid[id(f)]}: another record for {name}, who is already ruled out.")
            continue
        a, b = f.data["start"], f.data["end"]
        if a <= lo and b >= hi:
            done.add(name)
        lines.append(f"{eid[id(f)]}: {name} is verified from {fmt(a)} to {fmt(b)}.")
        lines += (explain_coverage_compact(a, b, lo, hi, rewrite=False) if COMPACT and not VERBOSE_COMPARE
                  else explain_coverage(a, b, lo, hi, who=f"{name}'s alibi"))
        if a <= lo and b >= hi:
            lines.append(f"So {name} ({label_of[name]}) is ruled out.")
        else:
            lines.append(f"So this alibi does not clear {name}.")
    for f in by_kind.get("testimony", []):
        name = f.data["name"]
        if not keyholders or name in keyholders:
            lines.append(f"{eid[id(f)]}: {name}'s account is unverified testimony, so it does not clear them.")
    for f in by_kind.get("point", []):
        name = f.data["name"]
        if not keyholders or name in keyholders:
            lines.append(f"{eid[id(f)]}: {name} at {f.data['place']} at {fmt(f.data['time'])} is a single moment, "
                         "not the whole window, so it does not clear them.")
    cleared = {f.data["name"] for f in by_kind.get("alibi", []) if f.data["start"] <= lo and f.data["end"] >= hi}
    remaining = [(s, n) for s, n in remaining if n not in cleared]
    if TRACK:
        lines.append(_still(remaining))

    # 5. red herrings
    herrings = [f for f in facts if f.role == "herring"]
    if raw["answer_key"]["red_herrings"] and COMPACT:
        lines.append("\nStep 5 - Red herrings: " + ", ".join(raw["answer_key"]["red_herrings"])
                     + " (motives of people already excluded, or noise with an innocent explanation).")
    elif herrings:
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
    if TRACK:
        lines.append(f"\nConclusion: only {top} {culprit} is left, so {culprit} ({top}) is the culprit.")
    else:
        lines.append(f"\nConclusion: only {culprit} ({top}) had a key and no verified alibi for the whole window.")
    lines.append(_answer_block(answer))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write SFT traces from the train split.")
    ap.add_argument("--train", default=str(ROOT / "data" / "generated" / "train.jsonl"),
                    help="one or more comma-separated JSONL files, e.g. train.jsonl,liar_train.jsonl")
    ap.add_argument("--out", default=str(ROOT / "data" / "generated" / "sft_train.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--compact", action="store_true", help="shorter traces that fit small output budgets")
    ap.add_argument("--track", action="store_true", help="list the suspects still possible after each step")
    ap.add_argument("--verbose-compare", action="store_true", help="name each value in alibi comparisons")
    ap.add_argument("--index-records", action="store_true", help="group records by person in lying-witness traces")
    ap.add_argument("--read", action="store_true", help="start with what each evidence paragraph states")
    args = ap.parse_args()
    global COMPACT, TRACK, VERBOSE_COMPARE, INDEX_RECORDS, READ
    COMPACT, TRACK, VERBOSE_COMPARE, INDEX_RECORDS, READ = (args.compact, args.track, args.verbose_compare,
                                                            args.index_records, args.read)
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
