"""Combined family: a timeline case where one alibi comes from a lying witness.

Solving it takes both kinds of reasoning in sequence:

  1. Fix the death window from body cooling and the last sign of life.
  2. Every key holder's alibi comes from a witness ("I was with X at P from
     a to b").  Exactly one witness is lying, and it is the one covering for
     the culprit.  A reliable record exposes the lie: a record of the witness
     somewhere else (level 2), or a sighting of the culprit somewhere else
     during the claimed alibi (level 3).
  3. Void the liar's alibi, then apply access and the remaining alibis.

Hypotheses are pairs (liar, culprit).  A liar is consistent if every other
witness's claim fits the records; a culprit is consistent if they had a key
and no truthful alibi covers the window.  A case is kept only if exactly one
culprit survives across all consistent liars.

Mention balance: every suspect is named four times (key sentence,
whereabouts, record, motive) and every witness three times (alibi claim,
record, demeanour).
"""

from __future__ import annotations

import random
from typing import Any

from .generator import BODY_TEMP, COOLING_PER_HOUR, POOLS, Fact, death_window, fmt
from .liar import DEMEANOUR, PLACES, SOURCES

FAMILY = "composite"

DIFFICULTY = {
    2: {"suspects": 4, "herrings": (1, 2), "mechanism": "witness_record", "margin": (10, 60)},
    3: {"suspects": 5, "herrings": (2, 3), "mechanism": "suspect_record", "margin": (5, 40)},
}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def _claim_conflicts(claim: dict, records: list[dict]) -> bool:
    """A claim 'W was with X at P from a to b' conflicts with any record placing
    W or X somewhere other than P at a time inside [a, b]."""
    for r in records:
        if r["name"] in (claim["witness"], claim["suspect"]) and claim["start"] <= r["time"] <= claim["end"] \
                and r["place"] != claim["place"]:
            return True
    return False


def consistent_liars(facts: list[Fact], witnesses: list[str]) -> set[str]:
    records = [f.data for f in facts if f.kind == "record"]
    claims = [f.data for f in facts if f.kind == "witness_alibi"]
    out = set()
    for w in witnesses:
        if not any(_claim_conflicts(c, records) for c in claims if c["witness"] != w):
            out.add(w)
    return out


def solve(facts: list[Fact], suspects: list[str], witnesses: list[str], with_accident: bool) -> set[str]:
    lo, hi = death_window(facts)
    kinds = {f.kind for f in facts}
    keyholders = next((set(f.data["names"]) for f in facts if f.kind == "keyholders"), None)
    access_matters = "no_forced_entry" in kinds and keyholders is not None
    claims = [f.data for f in facts if f.kind == "witness_alibi"]
    verified = [f.data for f in facts if f.kind == "alibi"]
    out: set[str] = set()
    for liar in consistent_liars(facts, witnesses):
        for s in suspects:
            if access_matters and s not in keyholders:
                continue
            cleared = any(c["suspect"] == s and c["witness"] != liar and c["start"] <= lo and c["end"] >= hi
                          for c in claims)
            cleared |= any(a["name"] == s and a["start"] <= lo and a["end"] >= hi for a in verified)
            if not cleared:
                out.add(s)
    if with_accident and "homicide" not in kinds:
        out.add("accident")
    return out


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

def simulate(rng: random.Random, level: int, pool_name: str) -> dict[str, Any] | None:
    cfg, pool = DIFFICULTY[level], POOLS[pool_name]
    n = cfg["suspects"]
    people = [f"{a} {b}" for a, b in zip(rng.sample(pool["first"], 2 * n + 1), rng.sample(pool["last"], 2 * n + 1))]
    victim, suspects = people[0], people[1:n + 1]
    roles = dict(zip(suspects, rng.sample(pool["role"], n)))
    place = rng.choice(pool["place"])
    places = PLACES[pool_name]
    culprit = rng.choice(suspects)
    distractors = [s for s in suspects if s != culprit]

    t_death = rng.randint(9 * 60, 13 * 60 + 30)
    discovery = t_death + rng.randint(7 * 60, 10 * 60)
    temp = round((BODY_TEMP - COOLING_PER_HOUR * (discovery - t_death) / 60) * 2) / 2
    last_alive = t_death - rng.randint(10, 50)
    facts: list[Fact] = [
        Fact("rule", "Records are reliable. Exactly one witness below is lying; the others tell the truth.", {}, "neutral"),
        Fact("discovery", f"{victim} was found dead in the {place} at {fmt(discovery)}.", {"time": discovery}, "neutral"),
        Fact("body_temp",
             f"At {fmt(discovery)} the body temperature was {temp:.1f}°C. In that room a body cools by about "
             f"{COOLING_PER_HOUR:.0f}°C per hour from {BODY_TEMP:.0f}°C, and the estimate is reliable to within about an hour.",
             {"temp": temp, "discovery": discovery}, "hard"),
        Fact("last_alive", f"{victim}'s phone sent a text message at {fmt(last_alive)}.", {"time": last_alive}, "hard"),
        Fact("homicide", "The fatal injury is a blow to the back of the head that no fall in the room could explain.", {}, "hard"),
        Fact("no_forced_entry", f"There was no sign of forced entry; the {place} door had been locked and unlocked with a key.", {}, "hard"),
    ]
    lo, hi = death_window(facts)

    access_excluded = set(rng.sample(distractors, rng.randint(1, max(1, len(distractors) // 2))))
    keyholders = [s for s in suspects if s not in access_excluded]
    rng.shuffle(keyholders)
    no_key = [s for s in suspects if s in access_excluded]
    others = " and ".join(no_key) if len(no_key) <= 2 else ", ".join(no_key[:-1]) + f" and {no_key[-1]}"
    holders = ", ".join([victim] + keyholders[:-1]) + f" and {keyholders[-1]}"
    facts.append(Fact("keyholders", f"Keys to the {place} are held only by {holders}; {others} never had one.",
                      {"names": sorted(keyholders)}, "hard"))

    # One witness per key holder; the culprit's witness is the liar.
    witnesses = people[n + 1:n + 1 + len(keyholders)]
    vouch = dict(zip(keyholders, witnesses))
    liar = vouch[culprit]
    lo_m, hi_m = cfg["margin"]
    claim_facts: dict[str, Fact] = {}
    for s, w in vouch.items():
        a, b = lo - rng.randint(lo_m, hi_m), hi + rng.randint(lo_m, hi_m)
        p = rng.choice(places)
        data = {"witness": w, "suspect": s, "place": p, "start": a, "end": b}
        claim_facts[s] = Fact("witness_alibi", f'{w} says: "I was with {s} at {p} from {fmt(a)} to {fmt(b)}."',
                              data, "hard" if w == liar else "neutral")
    facts.extend(claim_facts.values())

    def record(name: str, t: int, p: str, role: str = "neutral") -> Fact:
        return Fact("record", rng.choice(SOURCES[pool_name]).format(p=name, l=p, t=fmt(t)),
                    {"name": name, "time": t, "place": p}, role)

    lie = claim_facts[culprit].data
    inside = rng.randint(lie["start"] + 5, lie["end"] - 5)
    elsewhere = rng.choice([p for p in places if p != lie["place"]])

    # Records for witnesses: truthful ones agree with their claim; the liar's may expose the lie.
    for s, w in vouch.items():
        c = claim_facts[s].data
        if w == liar and cfg["mechanism"] == "witness_record":
            facts.append(record(w, inside, elsewhere, "hard"))
        else:
            facts.append(record(w, rng.randint(c["start"] + 5, c["end"] - 5), c["place"]))

    # Records for suspects: consistent with any truthful claim; for the culprit at level 3, the exposing sighting.
    for s in suspects:
        if s == culprit and cfg["mechanism"] == "suspect_record":
            facts.append(record(s, inside, elsewhere, "hard"))
        elif s in claim_facts:
            c = claim_facts[s].data
            facts.append(record(s, rng.randint(c["start"] + 5, c["end"] - 5), c["place"]))
        else:
            facts.append(record(s, lo - rng.randint(60, 180), rng.choice(places)))

    # Whereabouts for key-less suspects (never clearing; they are excluded by access).
    for s in no_key:
        facts.append(Fact("testimony", f"{s} says they were at {rng.choice(places)} all evening.", {"name": s}, "neutral"))

    for s in suspects:
        text = f"{s}, the {roles[s]}, " + rng.choice(pool["motive"]).format(v=victim) + "."
        facts.append(Fact("motive", text, {"name": s}, "neutral" if s == culprit else "herring"))
    for w in witnesses:
        facts.append(Fact("demeanour", rng.choice(DEMEANOUR).format(w=w), {"name": w}, "herring"))
    for text in rng.sample(pool["salient"], min(rng.randint(*cfg["herrings"]), len(pool["salient"]))):
        facts.append(Fact("salient", text.format(t=fmt(rng.randint(lo - 60, hi))), {}, "herring"))

    return {"victim": victim, "suspects": suspects, "witnesses": witnesses, "roles": roles, "place": place,
            "culprit": culprit, "liar": liar, "facts": facts, "t_death": t_death}


def build_case(world: dict[str, Any], case_id: str, level: int, rng: random.Random) -> dict[str, Any] | None:
    facts, suspects, witnesses = world["facts"], world["suspects"], world["witnesses"]
    culprit = world["culprit"]

    def sol(fs: list[Fact]) -> set[str]:
        return solve(fs, suspects, witnesses, with_accident=True)

    if sol(facts) != {culprit} or consistent_liars(facts, witnesses) != {world["liar"]}:
        return None
    for f in facts:
        if f.role == "herring" and sol([g for g in facts if g is not f]) != {culprit}:
            return None
    lo, hi = death_window(facts)
    if not lo <= world["t_death"] <= hi:
        return None

    rule = [f for f in facts if f.kind == "rule"]
    body = [f for f in facts if f.kind != "rule"]
    rng.shuffle(body)
    facts = rule + body
    labels = suspects + ["accident"]
    rng.shuffle(labels)
    text = {s: f"{s}, the {world['roles'][s]}, killed {world['victim']}." for s in suspects}
    text["accident"] = f"{world['victim']}'s death was an accident."
    sid = {lab: f"S{i}" for i, lab in enumerate(labels, 1)}
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}
    key = [eid[id(f)] for f in facts if f.kind != "rule" and sol([g for g in facts if g is not f]) != {culprit}]
    if not key:
        return None

    return {
        "id": case_id,
        "title": f"The alibi in the {world['place']}",
        "category": "generated-composite",
        "summary": (f"{world['victim']} was found dead in the {world['place']}. {len(suspects)} people are under "
                    "suspicion, and witnesses have vouched for some of them. One witness is lying. "
                    "Work out who killed the victim and when."),
        "evidence": [f.text for f in facts],
        "detective_questions": [
            "What window of time does the physical evidence allow for the death?",
            "Which witness statement is contradicted by a reliable record?",
            "Once that witness is discounted, whose alibi still covers the whole window?",
            "Who could have entered the room?",
        ],
        "analysis_protocol": ["Fix the time window from physical evidence.", "Find the lying witness using records.",
                              "Discard the liar's alibi, then apply access and the remaining alibis."],
        "scenarios": [text[l] for l in labels],
        "hidden_truth": (f"{culprit} killed {world['victim']} between {fmt(lo)} and {fmt(hi)}; "
                         f"{world['liar']} lied to cover for them."),
        "false_narrative": "",
        "requires_all": [],
        "teaches": [],
        "solution": {"unlock_key": "GENERATED"},
        "answer_key": {
            "true_scenario": sid[culprit],
            "ruled_out": sorted((sid[l] for l in labels if l != culprit), key=lambda x: int(x[1:])),
            "key_evidence": key,
            "red_herrings": [eid[id(f)] for f in facts if f.role == "herring"],
            "time_window": {"label": "time of death", "earliest": fmt(lo), "latest": fmt(hi)},
        },
        "generator": {"version": 2, "family": FAMILY, "level": level, "witnesses": witnesses,
                      "facts": [f.to_dict() for f in facts]},
    }


def generate(n: int, seed: int, levels: tuple[int, ...] = (2, 3), pool: str = "A",
             prefix: str = "C") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("composite generator rejected too many worlds")
        level = levels[len(cases) % len(levels)]
        world = simulate(rng, level, pool)
        if world is None:
            continue
        case = build_case(world, f"{prefix}{seed:03d}-{len(cases) + 1:05d}", level, rng)
        if case is not None:
            cases.append(case)
    return cases
