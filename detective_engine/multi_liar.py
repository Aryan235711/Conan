"""v11 family 1: how many witnesses are lying, and which?

The lying-witness family always has exactly one liar, so a model can learn to
stop after the first contradiction. Here the rule states how many witnesses
lie (one or two, mixed within the split), so the rule must be read and every
statement checked: with two liars, finding one is not enough.

Generation follows the lying-witness family (detective_engine/liar.py):

  1. Simulate where each witness was at each hour.
  2. Write truthful statements, and for each liar one false statement that a
     reliable record exposes, either directly (a record of the liar elsewhere)
     or through a false sighting (a record of the person they claim to have
     seen, somewhere else at that time).
  3. Solve: a hypothesis "exactly these k witnesses lie" is consistent if the
     records plus everyone else's statements have a model and each
     hypothesised liar has at least one statement that can be false in it.
     Keep the case only if exactly one set of k witnesses is consistent.

Scenarios are the k-sized sets of witnesses ("A is lying." or "A and B are
lying."), so a two-liar case with five witnesses has ten scenarios.
"""

from __future__ import annotations

import random
from itertools import combinations
from typing import Any

from .generator import POOLS, Fact, fmt
from .liar import DEMEANOUR, PLACES, SOURCES, satisfiable

FAMILY = "multi_liar"

DIFFICULTY = {
    2: {"witnesses": 4, "slots": 5, "places": 5, "salient": 1, "p_sighting": 0.0},
    3: {"witnesses": 5, "slots": 6, "places": 6, "salient": 2, "p_sighting": 0.6},
}
WORDS = {1: "one", 2: "two"}


def rule_text(k: int) -> str:
    if k == 1:
        return "Exactly one witness is lying; the others tell the truth. Records are reliable."
    return f"Exactly {WORDS[k]} witnesses are lying; the others tell the truth. Records are reliable."


def _atoms(f: Fact) -> list[tuple]:
    return [tuple(a) for a in f.data.get("atoms", [])]


def solve(facts: list[Fact], witnesses: list[str], n_places: int, k: int) -> set[frozenset]:
    records = [a for f in facts if f.kind == "record" for a in _atoms(f)]
    out: set[frozenset] = set()
    for hyp in combinations(witnesses, k):
        others = [a for f in facts if f.kind == "statement" and f.data["speaker"] not in hyp for a in _atoms(f)]
        base = records + others
        if not satisfiable(base, n_places):
            continue
        ok = all(
            any(satisfiable(base + [(p, t, l, not pos)], n_places)
                for f in facts if f.kind == "statement" and f.data["speaker"] == h for (p, t, l, pos) in _atoms(f))
            for h in hyp)
        if ok:
            out.add(frozenset(hyp))
    return out


def simulate(rng: random.Random, level: int, pool_name: str, k: int) -> dict[str, Any] | None:
    cfg, pool = DIFFICULTY[level], POOLS[pool_name]
    n = cfg["witnesses"]
    ws = [f"{a} {b}" for a, b in zip(rng.sample(pool["first"], n), rng.sample(pool["last"], n))]
    places = rng.sample(PLACES[pool_name], cfg["places"])
    start = rng.randint(6, 8) * 60
    times = [start + 60 * i for i in range(cfg["slots"])]
    world = {(w, t): rng.choice(places) for w in ws for t in times}
    liars = rng.sample(ws, k)
    mech = {L: ("sighting" if rng.random() < cfg["p_sighting"] else "direct") for L in liars}

    while True:                                        # each witness sees exactly one other, and is seen once
        seen = ws[:]
        rng.shuffle(seen)
        if all(a != b for a, b in zip(ws, seen)):
            break
    sees = dict(zip(ws, seen))
    sight_t = dict(zip(ws, rng.sample(times, n)))      # a distinct hour for each sighting
    for w in ws:                                       # true sightings force the two people together
        if not (w in liars and mech[w] == "sighting"):
            world[(sees[w], sight_t[w])] = world[(w, sight_t[w])]

    facts: list[Fact] = [Fact("rule", rule_text(k), {"k": k}, "neutral")]
    rec_cell: dict[str, int] = {}
    self_t: dict[str, int] = {}
    for L in liars:
        if mech[L] == "direct":
            t = rng.choice([x for x in times if x != sight_t[L]])
            fake = rng.choice([l for l in places if l != world[(L, t)]])
            facts.append(Fact("statement", f'{L} says: "I was at {fake} at {fmt(t)}."',
                              {"speaker": L, "atoms": [[L, t, fake, True]]}, "hard"))
            if L in rec_cell:
                return None
            rec_cell[L] = t
            self_t[L] = t
        else:
            q, t = sees[L], sight_t[L]
            fake = rng.choice([l for l in places if l != world[(q, t)]])
            world[(L, t)] = fake                       # the liar really was there; q was not
            facts.append(Fact("statement", f'{L} says: "I saw {q} at {fake} at {fmt(t)}."',
                              {"speaker": L, "atoms": [[L, t, fake, True], [q, t, fake, True]]}, "hard"))
            if q in rec_cell:
                return None
            rec_cell[q] = t

    for w in ws:                                       # everyone else's sightings are true
        if w in liars and mech[w] == "sighting":
            continue
        t = sight_t[w]
        l = world[(w, t)]
        facts.append(Fact("statement", f'{w} says: "I saw {sees[w]} at {l} at {fmt(t)}."',
                          {"speaker": w, "atoms": [[w, t, l, True], [sees[w], t, l, True]]}, "neutral"))
    for w in ws:                                       # true place claims for the others
        if w in self_t:
            continue
        t = rng.choice([x for x in times if x != sight_t[w]])
        facts.append(Fact("statement", f'{w} says: "I was at {world[(w, t)]} at {fmt(t)}."',
                          {"speaker": w, "atoms": [[w, t, world[(w, t)], True]]}, "neutral"))
    for w in ws:                                       # one reliable record per person
        t = rec_cell.get(w, rng.choice(times))
        facts.append(Fact("record", rng.choice(SOURCES[pool_name]).format(p=w, l=world[(w, t)], t=fmt(t)),
                          {"atoms": [[w, t, world[(w, t)], True]]}, "hard" if w in rec_cell else "neutral"))
    for w in ws:
        facts.append(Fact("demeanour", rng.choice(DEMEANOUR).format(w=w), {"name": w}, "herring"))
    for text in rng.sample(pool["salient"], cfg["salient"]):
        facts.append(Fact("salient", text.format(t=fmt(rng.choice(times))), {}, "herring"))
    return {"witnesses": ws, "liars": liars, "k": k, "facts": facts, "n_places": len(places)}


def _scenario_text(group: tuple[str, ...]) -> str:
    return f"{group[0]} is lying." if len(group) == 1 else f"{group[0]} and {group[1]} are lying."


def build_case(world: dict[str, Any], case_id: str, level: int, rng: random.Random) -> dict[str, Any] | None:
    facts, ws, k, n_places = world["facts"], world["witnesses"], world["k"], world["n_places"]
    truth = frozenset(world["liars"])
    if solve(facts, ws, n_places, k) != {truth}:
        return None
    for f in facts:
        if f.role == "herring" and solve([g for g in facts if g is not f], ws, n_places, k) != {truth}:
            return None

    rule = [f for f in facts if f.kind == "rule"]
    body = [f for f in facts if f.kind != "rule"]
    rng.shuffle(body)
    facts = rule + body
    groups = [tuple(sorted(g, key=ws.index)) for g in combinations(ws, k)]
    rng.shuffle(groups)
    sid = {frozenset(g): f"S{i}" for i, g in enumerate(groups, 1)}
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}
    key = [eid[id(f)] for f in facts
           if f.kind != "rule" and solve([g for g in facts if g is not f], ws, n_places, k) != {truth}]
    if not key:
        return None
    many = "two of them are" if k == 2 else "one of them is"
    return {
        "id": case_id,
        "title": "How many are lying?",
        "category": "generated-multi-liar",
        "summary": (f"{len(ws)} witnesses gave statements about one evening. Exactly {many} lying. "
                    "Work out who, using the records and the other statements."),
        "evidence": [f.text for f in facts],
        "detective_questions": [
            "How many witnesses does the rule say are lying?",
            "Which statements does a record contradict?",
            "Does every other statement fit the records once the liars are set aside?",
        ],
        "analysis_protocol": ["Read how many witnesses lie.", "Check every statement against the records.",
                              "Ignore demeanour; it is not evidence."],
        "scenarios": [_scenario_text(g) for g in groups],
        "hidden_truth": " and ".join(world["liars"]) + (" lied." if k == 2 else " lied."),
        "false_narrative": "",
        "requires_all": [],
        "teaches": [],
        "solution": {"unlock_key": "GENERATED"},
        "answer_key": {
            "true_scenario": sid[truth],
            "ruled_out": sorted((sid[frozenset(g)] for g in groups if frozenset(g) != truth), key=lambda x: int(x[1:])),
            "key_evidence": key,
            "red_herrings": [eid[id(f)] for f in facts if f.role == "herring"],
        },
        "generator": {"version": 2, "family": FAMILY, "level": level, "k": k, "n_places": n_places,
                      "witnesses": ws, "scenario_sets": [list(g) for g in groups],
                      "facts": [f.to_dict() for f in facts]},
    }


def generate(n: int, seed: int, levels: tuple[int, ...] = (2, 3), pool: str = "A",
             prefix: str = "M") -> list[dict[str, Any]]:
    """Half the cases have one liar and half two, so the rule has to be read."""
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 300:
            raise RuntimeError("multi_liar generator rejected too many worlds")
        level = levels[len(cases) % len(levels)]
        k = 1 + (len(cases) // len(levels)) % 2
        world = simulate(rng, level, pool, k)
        if world is None:
            continue
        case = build_case(world, f"{prefix}{seed:03d}-{len(cases) + 1:05d}", level, rng)
        if case is not None:
            cases.append(case)
    return cases
