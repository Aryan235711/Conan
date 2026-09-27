"""Second reasoning family: which witness is lying?

Several witnesses give statements about where they were and whom they saw.
Reliable records (card logs, CCTV) pin some people to places.  Exactly one
witness lies; finding them means checking every statement against the
records and against the other statements.

Generation follows the same recipe as the timeline family:

  1. Simulate a ground-truth world: each witness's location at each time.
  2. Write truthful statements from that world, and one false statement for
     the liar that a record contradicts, directly (level 1) or only through
     an "I saw X" claim about someone else (levels 2 and 3).
  3. Solve: a hypothesis "W is the liar" is consistent if the records plus
     everyone else's statements have a model, and at least one of W's
     statements can be false in it.  Keep the case only if exactly one
     witness is consistent.

Mention balance: every witness is named exactly five times (own place
claim, own sighting claim, being sighted once, one record, one demeanour
line), so counting names reveals nothing.
"""

from __future__ import annotations

import random
from typing import Any

from .generator import POOLS, Fact, fmt

FAMILY = "liar"

PLACES = {
    "A": ["the café", "the library", "the station", "the gym", "the bakery", "the pharmacy"],
    "B": ["the ferry terminal", "the museum", "the market hall", "the chess club", "the laundrette", "the observatory"],
}
SOURCES = {
    "A": ["Card logs show {p} at {l} at {t}.", "CCTV shows {p} at {l} at {t}.", "A timestamped receipt puts {p} at {l} at {t}."],
    "B": ["Ticket scans place {p} at {l} at {t}.", "Security footage shows {p} at {l} at {t}.", "A booking system records {p} at {l} at {t}."],
}
DEMEANOUR = [
    "{w} seemed nervous and avoided eye contact during the interview.",
    "{w} answered every question calmly and in great detail.",
    "{w} contradicted themselves about what they had for dinner.",
    "{w} was irritated at being questioned.",
    "{w} kept checking their phone while giving a statement.",
]
DIFFICULTY = {
    1: {"witnesses": 3, "slots": 4, "places": 4, "mechanism": "direct", "salient": 0},
    2: {"witnesses": 4, "slots": 5, "places": 5, "mechanism": "sighting", "salient": 1},
    3: {"witnesses": 5, "slots": 6, "places": 6, "mechanism": "sighting", "salient": 2},
}

Atom = tuple  # (person, time, place, positive)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def satisfiable(atoms: list[Atom], n_places: int) -> bool:
    pos: dict[tuple, set] = {}
    neg: dict[tuple, set] = {}
    for p, t, l, positive in atoms:
        (pos if positive else neg).setdefault((p, t), set()).add(l)
    for cell, ls in pos.items():
        if len(ls) > 1 or ls & neg.get(cell, set()):
            return False
    for cell, ls in neg.items():
        if cell not in pos and len(ls) >= n_places:
            return False
    return True


def _atoms(f: Fact) -> list[Atom]:
    return [tuple(a) for a in f.data.get("atoms", [])]


def solve(facts: list[Fact], witnesses: list[str], n_places: int) -> set[str]:
    records = [a for f in facts if f.kind == "record" for a in _atoms(f)]
    consistent = set()
    for h in witnesses:
        others = [a for f in facts if f.kind == "statement" and f.data["speaker"] != h for a in _atoms(f)]
        base = records + others
        if not satisfiable(base, n_places):
            continue
        own = [f for f in facts if f.kind == "statement" and f.data["speaker"] == h]
        # h is a liar only if at least one of h's statements can be false.
        can_lie = any(
            satisfiable(base + [(p, t, l, not positive)], n_places)
            for f in own for (p, t, l, positive) in _atoms(f)
        )
        if can_lie:
            consistent.add(h)
    return consistent


# ---------------------------------------------------------------------------
# World simulation
# ---------------------------------------------------------------------------

def simulate(rng: random.Random, level: int, pool_name: str) -> dict[str, Any] | None:
    cfg, pool = DIFFICULTY[level], POOLS[pool_name]
    n = cfg["witnesses"]
    firsts, lasts = rng.sample(pool["first"], n), rng.sample(pool["last"], n)
    ws = [f"{a} {b}" for a, b in zip(firsts, lasts)]
    places = rng.sample(PLACES[pool_name], cfg["places"])
    start = rng.randint(6, 8) * 60                     # 18:00 .. 20:00 in minutes since noon
    times = [start + 60 * i for i in range(cfg["slots"])]
    world = {(w, t): rng.choice(places) for w in ws for t in times}
    liar = rng.choice(ws)

    # Derangement: each witness claims to have seen exactly one other, and is seen exactly once.
    while True:
        seen = ws[:]
        rng.shuffle(seen)
        if all(a != b for a, b in zip(ws, seen)):
            break
    sees = dict(zip(ws, seen))

    # Distinct slot per sighting so forced co-locations never collide.
    sight_t = dict(zip(ws, rng.sample(times, n)))
    for w in ws:
        if w != liar:
            world[(sees[w], sight_t[w])] = world[(w, sight_t[w])]

    facts: list[Fact] = [Fact("rule", "Exactly one witness is lying; the others tell the truth. Records are reliable.", {}, "neutral")]
    rec_cell: dict[str, int] = {}

    # Liar's false statement and the record that exposes it.
    if cfg["mechanism"] == "direct":
        t = rng.choice([x for x in times if x != sight_t[liar]])
        true_l = world[(liar, t)]
        fake = rng.choice([l for l in places if l != true_l])
        lie = Fact("statement", f'{liar} says: "I was at {fake} at {fmt(t)}."',
                   {"speaker": liar, "atoms": [[liar, t, fake, True]]}, "hard")
        rec_cell[liar] = t
        self_t = {liar: t}
        sight_lie = False
    else:
        q, t = sees[liar], sight_t[liar]
        world[(liar, t)] = rng.choice(places)
        true_q = world[(q, t)]
        fake = rng.choice([l for l in places if l != true_q])
        world[(liar, t)] = fake                        # the liar really was there; q was not
        lie = Fact("statement", f'{liar} says: "I saw {q} at {fake} at {fmt(t)}."',
                   {"speaker": liar, "atoms": [[liar, t, fake, True], [q, t, fake, True]]}, "hard")
        rec_cell[q] = t
        self_t = {}
        sight_lie = True
    facts.append(lie)

    # Everyone else's sightings are true.
    for w in ws:
        if w == liar and sight_lie:
            continue
        t = sight_t[w]
        l = world[(w, t)]
        if w == liar:                                  # direct mechanism: liar's sighting is true
            world[(sees[w], t)] = l
        facts.append(Fact("statement", f'{w} says: "I saw {sees[w]} at {l} at {fmt(t)}."',
                          {"speaker": w, "atoms": [[w, t, l, True], [sees[w], t, l, True]]}, "neutral"))

    # Place claims (true), one per witness; the liar's is the lie in the direct mechanism.
    for w in ws:
        if w in self_t:
            continue
        t = rng.choice([x for x in times if x != sight_t[w]])
        facts.append(Fact("statement", f'{w} says: "I was at {world[(w, t)]} at {fmt(t)}."',
                          {"speaker": w, "atoms": [[w, t, world[(w, t)], True]]}, "neutral"))

    # One reliable record per person.
    for w in ws:
        t = rec_cell.get(w, rng.choice(times))
        text = rng.choice(SOURCES[pool_name]).format(p=w, l=world[(w, t)], t=fmt(t))
        facts.append(Fact("record", text, {"atoms": [[w, t, world[(w, t)], True]]},
                          "hard" if w in rec_cell else "neutral"))

    # Demeanour: one per witness, never evidence.
    for w in ws:
        facts.append(Fact("demeanour", rng.choice(DEMEANOUR).format(w=w), {"name": w}, "herring"))
    for text in rng.sample(pool["salient"], cfg["salient"]):
        facts.append(Fact("salient", text.format(t=fmt(rng.choice(times))), {}, "herring"))

    return {"witnesses": ws, "liar": liar, "facts": facts, "n_places": len(places)}


def build_case(world: dict[str, Any], case_id: str, level: int, rng: random.Random) -> dict[str, Any] | None:
    facts, ws, liar, n_places = world["facts"], world["witnesses"], world["liar"], world["n_places"]
    if solve(facts, ws, n_places) != {liar}:
        return None
    for f in facts:
        if f.role == "herring" and solve([g for g in facts if g is not f], ws, n_places) != {liar}:
            return None

    rule = [f for f in facts if f.kind == "rule"]
    body = [f for f in facts if f.kind != "rule"]
    rng.shuffle(body)
    facts = rule + body
    labels = ws[:]
    rng.shuffle(labels)
    sid = {w: f"S{i}" for i, w in enumerate(labels, 1)}
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}

    key = [eid[id(f)] for f in facts
           if f.kind != "rule" and solve([g for g in facts if g is not f], ws, n_places) != {liar}]
    if not key:
        return None
    answer_key = {
        "true_scenario": sid[liar],
        "ruled_out": sorted((sid[w] for w in ws if w != liar), key=lambda x: int(x[1:])),
        "key_evidence": key,
        "red_herrings": [eid[id(f)] for f in facts if f.role == "herring"],
    }
    return {
        "id": case_id,
        "title": "Who is lying?",
        "category": "generated-witness-consistency",
        "summary": (f"{len(ws)} witnesses gave statements about one evening. Exactly one of them is lying. "
                    "Work out who, using the records and the other statements."),
        "evidence": [f.text for f in facts],
        "detective_questions": [
            "Which statements can be checked directly against a record?",
            "What does each 'I saw' statement imply about both people involved?",
            "If a given witness were the liar, would everyone else's statements still fit the records?",
            "Which details describe behaviour rather than facts?",
        ],
        "analysis_protocol": ["Treat records as fixed.", "Test each witness as the liar in turn.",
                              "Ignore demeanour; it is not evidence."],
        "scenarios": [f"{w} is lying." for w in labels],
        "hidden_truth": f"{liar} lied: " + " ".join(
            f.text for f in facts if f.role == "hard" and f.kind in ("statement", "record")),
        "false_narrative": "",
        "requires_all": [],
        "teaches": [],
        "solution": {"unlock_key": "GENERATED"},
        "answer_key": answer_key,
        "generator": {"version": 2, "family": FAMILY, "level": level, "n_places": n_places,
                      "facts": [f.to_dict() for f in facts]},
    }


def generate(n: int, seed: int, levels: tuple[int, ...] = (1, 2, 3), pool: str = "A",
             prefix: str = "L") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError("liar generator rejected too many worlds")
        level = levels[len(cases) % len(levels)]
        world = simulate(rng, level, pool)
        if world is None:
            continue
        case = build_case(world, f"{prefix}{seed:03d}-{len(cases) + 1:05d}", level, rng)
        if case is not None:
            cases.append(case)
    return cases
