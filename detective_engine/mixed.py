"""v11 family 2: alibis from mixed sources in one case.

Spot check S-1 showed the best model cannot yet combine skills it learned in
separate families: it read a CCTV-verified alibi inside a lying-witness case
without the person's name, and it invented paragraphs past the end of a case
shorter than any it had trained on. This family puts the pieces together:

  - the death window from body cooling, sometimes with an earlier sign of life
    beside the one that counts;
  - key holders, each cleared either by a witness ("I was with X at P from a to
    b") or by a verified record (CCTV, card logs) covering a span;
  - exactly one lying witness, exposed by a record of the witness or of the
    suspect elsewhere inside the claimed time;
  - traps: a record of the culprit at the claimed place *before* the claimed
    time (it looks like corroboration and tests nothing), a verified alibi for
    the culprit that covers only part of the window, a verified alibi that
    ends exactly when the window closes;
  - case length varied from about 12 to 30 paragraphs (three to five suspects,
    motives and demeanour for only some people).

The combined family's solver (composite.solve) already handles verified alibis,
so cases are verified the same way: exactly one culprit across consistent liars.
"""

from __future__ import annotations

import random
from typing import Any

from . import composite
from .generator import BODY_TEMP, COOLING_PER_HOUR, POOLS, Fact, death_window, fmt
from .liar import DEMEANOUR, PLACES, SOURCES

FAMILY = "mixed"

DIFFICULTY = {
    2: {"suspects": (3, 4), "mechanism": "witness_record", "margin": (10, 60), "herrings": (0, 1)},
    3: {"suspects": (4, 5), "mechanism": "suspect_record", "margin": (5, 40), "herrings": (1, 2)},
}


def simulate(rng: random.Random, level: int, pool_name: str) -> dict[str, Any] | None:
    cfg, pool = DIFFICULTY[level], POOLS[pool_name]
    n = rng.randint(*cfg["suspects"])
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
    ]
    if rng.random() < 0.4:                              # an earlier sign of life; the later one fixes the window
        t0 = last_alive - rng.randint(40, 150)
        facts.append(Fact("last_alive", f"A neighbour saw {victim} come home at {fmt(t0)}.", {"time": t0}, "neutral"))
    facts += [
        Fact("homicide", "The fatal injury is a blow to the back of the head that no fall in the room could explain.", {}, "hard"),
        Fact("no_forced_entry", f"There was no sign of forced entry; the {place} door had been locked and unlocked with a key.", {}, "hard"),
    ]
    lo, hi = death_window(facts)

    access_excluded = set(rng.sample(distractors, rng.randint(1, max(1, len(distractors) // 2))))
    keyholders = [s for s in suspects if s not in access_excluded]
    if len(keyholders) < 2:
        return None
    rng.shuffle(keyholders)
    no_key = [s for s in suspects if s in access_excluded]
    holders = ", ".join([victim] + keyholders[:-1]) + f" and {keyholders[-1]}"
    facts.append(Fact("keyholders", f"Keys to the {place} are held only by {holders}.", {"names": sorted(keyholders)}, "hard"))

    # How each key holder is cleared: the culprit by the lying witness; the others by a
    # truthful witness or by a verified record, with at least one truthful witness.
    others = [s for s in keyholders if s != culprit]
    source = {s: rng.choice(["witness", "verified"]) for s in others}
    if all(v == "verified" for v in source.values()):
        source[rng.choice(others)] = "witness"
    source[culprit] = "witness"
    witness_for = [s for s in keyholders if source[s] == "witness"]
    witnesses = people[n + 1:n + 1 + len(witness_for)]
    vouch = dict(zip(witness_for, witnesses))
    liar = vouch[culprit]
    lo_m, hi_m = cfg["margin"]
    claims: dict[str, Fact] = {}
    for s, w in vouch.items():
        a, b = lo - rng.randint(lo_m, hi_m), hi + rng.randint(lo_m, hi_m)
        p = rng.choice(places)
        claims[s] = Fact("witness_alibi", f'{w} says: "I was with {s} at {p} from {fmt(a)} to {fmt(b)}."',
                         {"witness": w, "suspect": s, "place": p, "start": a, "end": b}, "hard" if w == liar else "neutral")
    facts.extend(claims.values())

    def verified(s: str, a: int, b: int, role: str) -> Fact:
        src = rng.choice(pool["source"]).format(p=rng.choice(pool["alibi_place"]), s=s)
        return Fact("alibi", f"{src} from {fmt(a)} to {fmt(b)}.", {"name": s, "start": a, "end": b}, role)

    for s in others:
        if source[s] == "verified":
            a = lo - rng.randint(lo_m, hi_m)
            b = hi if rng.random() < 0.25 else hi + rng.randint(lo_m, hi_m)   # sometimes ends exactly at window close
            facts.append(verified(s, a, b, "hard"))
    if rng.random() < 0.35 and hi - lo >= 60:           # a verified alibi for the culprit that covers only part
        end = rng.randint(lo + 10, hi - 20)
        facts.append(verified(culprit, end - rng.randint(60, 150), end, "neutral"))

    def record(name: str, t: int, p: str, role: str = "neutral") -> Fact:
        return Fact("record", rng.choice(SOURCES[pool_name]).format(p=name, l=p, t=fmt(t)),
                    {"name": name, "time": t, "place": p}, role)

    lie = claims[culprit].data
    inside = rng.randint(lie["start"] + 5, lie["end"] - 5)
    elsewhere = rng.choice([p for p in places if p != lie["place"]])
    for s, w in vouch.items():
        c = claims[s].data
        if w == liar and cfg["mechanism"] == "witness_record":
            facts.append(record(w, inside, elsewhere, "hard"))
        else:
            facts.append(record(w, rng.randint(c["start"] + 5, c["end"] - 5), c["place"]))
    for s in suspects:
        if s == culprit and cfg["mechanism"] == "suspect_record":
            facts.append(record(s, inside, elsewhere, "hard"))
        elif s in claims and rng.random() < 0.7:
            c = claims[s].data
            facts.append(record(s, rng.randint(c["start"] + 5, c["end"] - 5), c["place"]))
        elif s in no_key and rng.random() < 0.6:
            facts.append(record(s, lo - rng.randint(60, 180), rng.choice(places)))
    if rng.random() < 0.5:                              # decoy: the culprit at the claimed place before the claim
        facts.append(record(culprit, lie["start"] - rng.randint(10, 50), lie["place"]))

    for s in no_key:
        facts.append(Fact("testimony", f"{s} says they were at {rng.choice(places)} all evening.", {"name": s}, "neutral"))
    for s in suspects:
        if rng.random() < 0.6:
            text = f"{s}, the {roles[s]}, " + rng.choice(pool["motive"]).format(v=victim) + "."
            facts.append(Fact("motive", text, {"name": s}, "neutral" if s == culprit else "herring"))
    for w in witnesses:
        if rng.random() < 0.5:
            facts.append(Fact("demeanour", rng.choice(DEMEANOUR).format(w=w), {"name": w}, "herring"))
    for text in rng.sample(pool["salient"], min(rng.randint(*cfg["herrings"]), len(pool["salient"]))):
        facts.append(Fact("salient", text.format(t=fmt(rng.randint(lo - 60, hi))), {}, "herring"))

    return {"victim": victim, "suspects": suspects, "witnesses": witnesses, "roles": roles, "place": place,
            "culprit": culprit, "liar": liar, "facts": facts, "t_death": t_death}


def build_case(world: dict[str, Any], case_id: str, level: int, rng: random.Random) -> dict[str, Any] | None:
    case = composite.build_case(world, case_id, level, rng)
    if case is None:
        return None
    case["title"] = f"Alibis in the {world['place']}"
    case["category"] = "generated-mixed-alibi"
    case["generator"]["family"] = FAMILY
    return case


def generate(n: int, seed: int, levels: tuple[int, ...] = (2, 3), pool: str = "A",
             prefix: str = "X") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 300:
            raise RuntimeError("mixed generator rejected too many worlds")
        level = levels[len(cases) % len(levels)]
        world = simulate(rng, level, pool)
        if world is None:
            continue
        case = build_case(world, f"{prefix}{seed:03d}-{len(cases) + 1:05d}", level, rng)
        if case is not None:
            cases.append(case)
    return cases
