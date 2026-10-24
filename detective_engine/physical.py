"""v11 family 4: physical evidence that changes the reasoning (stomach contents, lividity).

Two forensic clues, each stated with its rule inside the case so it can be
learned and checked:

  - Stomach contents date the death from the last meal. Undigested food means
    death within one hour of the meal; partly digested means one to three hours
    after it; an empty stomach means four hours or more. The digestion window is
    intersected with the body-cooling window, so an alibi that does not cover the
    cooling window can still cover the narrower one and clear its holder.
  - Lividity shows whether the body was moved. If it does not match how the
    body lay, the death happened elsewhere, so the key list of the room where the
    body was found no longer limits who could have done it; a suspect with no
    key can be the culprit.

Traps: ignoring the stomach leaves a cleared suspect in play; applying the key
rule after the body was moved wrongly excludes the culprit; lividity that does
match changes nothing; an unverified account clears nobody.
"""

from __future__ import annotations

import random
from typing import Any

from .generator import BODY_TEMP, COOLING_PER_HOUR, POOLS, Fact, death_window, fmt

FAMILY = "physical"

RULE = ("Medical rules for this case: food stays undigested for about an hour after a meal, is partly digested "
        "from one to three hours after it, and has left the stomach after four hours. Lividity settles where the "
        "body lies; lividity that does not match how the body lay means it was moved after death, so the death "
        "happened somewhere else.")
STOMACH = {"undigested": (0, 60), "partly digested": (60, 180), "empty": (240, 24 * 60)}
MEALS = {"A": ["the trattoria", "the noodle bar", "the hotel restaurant", "a friend's kitchen"],
         "B": ["the brasserie", "the dumpling house", "the club dining room", "the canteen"]}
DIFFICULTY = {
    2: {"suspects": 4, "herrings": (1, 2), "margin": (10, 40)},
    3: {"suspects": 5, "herrings": (1, 2), "margin": (5, 30)},
}


def window(facts: list[Fact]) -> tuple[int, int]:
    """Death window from cooling and signs of life, narrowed by the stomach contents if known."""
    lo, hi = death_window(facts)
    meal = next((f.data["time"] for f in facts if f.kind == "meal"), None)
    state = next((f.data["state"] for f in facts if f.kind == "stomach"), None)
    if meal is not None and state in STOMACH:
        a, b = STOMACH[state]
        lo, hi = max(lo, meal + a), min(hi, meal + b)
    return lo, hi


def solve(facts: list[Fact], suspects: list[str], with_accident: bool) -> set[str]:
    lo, hi = window(facts)
    kinds = {f.kind for f in facts}
    moved = any(f.kind == "lividity" and f.data["moved"] for f in facts)
    holders = next((set(f.data["names"]) for f in facts if f.kind == "keyholders"), None)
    access = "no_forced_entry" in kinds and holders is not None and not moved
    cleared = {f.data["name"] for f in facts if f.kind == "alibi" and f.data["start"] <= lo and f.data["end"] >= hi}
    out = {s for s in suspects if not (access and s not in holders) and s not in cleared}
    if with_accident and "homicide" not in kinds:
        out.add("accident")
    return out


def simulate(rng: random.Random, level: int, pool_name: str) -> dict[str, Any] | None:
    cfg, pool = DIFFICULTY[level], POOLS[pool_name]
    n = cfg["suspects"]
    people = [f"{a} {b}" for a, b in zip(rng.sample(pool["first"], n + 1), rng.sample(pool["last"], n + 1))]
    victim, suspects = people[0], people[1:]
    roles = dict(zip(suspects, rng.sample(pool["role"], n)))
    place = rng.choice(pool["place"])
    use_stomach = rng.random() < 0.75
    moved = rng.random() < 0.4
    use_lividity = moved or rng.random() < 0.5

    t_death = rng.randint(9 * 60, 13 * 60 + 30)
    discovery = t_death + rng.randint(7 * 60, 10 * 60)
    temp = round((BODY_TEMP - COOLING_PER_HOUR * (discovery - t_death) / 60) * 2) / 2
    last_alive = t_death - rng.randint(40, 100)
    facts: list[Fact] = [
        Fact("physical_rule", RULE, {}, "neutral"),
        Fact("discovery", f"{victim} was found dead in the {place} at {fmt(discovery)}.", {"time": discovery}, "neutral"),
        Fact("body_temp",
             f"At {fmt(discovery)} the body temperature was {temp:.1f}°C. In that room a body cools by about "
             f"{COOLING_PER_HOUR:.0f}°C per hour from {BODY_TEMP:.0f}°C, and the estimate is reliable to within about an hour.",
             {"temp": temp, "discovery": discovery}, "hard"),
        Fact("last_alive", f"{victim}'s phone sent a text message at {fmt(last_alive)}.", {"time": last_alive}, "hard"),
        Fact("homicide", "The fatal injury is a blow to the back of the head that no fall could explain.", {}, "hard"),
        Fact("no_forced_entry", f"There was no sign of forced entry; the {place} door had been locked and unlocked with a key.", {}, "hard"),
    ]
    cool_lo, cool_hi = death_window(facts)
    if use_stomach:
        state = rng.choice(["undigested", "partly digested", "empty"])
        a, b = STOMACH[state]
        meal = t_death - rng.randint(a + 5, min(b, a + 170) - 5)
        meal -= meal % 5
        if not (meal + a <= t_death <= meal + b):
            return None
        facts.append(Fact("meal", f"{victim} ate dinner at {rng.choice(MEALS[pool_name])} at {fmt(meal)}; the others there remember.",
                          {"time": meal}, "hard"))
        facts.append(Fact("stomach", f"The post-mortem found the stomach contents {state}.", {"state": state}, "hard"))
    lo, hi = window(facts)
    if hi - lo < 25 or not lo <= t_death <= hi:
        return None
    if use_stomach and not (lo >= cool_lo + 15 or hi <= cool_hi - 15):
        return None                                    # keep the stomach only when it narrows the window

    no_key = set(rng.sample(suspects, rng.randint(1, max(1, n // 2))))
    if moved and rng.random() < 0.6:
        culprit = rng.choice(sorted(no_key))           # the key rule would wrongly exclude them
    else:
        culprit = rng.choice([s for s in suspects if s not in no_key])
    keyholders = [s for s in suspects if s not in no_key]
    rng.shuffle(keyholders)
    holders = ", ".join([victim] + keyholders[:-1]) + f" and {keyholders[-1]}"
    facts.append(Fact("keyholders", f"Keys to the {place} are held only by {holders}.", {"names": sorted(keyholders)}, "hard"))
    if use_lividity:
        facts.append(Fact("lividity",
                          "Lividity is on the victim's back, but the body was found lying face down." if moved else
                          "Lividity is on the victim's back, matching how the body lay.",
                          {"moved": moved}, "hard" if moved else "neutral"))

    lo_m, hi_m = cfg["margin"]
    trap_done = False
    for s in suspects:
        p = rng.choice(pool["alibi_place"])
        needs_clearing = s != culprit and (moved or s not in no_key)
        if needs_clearing:
            if use_stomach and not trap_done and (lo > cool_lo + 15 or hi < cool_hi - 15):
                # covers the narrowed window but not the cooling window: only the stomach clears them
                a, b = lo - rng.randint(5, 12), hi + rng.randint(5, 12)
                if a > cool_lo or b < cool_hi:
                    trap_done = True
            else:
                a, b = lo - rng.randint(lo_m, hi_m), hi + rng.randint(lo_m, hi_m)
            facts.append(Fact("alibi", f"CCTV at {p} shows {s} there from {fmt(a)} to {fmt(b)}.",
                              {"name": s, "start": a, "end": b}, "hard"))
        elif s == culprit and rng.random() < 0.5 and hi - lo >= 30:
            end = rng.randint(lo + 5, hi - 10)
            start = end - rng.randint(60, 150)
            facts.append(Fact("alibi", f"CCTV at {p} shows {s} there from {fmt(start)} to {fmt(end)}.",
                              {"name": s, "start": start, "end": end}, "neutral"))
        else:
            facts.append(Fact("testimony", f"{s} says they were at home all evening.", {"name": s}, "neutral"))
    for s in suspects:
        if rng.random() < 0.7:
            text = f"{s}, the {roles[s]}, " + rng.choice(pool["motive"]).format(v=victim) + "."
            facts.append(Fact("motive", text, {"name": s}, "neutral" if s == culprit else "herring"))
    for text in rng.sample(pool["salient"], min(rng.randint(*cfg["herrings"]), len(pool["salient"]))):
        facts.append(Fact("salient", text.format(t=fmt(rng.randint(lo - 60, hi))), {}, "herring"))
    return {"victim": victim, "suspects": suspects, "roles": roles, "place": place, "culprit": culprit,
            "facts": facts, "t_death": t_death, "accident": True}


def build_case(world: dict[str, Any], case_id: str, level: int, rng: random.Random) -> dict[str, Any] | None:
    facts, suspects, culprit = world["facts"], world["suspects"], world["culprit"]

    def sol(fs: list[Fact]) -> set[str]:
        return solve(fs, suspects, True)

    if sol(facts) != {culprit}:
        return None
    for f in facts:
        if f.role == "herring" and sol([g for g in facts if g is not f]) != {culprit}:
            return None
    lo, hi = window(facts)
    if not lo <= world["t_death"] <= hi:
        return None
    rule = [f for f in facts if f.kind == "physical_rule"]
    body = [f for f in facts if f.kind != "physical_rule"]
    rng.shuffle(body)
    facts = rule + body
    labels = suspects + ["accident"]
    rng.shuffle(labels)
    text = {s: f"{s}, the {world['roles'][s]}, killed {world['victim']}." for s in suspects}
    text["accident"] = f"{world['victim']}'s death was an accident."
    sid = {lab: f"S{i}" for i, lab in enumerate(labels, 1)}
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}
    key = [eid[id(f)] for f in facts if f.kind != "physical_rule" and sol([g for g in facts if g is not f]) != {culprit}]
    if not key:
        return None
    return {
        "id": case_id,
        "title": f"What the body showed in the {world['place']}",
        "category": "generated-physical-evidence",
        "summary": (f"{world['victim']} was found dead in the {world['place']}. {len(suspects)} people are under "
                    "suspicion. The post-mortem matters. Work out who could have done it and when."),
        "evidence": [f.text for f in facts],
        "detective_questions": [
            "What window do body cooling and the last sign of life allow?",
            "Does the stomach narrow that window?",
            "Was the body moved, and does the room's key list still matter?",
            "Which verified alibis cover the whole window?",
        ],
        "analysis_protocol": ["Fix the window from all physical evidence.", "Decide whether access applies.",
                              "Check every verified alibi against the window."],
        "scenarios": [text[l] for l in labels],
        "hidden_truth": f"{culprit} killed {world['victim']} between {fmt(lo)} and {fmt(hi)}.",
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
        "generator": {"version": 2, "family": FAMILY, "level": level, "facts": [f.to_dict() for f in facts]},
    }


def generate(n: int, seed: int, levels: tuple[int, ...] = (2, 3), pool: str = "A",
             prefix: str = "F") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 400:
            raise RuntimeError("physical generator rejected too many worlds")
        level = levels[len(cases) % len(levels)]
        world = simulate(rng, level, pool)
        if world is None:
            continue
        case = build_case(world, f"{prefix}{seed:03d}-{len(cases) + 1:05d}", level, rng)
        if case is not None:
            cases.append(case)
    return cases
