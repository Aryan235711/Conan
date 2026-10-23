"""v11 family 3: travel alibis (tickets, timetables and travel times).

A verified alibi so far was a span at one place: "CCTV shows X at the gym from
a to b". Real alibis are often journeys. Here a key holder's alibi can be a
ticket for a train; the timetable gives its arrival; travel times link the
stations to the scene. Someone on the 22:40 from Northgate (20 minutes from
the scene), arriving at Eastbay at 23:55 (50 minutes from the scene), cannot
have been at the scene from 22:20 to 00:45: they had to leave by 22:20 and could
not be back before 00:45. Clearing a suspect therefore takes three steps the
other families never needed:

  1. look up the ticket's service in a timetable with decoy services;
  2. time arithmetic across hours and midnight (departure minus the walk to the
     station, arrival plus the trip back);
  3. the usual coverage check of that extended span against the death window.

Traps: the train times alone often do not cover the window while the extended
span does (and the reverse for the culprit, whose return is fast enough); decoy
timetable rows from the same station or at the same time; a journey that is only
claimed, with no ticket; a verified stay that covers only part of the window.

Travel times are stated as the fastest possible, so the solver treats a ticket
holder as away from the scene for exactly [departure - walk out, arrival + trip back].
"""

from __future__ import annotations

import random
from typing import Any

from .generator import BODY_TEMP, COOLING_PER_HOUR, POOLS, Fact, death_window, fmt

FAMILY = "travel"

STATIONS = {
    "A": ["Northgate", "Eastbay", "Millbrook", "Kingsferry", "Ashworth", "Riverside", "Hollowmere", "Stonebridge"],
    "B": ["Larkspur", "Westholm", "Granby", "Fellingham", "Oakhurst", "Brackwell", "Thornbury"],
}
DIFFICULTY = {
    2: {"suspects": 4, "herrings": (1, 2), "margin": (10, 40), "decoys": (1, 1)},
    3: {"suspects": 5, "herrings": (1, 2), "margin": (5, 30), "decoys": (1, 2)},
}
RULE = "Travel times below are the fastest possible. Tickets, timetables and records are reliable."


def spans(facts: list[Fact]) -> list[tuple[str, int, int]]:
    """Every (person, start, end) during which someone verifiably could not be at the scene."""
    table = {(f.data["station"], f.data["dep"]): (f.data["dest"], f.data["arr"]) for f in facts if f.kind == "timetable"}
    walk = {f.data["station"]: f.data["minutes"] for f in facts if f.kind == "travel"}
    out = [(f.data["name"], f.data["start"], f.data["end"]) for f in facts if f.kind == "alibi"]
    for f in facts:
        if f.kind == "ticket":
            key = (f.data["station"], f.data["dep"])
            if key in table and f.data["station"] in walk and table[key][0] in walk:
                dest, arr = table[key]
                out.append((f.data["name"], f.data["dep"] - walk[f.data["station"]], arr + walk[dest]))
    return out


def solve(facts: list[Fact], suspects: list[str], with_accident: bool) -> set[str]:
    lo, hi = death_window(facts)
    kinds = {f.kind for f in facts}
    holders = next((set(f.data["names"]) for f in facts if f.kind == "keyholders"), None)
    access = "no_forced_entry" in kinds and holders is not None
    cleared = {p for p, a, b in spans(facts) if a <= lo and b >= hi}
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
    culprit = rng.choice(suspects)
    distractors = [s for s in suspects if s != culprit]
    stations = rng.sample(STATIONS[pool_name], 6)
    walk = {s: rng.choice(range(10, 75, 5)) for s in stations}

    t_death = rng.randint(9 * 60, 13 * 60 + 30)
    discovery = t_death + rng.randint(7 * 60, 10 * 60)
    temp = round((BODY_TEMP - COOLING_PER_HOUR * (discovery - t_death) / 60) * 2) / 2
    last_alive = t_death - rng.randint(10, 50)
    facts: list[Fact] = [
        Fact("travel_rule", RULE, {}, "neutral"),
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
    holders = ", ".join([victim] + keyholders[:-1]) + f" and {keyholders[-1]}"
    facts.append(Fact("keyholders", f"Keys to the {place} are held only by {holders}.", {"names": sorted(keyholders)}, "hard"))

    used: set[str] = set()
    timetable: list[tuple[str, int, str, int]] = []

    def journey(s: str, cover: bool, role: str) -> bool:
        """Add a ticket (and its timetable row) whose extended span covers the window or, for the
        culprit, falls short at one end. Returns False if no workable pair of stations was found."""
        lo_m, hi_m = cfg["margin"]
        for _ in range(40):
            s1, s2 = rng.sample(stations, 2)
            m1, m2 = rng.randint(lo_m, hi_m), rng.randint(lo_m, hi_m)
            if cover:
                dep, arr = lo - m1 + walk[s1], hi + m2 - walk[s2]
            elif rng.random() < 0.5:                     # back at the scene before the window closes
                dep, arr = lo - m1 + walk[s1], hi - m2 - walk[s2]
            else:                                        # could still have been at the scene when it opened
                dep, arr = lo + m1 + walk[s1], hi + m2 - walk[s2]
            dep, arr = dep - dep % 5, arr - arr % 5         # timetables run on five-minute marks
            a, b = dep - walk[s1], arr + walk[s2]
            if any((s1, dep) == (x[0], x[1]) for x in timetable):   # one row per station and departure
                continue
            if arr - dep < 20 or (cover and not (a <= lo - 5 and b >= hi + 5)) or \
                    (not cover and a <= lo + 5 and b >= hi - 5):
                continue
            if True:
                facts.append(Fact("ticket", f"{s}'s ticket was scanned on the {fmt(dep)} train from {s1}.",
                                  {"name": s, "station": s1, "dep": dep}, role))
                timetable.append((s1, dep, s2, arr))
                used.update((s1, s2))
                return True
        return False

    for s in keyholders:
        if s == culprit:
            pick = rng.choice(["journey", "journey", "claim", "partial"])
            if pick == "journey" and journey(s, cover=False, role="neutral"):
                continue
            if pick == "partial" and hi - lo >= 60:
                end = rng.randint(lo + 10, hi - 20)
                start = end - rng.randint(60, 150)
                p = rng.choice(pool["alibi_place"])
                facts.append(Fact("alibi", f"CCTV at {p} shows {s} there from {fmt(start)} to {fmt(end)}.",
                                  {"name": s, "start": start, "end": end}, "neutral"))
                continue
            st, at = rng.choice(stations), lo - rng.randint(10, 40)
            at -= at % 5
            facts.append(Fact("testimony", f"{s} says they took the {fmt(at)} train from {st}, "
                                           "but no ticket or record shows it.", {"name": s, "via": st, "at": at}, "neutral"))
        elif rng.random() < 0.75:
            if not journey(s, cover=True, role="hard"):
                return None
        else:
            lo_m, hi_m = cfg["margin"]
            a, b = lo - rng.randint(lo_m, hi_m), hi + rng.randint(lo_m, hi_m)
            p = rng.choice(pool["alibi_place"])
            facts.append(Fact("alibi", f"CCTV at {p} shows {s} there from {fmt(a)} to {fmt(b)}.",
                              {"name": s, "start": a, "end": b}, "hard"))
    for s in suspects:
        if s in access_excluded:
            facts.append(Fact("testimony", f"{s} says they were at home all evening.", {"name": s}, "neutral"))

    # Decoy timetable rows: the same station at another time, or the same time from another station.
    for s1, dep, s2, arr in list(timetable):
        for _ in range(rng.randint(*cfg["decoys"])):
            if rng.random() < 0.5:
                d = dep + rng.choice([-1, 1]) * rng.choice(range(15, 95, 5))
                decoy = (s1, d, rng.choice([x for x in stations if x != s1]), d + rng.choice(range(25, 120, 5)))
            else:
                other = rng.choice([x for x in stations if x != s1])
                decoy = (other, dep, rng.choice([x for x in stations if x != other]), dep + rng.choice(range(25, 120, 5)))
            if all((decoy[0], decoy[1]) != (t[0], t[1]) for t in timetable):
                timetable.append(decoy)
                used.update((decoy[0], decoy[2]))
    for s1, dep, s2, arr in timetable:
        facts.append(Fact("timetable", f"The {fmt(dep)} train from {s1} arrives at {s2} at {fmt(arr)}.",
                          {"station": s1, "dep": dep, "dest": s2, "arr": arr}, "neutral"))
    for st in sorted(used):
        facts.append(Fact("travel", f"{st} station is {walk[st]} minutes from the {place}.",
                          {"station": st, "minutes": walk[st]}, "neutral"))

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
    lo, hi = death_window(facts)
    if not lo <= world["t_death"] <= hi:
        return None
    rule = [f for f in facts if f.kind == "travel_rule"]
    body = [f for f in facts if f.kind != "travel_rule"]
    rng.shuffle(body)
    facts = rule + body
    labels = suspects + ["accident"]
    rng.shuffle(labels)
    text = {s: f"{s}, the {world['roles'][s]}, killed {world['victim']}." for s in suspects}
    text["accident"] = f"{world['victim']}'s death was an accident."
    sid = {lab: f"S{i}" for i, lab in enumerate(labels, 1)}
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}
    key = [eid[id(f)] for f in facts if f.kind != "travel_rule" and sol([g for g in facts if g is not f]) != {culprit}]
    if not key:
        return None
    return {
        "id": case_id,
        "title": f"The journey from the {world['place']}",
        "category": "generated-travel-alibi",
        "summary": (f"{world['victim']} was found dead in the {world['place']}. {len(suspects)} people are under "
                    "suspicion, and some say they were travelling. Work out who could have done it and when."),
        "evidence": [f.text for f in facts],
        "detective_questions": [
            "What window of time does the physical evidence allow for the death?",
            "For each ticket, which train was it, and when does that train arrive?",
            "Counting the trip to and from the stations, when could each traveller have been at the scene?",
            "Who could have entered the room?",
        ],
        "analysis_protocol": ["Fix the time window first.", "Turn each ticket into the span its holder was away.",
                              "Apply access, then check every span against the window."],
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
             prefix: str = "J") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 300:
            raise RuntimeError("travel generator rejected too many worlds")
        level = levels[len(cases) % len(levels)]
        world = simulate(rng, level, pool)
        if world is None:
            continue
        case = build_case(world, f"{prefix}{seed:03d}-{len(cases) + 1:05d}", level, rng)
        if case is not None:
            cases.append(case)
    return cases
