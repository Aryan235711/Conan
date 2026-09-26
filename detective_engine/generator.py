"""Case generator v2 — simulate a world, derive evidence, verify with a solver.

Template generators fill blanks in a fixed story, so every case of a template
has the same logic and a model can memorize it.  This generator instead:

  1. Simulates a ground-truth world: a victim, suspects, a true time of
     death, who holds keys, and where each suspect verifiably was.
  2. Writes evidence from that world as structured facts plus text
     (body cooling, last sign of life, lock state, alibi records,
     testimony, motives, and deliberate red herrings).
  3. Runs a solver over the facts and keeps the case only if exactly one
     scenario survives.  The answer key is computed, not hand-written:
       - ruled_out    every scenario the solver excludes
       - key_evidence facts whose removal makes the answer ambiguous
       - red_herrings generated distractors, verified to change nothing
       - time_window  the death window implied by the evidence

Difficulty controls suspects, red herrings, an accident scenario, a
partial-alibi trap for the culprit, and how tight alibi margins are.

CLI:
    python3 -m detective_engine.generator --out data/generated
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .engine.models import CaseDefinition

# ---------------------------------------------------------------------------
# Vocabulary pools.  Pool "B" is held out entirely for the OOD test split.
# ---------------------------------------------------------------------------

POOLS: dict[str, dict[str, list[str]]] = {
    "A": {
        "first": ["Ava", "Ben", "Chloe", "Daniel", "Elif", "Farah", "Gabriel", "Hana", "Ivan", "Jonas",
                  "Kavya", "Leo", "Maya", "Nikhil", "Olga", "Pedro", "Quinn", "Rosa", "Sami", "Tara"],
        "last": ["Adler", "Brandt", "Costa", "Dube", "Evans", "Fischer", "Garcia", "Hughes", "Iyer", "Jensen",
                 "Kowalski", "Larsen", "Moreau", "Nakamura", "Okafor", "Petrov"],
        "role": ["business partner", "younger brother", "neighbour", "tenant", "former assistant",
                 "ex-spouse", "accountant", "cousin", "landlord", "old friend"],
        "place": ["study", "workshop", "flat", "office", "studio"],
        "alibi_place": ["a restaurant", "a cinema", "the gym", "a train", "a hospital shift", "a friend's party",
                        "the office", "a hotel bar"],
        "source": ["CCTV at {p} shows {s} there", "Card payments put {s} at {p}", "Staff at {p} confirm {s} was there",
                   "{p}'s entry logs show {s} there"],
        "motive": ["had argued loudly with {v} about money the week before",
                   "stood to inherit a share of {v}'s savings",
                   "had been threatened with a lawsuit by {v}",
                   "was overheard saying {v} would regret the new will"],
        "salient": ["A muddy footprint was found in the garden; the gardener confirms it is his own.",
                    "A hallway window was unlatched; the caretaker says it has not closed properly for months.",
                    "A dog barked around {t}; neighbours say it barks at the same time every night.",
                    "An unfamiliar car was parked outside; it belongs to a visitor of the house next door.",
                    "A torn letter was found in the bin; it is a routine notice from the bank."],
    },
    "B": {
        "first": ["Anouk", "Bruno", "Carmen", "Dmitri", "Esme", "Femi", "Greta", "Hugo", "Ines", "Jae",
                  "Kofi", "Lina", "Mateo", "Nadia", "Oskar", "Priya"],
        "last": ["Albescu", "Bergström", "Castillo", "Delacroix", "Eze", "Fontaine", "Gupta", "Halvorsen",
                 "Ishikawa", "Janssen", "Kaur", "Lindqvist"],
        "role": ["stepdaughter", "co-founder", "lodger", "gardener", "estate lawyer", "rival collector",
                 "nephew", "housekeeper"],
        "place": ["library", "greenhouse", "boathouse", "gallery"],
        "alibi_place": ["a concert hall", "the airport", "a night class", "a pharmacy", "a chess club", "a late ferry"],
        "source": ["Ticket scans at {p} place {s} there", "A timestamped receipt from {p} puts {s} there",
                   "Security footage from {p} shows {s} there"],
        "motive": ["owed {v} a large sum that was due that month",
                   "had been cut out of a business deal by {v}",
                   "was in a long dispute with {v} over a property boundary"],
        "salient": ["Broken glass lay on the path; a delivery van had dropped a crate there that afternoon.",
                    "The victim's phone showed a missed call from an unknown number; it was a courier's automated line.",
                    "Scratches marked the back door; the owner says the cat does it every night."],
    },
}

# partial_trap: chance that a partial-coverage alibi is among the non-clearing whereabouts options.
DIFFICULTY: dict[int, dict[str, Any]] = {
    1: {"suspects": 3, "herrings": (1, 1), "accident": False, "partial_trap": 0.0, "margin": (40, 90)},
    2: {"suspects": 4, "herrings": (1, 2), "accident": True, "partial_trap": 0.5, "margin": (10, 60)},
    3: {"suspects": 5, "herrings": (2, 3), "accident": True, "partial_trap": 0.5, "margin": (5, 40)},
}

BODY_TEMP = 37.0
COOLING_PER_HOUR = 1.0
ESTIMATE_TOLERANCE = 60   # minutes


def fmt(m: int) -> str:
    """Minutes since noon -> HH:MM."""
    total = (m + 12 * 60) % (24 * 60)
    return f"{total // 60:02d}:{total % 60:02d}"


# ---------------------------------------------------------------------------
# Facts and solver
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    kind: str                  # discovery | body_temp | last_alive | homicide | no_forced_entry |
                               # keyholders | alibi | testimony | motive | salient
    text: str
    data: dict[str, Any]
    role: str = "neutral"      # hard | neutral | herring

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "text": self.text, "data": self.data, "role": self.role}


def death_window(facts: list[Fact]) -> tuple[int, int]:
    lo, hi = 0, 24 * 60 - 1
    for f in facts:
        if f.kind == "discovery":
            hi = min(hi, f.data["time"])
        elif f.kind == "last_alive":
            lo = max(lo, f.data["time"])
        elif f.kind == "body_temp":
            elapsed = (BODY_TEMP - f.data["temp"]) / COOLING_PER_HOUR * 60
            est = round(f.data["discovery"] - elapsed)
            lo, hi = max(lo, est - ESTIMATE_TOLERANCE), min(hi, est + ESTIMATE_TOLERANCE)
    return lo, hi


def solve(facts: list[Fact], suspects: list[str], with_accident: bool) -> set[str]:
    """Return the set of scenario labels consistent with the hard facts.

    Labels are suspect names plus "accident".  Testimony, motives and
    red herrings are never constraints.
    """
    lo, hi = death_window(facts)
    kinds = {f.kind for f in facts}
    keyholders: set[str] | None = None
    for f in facts:
        if f.kind == "keyholders":
            keyholders = set(f.data["names"])
    access_matters = "no_forced_entry" in kinds and keyholders is not None

    consistent: set[str] = set()
    for s in suspects:
        if access_matters and s not in keyholders:
            continue
        covered = any(
            f.kind == "alibi" and f.data["name"] == s and f.data["start"] <= lo and f.data["end"] >= hi
            for f in facts
        )
        if not covered:
            consistent.add(s)
    if with_accident and "homicide" not in kinds:
        consistent.add("accident")
    return consistent


# ---------------------------------------------------------------------------
# World simulation
# ---------------------------------------------------------------------------

def _unique_people(rng: random.Random, pool: dict[str, list[str]], n: int) -> list[str]:
    firsts = rng.sample(pool["first"], n)
    lasts = rng.sample(pool["last"], n)
    return [f"{a} {b}" for a, b in zip(firsts, lasts)]


def simulate(rng: random.Random, level: int, pool_name: str) -> dict[str, Any] | None:
    cfg, pool = DIFFICULTY[level], POOLS[pool_name]
    people = _unique_people(rng, pool, cfg["suspects"] + 1)
    victim, suspects = people[0], people[1:]
    roles = dict(zip(suspects, rng.sample(pool["role"], len(suspects))))
    place = rng.choice(pool["place"])
    culprit = rng.choice(suspects)
    distractors = [s for s in suspects if s != culprit]

    t_death = rng.randint(9 * 60, 13 * 60 + 30)              # 21:00 .. 01:30
    discovery = t_death + rng.randint(7 * 60, 10 * 60)
    elapsed_h = (discovery - t_death) / 60
    temp = round((BODY_TEMP - COOLING_PER_HOUR * elapsed_h) * 2) / 2
    last_alive = t_death - rng.randint(10, 50)

    facts: list[Fact] = [
        Fact("discovery", f"{victim} was found dead in the {place} at {fmt(discovery)}.", {"time": discovery}, "neutral"),
        Fact("body_temp",
             f"At {fmt(discovery)} the body temperature was {temp:.1f}°C. In that room a body cools by about "
             f"{COOLING_PER_HOUR:.0f}°C per hour from {BODY_TEMP:.0f}°C, and the estimate is reliable to within about an hour.",
             {"temp": temp, "discovery": discovery}, "hard"),
        Fact("last_alive", rng.choice([
            f"{victim}'s phone sent a text message at {fmt(last_alive)}.",
            f"{victim} adjusted the heating from their phone at {fmt(last_alive)}.",
            f"A neighbour spoke to {victim} on the landing at {fmt(last_alive)}.",
        ]), {"time": last_alive}, "hard"),
    ]

    lo, hi = death_window(facts)

    # Every suspect is named exactly three times (key sentence, whereabouts,
    # motive) so mention frequency carries no signal about guilt.

    # Access: culprit is always a keyholder; some distractors are excluded by access.
    n_access_excluded = rng.randint(1, max(1, len(distractors) // 2))
    access_excluded = set(rng.sample(distractors, n_access_excluded))
    keyholders = [s for s in suspects if s not in access_excluded]
    rng.shuffle(keyholders)
    no_key = [s for s in suspects if s in access_excluded]
    rng.shuffle(no_key)
    facts.append(Fact("no_forced_entry",
                      f"There was no sign of forced entry; the {place} door had been locked and unlocked with a key.",
                      {}, "hard"))
    holders = ", ".join([victim] + keyholders[:-1]) + f" and {keyholders[-1]}"
    others = " and ".join(no_key) if len(no_key) <= 2 else ", ".join(no_key[:-1]) + f" and {no_key[-1]}"
    facts.append(Fact("keyholders",
                      f"Keys to the {place} are held only by {holders}; {others} never had one.",
                      {"names": sorted(keyholders)}, "hard"))

    # Whereabouts: one line per suspect.
    lo_m, hi_m = cfg["margin"]

    def verified(s: str, a: int, b: int, role: str) -> Fact:
        src = rng.choice(pool["source"]).format(p=rng.choice(pool["alibi_place"]), s=s)
        return Fact("alibi", f"{src} from {fmt(a)} to {fmt(b)}.", {"name": s, "start": a, "end": b}, role)

    def testimony(s: str) -> Fact:
        where = rng.choice(pool["alibi_place"] + ["home"])
        return Fact("testimony", f"{s} says they were at {where} all evening.", {"name": s}, "neutral")

    def partial(s: str) -> Fact | None:
        if hi - lo < 60:
            return None
        if rng.random() < 0.5:                       # verified alibi that ends inside the window
            end = rng.randint(lo + 10, hi - 30)
            return verified(s, end - rng.randint(60, 150), end, "neutral")
        start = rng.randint(lo + 30, hi - 10)        # ... or starts inside it
        return verified(s, start, start + rng.randint(60, 150), "neutral")

    def early(s: str) -> Fact:                        # verified, but over before the window opens
        end = lo - rng.randint(20, 120)
        return verified(s, end - rng.randint(60, 180), end, "neutral")

    def non_clearing(s: str) -> Fact:
        """Whereabouts that do not clear anyone.  The culprit and key-less
        suspects draw from the same mix, so the form of the line carries no
        signal; only its timing does."""
        options = ["testimony", "early"] + (["partial"] if rng.random() < cfg["partial_trap"] else [])
        pick = rng.choice(options)
        if pick == "partial":
            return partial(s) or testimony(s)
        return early(s) if pick == "early" else testimony(s)

    for s in suspects:
        if s == culprit or s in access_excluded:
            facts.append(non_clearing(s))
        else:                                         # keyholder: needs a verified alibi covering the window
            facts.append(verified(s, lo - rng.randint(lo_m, hi_m), hi + rng.randint(lo_m, hi_m), "hard"))

    # Accident scenario and the clue that rules it out.
    if cfg["accident"]:
        facts.append(Fact("homicide", rng.choice([
            "The fatal injury is a blow to the back of the head that no fall in the room could explain.",
            "The post-mortem finds bruising on both wrists consistent with a struggle.",
        ]), {}, "hard"))

    # Motives: everyone has one.  Only the culprit's is not a red herring,
    # and it is not decisive either, so it is neutral.
    for s in suspects:
        text = f"{s}, the {roles[s]}, " + rng.choice(pool["motive"]).format(v=victim) + "."
        facts.append(Fact("motive", text, {"name": s}, "neutral" if s == culprit else "herring"))

    # Salient noise.
    k = rng.randint(*cfg["herrings"])
    for text in rng.sample(pool["salient"], min(k, len(pool["salient"]))):
        facts.append(Fact("salient", text.format(t=fmt(rng.randint(lo - 60, hi))), {}, "herring"))

    return {
        "victim": victim, "suspects": suspects, "roles": roles, "place": place, "culprit": culprit,
        "facts": facts, "accident": cfg["accident"], "t_death": t_death,
    }


# ---------------------------------------------------------------------------
# Case assembly + verification
# ---------------------------------------------------------------------------

def build_case(world: dict[str, Any], case_id: str, level: int, rng: random.Random) -> dict[str, Any] | None:
    facts: list[Fact] = world["facts"]
    suspects: list[str] = world["suspects"]
    culprit = world["culprit"]

    # 1. Unique solution.
    if solve(facts, suspects, world["accident"]) != {culprit}:
        return None
    # 2. Red herrings change nothing.
    for f in facts:
        if f.role == "herring":
            rest = [g for g in facts if g is not f]
            if solve(rest, suspects, world["accident"]) != {culprit}:
                return None
    lo, hi = death_window(facts)
    if not lo <= world["t_death"] <= hi:
        return None

    # Shuffle evidence and scenarios so position carries no signal.
    order = list(range(len(facts)))
    rng.shuffle(order)
    facts = [facts[i] for i in order]
    labels = suspects + (["accident"] if world["accident"] else [])
    rng.shuffle(labels)
    scen_text = {s: f"{s}, the {world['roles'][s]}, killed {world['victim']}." for s in suspects}
    scen_text["accident"] = f"{world['victim']}'s death was an accident."
    sid = {lab: f"S{i}" for i, lab in enumerate(labels, 1)}
    eid = {id(f): f"E{i}" for i, f in enumerate(facts, 1)}

    # 3. Key evidence = facts whose removal breaks uniqueness.
    key = []
    for f in facts:
        rest = [g for g in facts if g is not f]
        if solve(rest, suspects, world["accident"]) != {culprit}:
            key.append(eid[id(f)])
    if not key:
        return None

    answer_key = {
        "true_scenario": sid[culprit],
        "ruled_out": sorted((sid[l] for l in labels if l != culprit), key=lambda x: int(x[1:])),
        "key_evidence": key,
        "red_herrings": [eid[id(f)] for f in facts if f.role == "herring"],
        "time_window": {"label": "time of death", "earliest": fmt(lo), "latest": fmt(hi)},
    }
    return {
        "id": case_id,
        "title": f"Death in the {world['place']}",
        "category": "generated-timeline-access",
        "summary": (f"{world['victim']} was found dead in the {world['place']}. "
                    f"{len(suspects)} people are under suspicion. Work out who could have done it and when."),
        "evidence": [f.text for f in facts],
        "detective_questions": [
            "What window of time does the physical evidence allow for the death?",
            "Who could have entered the room, and how do you know?",
            "Which alibis are verified, and do they cover the whole window?",
            "Which details look important but change nothing?",
        ],
        "analysis_protocol": ["Fix the time window from physical evidence first.",
                              "Then apply access, then verified alibis.",
                              "Treat testimony and motive as unverified."],
        "scenarios": [scen_text[l] for l in labels],
        "hidden_truth": f"{culprit} killed {world['victim']} between {fmt(lo)} and {fmt(hi)}.",
        "false_narrative": "",
        "requires_all": [],
        "teaches": [],
        "solution": {"unlock_key": "GENERATED"},
        "answer_key": answer_key,
        "generator": {"version": 2, "level": level, "facts": [f.to_dict() for f in facts]},
    }


def generate(n: int, seed: int, levels: tuple[int, ...] = (1, 2, 3), pool: str = "A",
             prefix: str = "G") -> list[dict[str, Any]]:
    rng = random.Random(seed)
    cases: list[dict[str, Any]] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 50:
            raise RuntimeError("generator rejected too many worlds; check difficulty settings")
        level = levels[len(cases) % len(levels)]
        world = simulate(rng, level, pool)
        if world is None:
            continue
        case = build_case(world, f"{prefix}{seed:03d}-{len(cases) + 1:05d}", level, rng)
        if case is not None:
            cases.append(case)
    return cases


def load_cases(path: str | Path) -> list[CaseDefinition]:
    """Load a JSONL file of generated cases as CaseDefinition objects."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(CaseDefinition.from_dict(json.loads(line)))
    return out


SPLITS = {
    # name: (count, seed, levels, pool)
    "train": (2000, 1, (1, 2, 3), "A"),
    "val": (200, 2, (1, 2, 3), "A"),
    "test_id": (200, 3, (1, 2, 3), "A"),
    "test_ood": (200, 4, (3,), "B"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate solver-verified detective cases.")
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply split sizes (e.g. 0.1 for a quick run)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (count, seed, levels, pool) in SPLITS.items():
        cases = generate(max(1, int(count * args.scale)), seed, levels, pool, prefix=name[:2].upper())
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"{name:9} {len(cases):5d} cases -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
