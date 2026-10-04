"""Prose cases: the three generated families rewritten as varied paragraphs.

The generated splits state one fact per line in one fixed phrasing. On the
hand-written reliability suite, models trained only on that format misread
prose: they attach a time to the wrong fact, pair a claim with a record about
someone else, and lose facts packed into one paragraph. This module takes a
simulated world from any family and writes it as prose with those traps:

  - several facts in one paragraph (discovery and temperature, access and key
    holders, two alibis, two records, both of a witness's statements);
  - a second, earlier sign of life beside the one that fixes the window;
  - an alibi reported short by one source and extended by another;
  - a timestamped receipt for someone whose account is unverified: a single
    moment, which clears nobody;
  - a returned key that looks like access but changes nothing.

Each case is then rebuilt by the reliability builder, which requires every
name, time and place of a paragraph's facts to appear in its text, re-solves
the case, requires exactly one answer, and recomputes the decisive paragraphs,
red herrings and death window. Nothing is trusted from the rendering step.

Phrasings come in two disjoint banks. Training cases use the "train" bank and
name pool A; the held-out prose test uses the "test" bank and pool B, so a
score on it measures reading prose the model has never seen.

    python3 -m detective_engine.prose --out data/generated
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Any

from . import composite, liar
from . import generator as timeline
from . import prose_grammar as G
from .generator import POOLS, death_window, fmt

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Phrasing banks. Every template states its fields verbatim (names, HH:MM
# times, places, temperatures) so the builder's text check can pass. No
# template uses gendered pronouns.
# ---------------------------------------------------------------------------

BANK: dict[str, dict[str, list[str]]] = {
    "found_temp": {
        "train": [
            "{v} was found dead in the {room} at {t}. At {t} the body temperature was {temp}°C; in that room a body "
            "cools by about 1°C per hour from 37°C, and the estimate is reliable to within about an hour.",
            "At {t} a cleaner found {v} dead in the {room}. The body measured {temp}°C at {t}. A body there loses "
            "about 1°C an hour from a normal 37°C, and the pathologist gives the estimate a margin of about an hour.",
            "The body of {v} was discovered in the {room} at {t}, with a temperature of {temp}°C. Cooling in that "
            "room runs at about 1°C per hour from 37°C; the estimate holds to within about an hour.",
        ],
        "test": [
            "Police were called to the {room} at {t}, where {v} lay dead. A reading taken at {t} gave a body "
            "temperature of {temp}°C. The examiner uses a cooling rate of about 1°C per hour from 37°C for that "
            "room, reliable to about an hour either way.",
            "{v} was discovered lifeless in the {room} at {t}; the thermometer read {temp}°C. The examiner's rule "
            "for that room is roughly 1°C lost per hour from 37°C, give or take an hour.",
        ],
    },
    "found": {
        "train": ["{v} was found dead in the {room} at {t}.",
                  "A cleaner found {v} dead in the {room} at {t}."],
        "test": ["The alarm was raised at {t}, when {v} was found dead in the {room}."],
    },
    "temp": {
        "train": [
            "At {t} the body temperature was {temp}°C. In that room a body cools by about 1°C per hour from 37°C, "
            "and the estimate is reliable to within about an hour.",
            "The pathologist took the body temperature at {t}: {temp}°C. A body there loses about 1°C an hour from "
            "37°C, with a margin of about an hour.",
        ],
        "test": [
            "A reading at {t} put the body at {temp}°C; the examiner allows about 1°C per hour of cooling from "
            "37°C in that room, accurate to about an hour.",
        ],
    },
    "alive": {
        "train": [
            "{v}'s phone sent a text message at {t}.",
            "The building's entry log shows {v} badging in at {t}, alone.",
            "A neighbour spoke to {v} on the landing at {t}.",
            "At {t}, {v} answered a video call from a sister, who is sure of the time from her call log.",
            "{v} adjusted the heating from their phone at {t}.",
        ],
        "test": [
            "Door-camera footage shows {v} letting the cat out at {t}.",
            "{v} replied to an email at {t}, typing at the desk in the {room}.",
        ],
    },
    "homicide": {
        "train": [
            "The fatal injury is a blow to the back of the head that no fall in the room could explain.",
            "The post-mortem finds bruising on both wrists consistent with a struggle.",
            "The pathologist rules out a fall: the skull fracture is on the crown, from a heavy object swung with force.",
        ],
        "test": [
            "Marks on the neck show the victim was strangled; the death was no accident.",
            "The wound was made by a weapon that is missing from the room, which rules out an accident.",
        ],
    },
    "access": {
        "train": [
            "There was no sign of forced entry; the {room} door had been locked and unlocked with a key. "
            "Keys are held only by {holders}.",
            "The lock on the {room} was undamaged and had been opened with a key. The key register lists "
            "{holders} as the only key holders.",
            "Nobody broke in: the {room} door was opened with a key. Only {holders} have keys to it.",
        ],
        "test": [
            "Investigators found the {room} lock intact; whoever came in used a key. Just {holders} were ever "
            "issued one.",
            "The door to the {room} shows no damage and was unlocked normally. The locksmith's records name "
            "{holders} as the people holding keys.",
        ],
    },
    "no_entry": {
        "train": ["There was no sign of forced entry; the {room} door had been locked and unlocked with a key.",
                  "The {room} door was opened with a key; nothing was forced."],
        "test": ["Whoever entered the {room} used a key; the lock and frame are undamaged."],
    },
    "holders": {
        "train": ["Keys to the {room} are held only by {holders}.",
                  "The key register shows keys to the {room} held only by {holders}."],
        "test": ["Only {holders} were ever given a key to the {room}."],
    },
    "never_key": {
        "train": ["{others} never had one.", "{others} never held a key."],
        "test": ["{others} had no key at any point."],
    },
    "key_returned": {
        "train": ["{x} borrowed a spare key to the {room} last spring and returned it the same week; the register "
                  "records the return."],
        "test": ["{x} once had a key to the {room}, but handed it back when the lease changed, as the register shows."],
    },
    "alibi": {
        "train": [
            "Camera footage from {p} shows {s} there continuously from {a} to {b}.",
            "{s} was working a shift at {p}; the rota and the time clock agree on {a} to {b}.",
            "Card payments and the door log at {p} put {s} there from {a} until {b}.",
            "{s} spent the evening at {p}. Staff and the booking system confirm the stay from {a} to {b}.",
        ],
        "test": [
            "The streamed video from {p} shows {s} in shot from {a} through {b} without a break.",
            "{s}'s phone joined the Wi-Fi at {p} at {a} and stayed connected until {b}; staff confirm {s} was "
            "there the whole time.",
        ],
    },
    "alibi_short": {
        "train": ["{s}'s membership card logged {s} in at {p} from {a} to {e}."],
        "test": ["The sign-in sheet at {p} has {s} there from {a}, with a note at {e}."],
    },
    "alibi_extend": {
        "train": ["The camera covering the only exit at {p} shows {s} did not leave until {b}; the card simply stopped "
                  "logging at {e}. So {s} was at {p} from {a} to {b}."],
        "test": ["Staff at {p} say the {e} note was a break, not a departure, and the exit camera confirms {s} "
                 "stayed until {b}. So {s} was there from {a} to {b}."],
    },
    "testimony": {
        "train": ["{s} says they were at {w} all evening.",
                  "{s} claims to have stayed at {w} all night, but no one can confirm it.",
                  "Asked where they were, {s} said {w}; nobody saw them there."],
        "test": ["{s} insists on never leaving {w} that night; there is no record either way."],
    },
    "point": {
        "train": ["A receipt shows {s} buying something at {q} at {t}.",
                  "{s}'s bank card was used at {q} at {t}."],
        "test": ["A till roll from {q} has {s} paying at {t}."],
    },
    "motive": {
        "train": ["{s}, the {role}, {m}."],
        "test": ["It later emerged that {s}, the {role}, {m}."],
    },
    "witness_alibi": {
        "train": [
            '{w} says: "I was with {s} at {p} from {a} to {b}."',
            "{w} told officers that {w} and {s} were together at {p} from {a} until {b}.",
            "According to {w}, {s} was with {w} at {p} the whole time between {a} and {b}.",
        ],
        "test": [
            "{w} vouches for {s}: the two of them were at {p}, {w} says, from {a} to {b}.",
            "In a signed statement, {w} puts {s} at {p} alongside {w} from {a} until {b}.",
        ],
    },
    "record": {
        "train": [
            "Card logs show {n} at {l} at {t}.", "CCTV shows {n} at {l} at {t}.",
            "A timestamped receipt puts {n} at {l} at {t}.",
            "The till at {l} logged a card payment by {n} at {t}.",
            "A doorbell camera at {l} filmed {n} there at {t}.",
        ],
        "test": [
            "{n}'s season ticket was scanned at {l} at {t}.",
            "A time-stamped photo from {l} shows {n} there at {t}.",
        ],
    },
    "claim": {
        "train": ['{s} says: "I was at {l} at {t}."',
                  "{s} told the inquiry that at {t} they were at {l}.",
                  "{s} states that they were at {l} at {t}."],
        "test": ["'I was at {l} at {t},' {s} insisted.",
                 "{s}'s account puts {s} at {l} at {t}."],
    },
    "sighting": {
        "train": ['{s} says: "I saw {q} at {l} at {t}."',
                  "{s} says that at {t} they were at {l} and saw {q} there.",
                  "{s} remembers bumping into {q} at {l} at {t}."],
        "test": ["{s} reports seeing {q} at {l} at {t}; {s} was there as well.",
                 "At {t}, {s} says, both {s} and {q} were at {l}."],
    },
    "demeanour": {
        "train": liar.DEMEANOUR,
        "test": ["{w} laughed nervously while giving evidence.",
                 "{w} asked twice whether a lawyer was needed."],
    },
    "rule_liar": {
        "train": ["Exactly one witness is lying; the others tell the truth. Records are reliable.",
                  "Detectives know that one, and only one, of the witnesses lied. Every record is reliable."],
        "test": ["One witness is not telling the truth; everyone else is honest, and the records can be trusted."],
    },
    "rule_combo": {
        "train": ["Records are reliable. Exactly one witness below is lying; the others tell the truth.",
                  "Every record is reliable, and exactly one of the witnesses who vouch for someone is lying."],
        "test": ["The records can be trusted. Of the witnesses giving alibis, one is lying and the rest are honest."],
    },
}

POINT_PLACES = ["the petrol station", "the corner shop", "the taxi rank", "the late-night pharmacy"]

LLM_BANK = ROOT / "data" / "paraphrase_bank.json"          # v9.2 (bank "wide2")
LLM_BANK_V2 = ROOT / "data" / "paraphrase_bank_v2.json"    # v9.3 (bank "wide3"): v1 plus a second generator
_LLM: dict[str, dict] = {}


def _llm_bank(bank: str = "wide2") -> dict:
    """Verified paraphrases written by training/paraphrase_bank.py."""
    path = LLM_BANK_V2 if bank == "wide3" else LLM_BANK
    if bank not in _LLM:
        _LLM[bank] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _LLM[bank]


class Writer:
    """Chooses phrasing. Banks "train" and "test" are the fixed templates v9 used.
    Bank "wide" mixes the train templates (a quarter of the time) with the
    phrasing grammar, merges more facts per paragraph and adds fillers; it
    draws extra random numbers, so it never changes the other banks' output."""

    def __init__(self, rng: random.Random, bank: str):
        self.rng, self.bank = rng, bank
        self.wide = bank in ("wide", "wide2", "wide3")

    def say(self, kind: str, **kw: Any) -> str:
        if self.bank in ("wide2", "wide3"):
            # Bank "wide2" adds verified paraphrases from a local LLM (data/paraphrase_bank.json):
            # 40% paraphrase where one exists, 15% train template, the rest grammar.
            if kind == "access":
                return self.say("no_entry", **kw) + " " + self.say("holders", **kw)
            if kind not in ("alibi_short", "alibi_extend"):
                llm = _llm_bank(self.bank).get(kind, [])
                roll = self.rng.random()
                if llm and roll < 0.4:
                    return self.rng.choice(llm).format(**kw)
                if roll < 0.55:
                    return self.rng.choice(BANK[kind]["train"]).format(**kw)
            return G.say(self.rng, kind, **kw)
        if self.wide:
            # the short record and its extension refer to each other, so both come from the grammar
            if kind not in ("alibi_short", "alibi_extend") and self.rng.random() < 0.25:
                return self.rng.choice(BANK[kind]["train"]).format(**kw)
            return G.say(self.rng, kind, **kw)
        return self.rng.choice(BANK[kind][self.bank]).format(**kw)

    def merge(self, p: float) -> tuple[float, int]:
        """Merge probability and longest run of paragraphs merged into one."""
        return (min(p + 0.2, 0.6), 3) if self.wide else (p, 2)


def _names(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + f" and {xs[-1]}"


def _fact(kind: str, **data: Any) -> dict:
    """Source-format fact: times as HH:MM strings, as the builder expects."""
    out = {}
    for k, v in data.items():
        out[k] = fmt(v) if k in ("time", "start", "end", "discovery") else v
    return {"kind": kind, "data": out}


def _atoms_fact(kind: str, atoms: list, **data: Any) -> dict:
    return {"kind": kind, "data": {**data, "atoms": [[p, fmt(t), l, pos] for p, t, l, pos in atoms]}}


def _para(text: str, facts: list[dict], herring: bool = False) -> dict:
    return {"text": text, "facts": facts, "herring": herring}


def _merge_pairs(rng: random.Random, paras: list[dict], p: float, max_run: int = 2) -> list[dict]:
    """Merge random neighbours (same herring status) into one paragraph, up to
    max_run paragraphs per merge."""
    out: list[dict] = []
    i = 0
    while i < len(paras):
        a = paras[i]
        if i + 1 < len(paras) and paras[i + 1]["herring"] == a["herring"] and rng.random() < p:
            b = paras[i + 1]
            merged = _para(a["text"] + " " + b["text"], a["facts"] + b["facts"], a["herring"])
            i += 2
            while (max_run > 2 and i < len(paras) and len(merged["facts"]) < max_run
                   and paras[i]["herring"] == a["herring"] and rng.random() < p):
                merged = _para(merged["text"] + " " + paras[i]["text"], merged["facts"] + paras[i]["facts"], a["herring"])
                i += 1
            out.append(merged)
        else:
            out.append(a)
            i += 1
    return out


# ---------------------------------------------------------------------------
# Shared physical evidence for the timeline and combined families
# ---------------------------------------------------------------------------

def _physical(world: dict, W: Writer) -> list[dict]:
    rng, v, room = W.rng, world["victim"], world["place"]
    by = {f.kind: f for f in world["facts"]}
    d, bt, la = by["discovery"], by["body_temp"], by["last_alive"]
    t, temp = d.data["time"], f"{bt.data['temp']:.1f}"
    paras = []
    if rng.random() < 0.6:
        paras.append(_para(W.say("found_temp", v=v, room=room, t=fmt(t), temp=temp),
                           [_fact("discovery", time=t), _fact("body_temp", temp=bt.data["temp"], discovery=t)]))
    else:
        paras.append(_para(W.say("found", v=v, room=room, t=fmt(t)), [_fact("discovery", time=t)]))
        paras.append(_para(W.say("temp", t=fmt(t), temp=temp), [_fact("body_temp", temp=bt.data["temp"], discovery=t)]))
    alive = [la.data["time"]]
    if rng.random() < 0.35:                                   # an earlier sign of life; the later one fixes the window
        alive.append(la.data["time"] - rng.randint(40, 150))
    if W.wide:
        for tm in alive:
            paras.append(_para(W.say("alive", v=v, t=fmt(tm), room=room), [_fact("last_alive", time=tm)]))
    else:
        templates = rng.sample(BANK["alive"][W.bank], len(alive))
        for tm, tpl in zip(alive, templates):
            paras.append(_para(tpl.format(v=v, t=fmt(tm), room=room), [_fact("last_alive", time=tm)]))
    if "homicide" in by:
        paras.append(_para(W.say("homicide"), [_fact("homicide")]))
    return paras


def _access(world: dict, W: Writer) -> list[dict]:
    rng, v, room = W.rng, world["victim"], world["place"]
    kh = next(f for f in world["facts"] if f.kind == "keyholders")
    names = list(kh.data["names"])
    rng.shuffle(names)
    holders = _names([v] + names)
    no_key = [s for s in world["suspects"] if s not in kh.data["names"]]
    kh_fact = _fact("keyholders", names=sorted(kh.data["names"]))
    if rng.random() < 0.6:
        text = W.say("access", room=room, holders=holders)
        if rng.random() < 0.5:
            text += " " + W.say("never_key", others=_names(no_key))
        paras = [_para(text, [_fact("no_forced_entry"), kh_fact])]
    else:
        paras = [_para(W.say("no_entry", room=room), [_fact("no_forced_entry")]),
                 _para(W.say("holders", room=room, holders=holders), [kh_fact])]
    if no_key and rng.random() < 0.3:
        paras.append(_para(W.say("key_returned", x=rng.choice(no_key), room=room), [], herring=True))
    return paras


def _motives(world: dict, W: Writer) -> list[dict]:
    pool = POOLS[world["pool"]]
    out = []
    for f in world["facts"]:
        if f.kind == "motive":
            s = f.data["name"]
            m = W.rng.choice(pool["motive"]).format(v=world["victim"])
            out.append(_para(W.say("motive", s=s, role=world["roles"][s], m=m), [_fact("motive", name=s)],
                             herring=f.role == "herring"))
    return out


def _salient(world: dict, W: Writer) -> list[dict]:
    out = []
    for f in world["facts"]:
        if f.kind == "salient":
            text = f.text
            if W.wide:
                text = W.rng.choice(POOLS[world["pool"]]["salient"] + G.SALIENT).format(
                    t=fmt(W.rng.randint(9 * 60, 13 * 60)))
            out.append(_para(text, [], herring=True))
    return out


def _order(rng: random.Random, head: list[dict], rest: list[dict]) -> list[dict]:
    """Half the cases read in a natural order (physical evidence first), half
    shuffled. A paragraph that refers back to another stays right after it."""
    follow = [p for p in rest if "after" in p]
    rest = [p for p in rest if "after" not in p]
    if rng.random() < 0.5:
        rng.shuffle(rest)
        allp = head + rest
    else:
        allp = head + rest
        rng.shuffle(allp)
    for p in follow:
        i = next(k for k, q in enumerate(allp) if id(q) == p["after"])
        allp.insert(i + 1, {k: v for k, v in p.items() if k != "after"})
    return allp


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def timeline_source(world: dict, W: Writer) -> dict:
    rng, pool = W.rng, POOLS[world["pool"]]
    head = _physical(world, W)
    rest = _access(world, W)
    alibis: list[dict] = []
    for f in world["facts"]:
        if f.kind == "alibi":
            s, a, b = f.data["name"], f.data["start"], f.data["end"]
            p = rng.choice(pool["alibi_place"])
            if f.role == "hard" and b - a > 120 and rng.random() < 0.25:
                e = rng.randint(a + 60, b - 45)             # a short record, extended by a second source
                short = _para(W.say("alibi_short", s=s, p=p, a=fmt(a), e=fmt(e)),
                              [_fact("alibi", name=s, start=a, end=e)])
                ext = _para(W.say("alibi_extend", s=s, p=p, a=fmt(a), b=fmt(b), e=fmt(e)),
                            [_fact("alibi", name=s, start=a, end=b)])
                ext["after"] = id(short)                    # refers back to it, so it must follow it
                rest.append(short)
                rest.append(ext)
            else:
                alibis.append(_para(W.say("alibi", s=s, p=p, a=fmt(a), b=fmt(b)),
                                    [_fact("alibi", name=s, start=a, end=b)]))
        elif f.kind == "testimony":
            s = f.data["name"]
            where = rng.choice(pool["alibi_place"] + ["home"])
            text, facts = W.say("testimony", s=s, w=where), [_fact("testimony", name=s)]
            if rng.random() < 0.4:                           # a single moment: clears nobody
                q, tm = rng.choice(POINT_PLACES), rng.randint(9 * 60, 14 * 60)
                text += " " + W.say("point", s=s, q=q, t=fmt(tm))
                facts.append(_fact("point", name=s, time=tm, place=q))
            rest.append(_para(text, facts))
    rest += _merge_pairs(rng, alibis, *W.merge(0.3)) + _motives(world, W) + _salient(world, W)
    scen = [{"text": f"{s}, the {world['roles'][s]}, killed {world['victim']}.", "label": s} for s in world["suspects"]]
    if world["accident"]:
        scen.append({"text": f"{world['victim']}'s death was an accident.", "label": "accident"})
    return {"family": "timeline", "evidence": _order(rng, head, rest), "scenarios": scen,
            "intended": world["culprit"],
            "title": f"Death in the {world['place']}",
            "summary": (f"{world['victim']} was found dead in the {world['place']}. {len(world['suspects'])} people "
                        "are under suspicion. Work out who could have done it and when."),
            "hidden_truth": f"{world['culprit']} killed {world['victim']}."}


def composite_source(world: dict, W: Writer) -> dict:
    rng = W.rng
    head = [_para(W.say("rule_combo"), [])] + _physical(world, W)
    rest = _access(world, W)
    records = []
    for f in world["facts"]:
        d = f.data
        if f.kind == "witness_alibi":
            rest.append(_para(W.say("witness_alibi", w=d["witness"], s=d["suspect"], p=d["place"],
                                    a=fmt(d["start"]), b=fmt(d["end"])),
                              [_fact("witness_alibi", witness=d["witness"], suspect=d["suspect"], place=d["place"],
                                     start=d["start"], end=d["end"])]))
        elif f.kind == "record":
            records.append(_para(W.say("record", n=d["name"], l=d["place"], t=fmt(d["time"])),
                                 [_fact("record", name=d["name"], time=d["time"], place=d["place"])]))
        elif f.kind == "testimony":
            where = rng.choice(liar.PLACES[world["pool"]])
            rest.append(_para(W.say("testimony", s=d["name"], w=where), [_fact("testimony", name=d["name"])]))
        elif f.kind == "demeanour":
            rest.append(_para(W.say("demeanour", w=d["name"]), [], herring=True))
    rng.shuffle(records)
    rest += _merge_pairs(rng, records, *W.merge(0.3)) + _motives(world, W) + _salient(world, W)
    scen = [{"text": f"{s}, the {world['roles'][s]}, killed {world['victim']}.", "label": s} for s in world["suspects"]]
    scen.append({"text": f"{world['victim']}'s death was an accident.", "label": "accident"})
    body = _order(rng, head[1:], rest)
    return {"family": "composite", "evidence": [head[0]] + body, "scenarios": scen, "intended": world["culprit"],
            "witnesses": world["witnesses"],
            "title": f"The alibi in the {world['place']}",
            "summary": (f"{world['victim']} was found dead in the {world['place']}. {len(world['suspects'])} people "
                        "are under suspicion, and witnesses have vouched for some of them. One witness is lying. "
                        "Work out who killed the victim and when."),
            "hidden_truth": f"{world['culprit']} killed {world['victim']}; {world['liar']} lied to cover for them."}


def liar_source(world: dict, W: Writer) -> dict:
    rng = W.rng
    by_speaker: dict[str, list[dict]] = {}
    records, herrings = [], []
    for f in world["facts"]:
        d = f.data
        if f.kind == "statement":
            atoms = [tuple(a) for a in d["atoms"]]
            s = d["speaker"]
            if len(atoms) == 1:
                _, t, l, _ = atoms[0]
                text = W.say("claim", s=s, l=l, t=fmt(t))
            else:
                (_, t, l, _), (q, _, _, _) = atoms
                text = W.say("sighting", s=s, q=q, l=l, t=fmt(t))
            by_speaker.setdefault(s, []).append(_para(text, [_atoms_fact("statement", atoms, speaker=s)]))
        elif f.kind == "record":
            p, t, l, pos = f.data["atoms"][0]
            records.append(_para(W.say("record", n=p, l=l, t=fmt(t)), [_atoms_fact("record", [(p, t, l, pos)])]))
        elif f.kind == "demeanour":
            herrings.append(_para(W.say("demeanour", w=d["name"]), [], herring=True))
        elif f.kind == "salient":
            text = f.text
            if W.wide:
                text = rng.choice(POOLS[world["pool"]]["salient"] + G.SALIENT).format(
                    t=fmt(rng.randint(6 * 60, 12 * 60)))
            herrings.append(_para(text, [], herring=True))
    statements = []
    for s, ps in by_speaker.items():
        if len(ps) == 2 and rng.random() < 0.5:           # both of a witness's statements in one paragraph
            statements.append(_para(ps[0]["text"] + " " + ps[1]["text"], ps[0]["facts"] + ps[1]["facts"]))
        else:
            statements += ps
    rng.shuffle(records)
    rng.shuffle(herrings)
    body = statements + _merge_pairs(rng, records, *W.merge(0.3)) + _merge_pairs(rng, herrings, *W.merge(0.2))
    rng.shuffle(body)
    ws = world["witnesses"]
    return {"family": "liar", "evidence": [_para(W.say("rule_liar"), [])] + body,
            "scenarios": [{"text": f"{w} is lying.", "label": w} for w in ws], "intended": world["liar"],
            "n_places": world["n_places"], "title": "Who is lying?",
            "summary": (f"{len(ws)} witnesses gave statements about one evening. Exactly one of them is lying. "
                        "Work out who, using the records and the other statements."),
            "hidden_truth": f"{world['liar']} lied."}


FAMILIES = {
    "timeline": (timeline.simulate, timeline_source),
    "composite": (composite.simulate, composite_source),
    "liar": (liar.simulate, liar_source),
}

_BUILD = None


def _builder():
    global _BUILD
    if _BUILD is None:
        spec = importlib.util.spec_from_file_location("reliability_build", ROOT / "benchmarks" / "reliability" / "build.py")
        _BUILD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BUILD)
    return _BUILD


def make_case(family: str, rng: random.Random, level: int, pool: str, bank: str, case_id: str) -> dict | None:
    simulate, to_source = FAMILIES[family]
    world = simulate(rng, level, pool)
    if world is None:
        return None
    world["pool"] = pool
    src = to_source(world, Writer(rng, bank))
    if bank in ("wide", "wide2", "wide3"):               # background sentences inside fact paragraphs
        for para in src["evidence"]:
            if para["facts"] and rng.random() < 0.12:
                para["text"] += " " + rng.choice(G.FILLERS)
    src.update({"id": case_id, "tier": "P", "verification": "solver"})
    b = _builder()
    try:
        case = b.build_solver_case(src)
    except b.BuildError:
        return None
    if family != "liar":
        from .generator import Fact
        facts = [Fact(f["kind"], "", f["data"], f["role"]) for f in case["generator"]["facts"]]
        lo, hi = death_window(facts)
        if not lo <= world["t_death"] <= hi:
            return None
    case.pop("reliability", None)
    case["category"] = f"generated-prose-{family}"
    case["solution"] = {"unlock_key": "GENERATED"}
    case["generator"].update({"version": 2, "level": level, "prose": True, "bank": bank})
    return case


def generate(n: int, seed: int, family: str, levels: tuple[int, ...], pool: str, bank: str,
             prefix: str) -> list[dict]:
    rng = random.Random(seed)
    cases: list[dict] = []
    attempts = 0
    while len(cases) < n:
        attempts += 1
        if attempts > n * 200:
            raise RuntimeError(f"prose {family}: too many rejected cases")
        level = levels[len(cases) % len(levels)]
        case = make_case(family, rng, level, pool, bank, f"{prefix}{seed:03d}-{len(cases) + 1:05d}")
        if case is not None:
            cases.append(case)
    return cases


SPLITS = {
    # name: list of (family, count, seed, levels, pool, bank, prefix)
    "prose_train": [("timeline", 1000, 21, (1, 2, 3), "A", "train", "PRT"),
                    ("liar", 1000, 22, (1, 2, 3), "A", "train", "PRL"),
                    ("composite", 1000, 23, (2, 3), "A", "train", "PRC")],
    # Diagnostic: training phrasings on held-out names and worlds. The gap between
    # this and prose_test measures how much of the prose error is unseen wording.
    "prose_val": [("timeline", 20, 27, (2, 3), "B", "train", "PVT"),
                  ("liar", 20, 28, (2, 3), "B", "train", "PVL"),
                  ("composite", 20, 29, (2, 3), "B", "train", "PVC")],
    # v9.1: the phrasing grammar, levels 2-3 like the test.
    "prose_wide_train": [("timeline", 1000, 31, (2, 3), "A", "wide", "PWT"),
                         ("liar", 1000, 32, (2, 3), "A", "wide", "PWL"),
                         ("composite", 1200, 33, (2, 3), "A", "wide", "PWC")],
    # v9.2: grammar plus verified LLM paraphrases.
    "prose_wide2_train": [("timeline", 1000, 41, (2, 3), "A", "wide2", "PZT"),
                          ("liar", 1000, 42, (2, 3), "A", "wide2", "PZL"),
                          ("composite", 1300, 43, (2, 3), "A", "wide2", "PZC")],
    # v9.3: the expanded paraphrase bank.
    "prose_wide3_train": [("timeline", 1000, 51, (2, 3), "A", "wide3", "PYT"),
                          ("liar", 1000, 52, (2, 3), "A", "wide3", "PYL"),
                          ("composite", 1200, 53, (2, 3), "A", "wide3", "PYC")],
    # A second, untouched held-out prose test (new worlds and names, same held-out
    # phrasing bank). prose_test guided five rounds of decisions, so final numbers
    # for the v9 series are reported on this one too.
    "prose_test2": [("timeline", 50, 74, (2, 3), "B", "test", "PQT"),
                    ("liar", 50, 75, (2, 3), "B", "test", "PQL"),
                    ("composite", 50, 76, (2, 3), "B", "test", "PQC")],
    "prose_test": [("timeline", 50, 24, (2, 3), "B", "test", "PXT"),
                   ("liar", 50, 25, (2, 3), "B", "test", "PXL"),
                   ("composite", 50, 26, (2, 3), "B", "test", "PXC")],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/generated")
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, parts in SPLITS.items():
        need = {"wide2": LLM_BANK, "wide3": LLM_BANK_V2}
        missing = [need[p[5]] for p in parts if p[5] in need and not need[p[5]].exists()]
        if missing:
            print(f"{name:12} skipped: {missing[0].name} not built yet (training/paraphrase_bank.py)")
            continue
        cases = []
        for family, count, seed, levels, pool, bank, prefix in parts:
            cases += generate(max(1, int(count * args.scale)), seed, family, levels, pool, bank, prefix)
        # interleave families so --limit takes a balanced sample
        by_fam = [[c for c in cases if c["generator"]["family"] == f] for f in ("timeline", "liar", "composite")]
        mixed = [c for group in zip(*by_fam) for c in group] if len({len(g) for g in by_fam}) == 1 else cases
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for c in mixed:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"{name:12} {len(mixed):5d} cases -> {out / (name + '.jsonl')}")


if __name__ == "__main__":
    main()
