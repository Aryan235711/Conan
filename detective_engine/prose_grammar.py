"""A phrasing grammar for prose cases: many ways to state each fact.

SFT v9 learned the three to five training templates per fact type instead of
learning to read: on held-out wording it read under half of the paragraphs
correctly. This module composes each sentence from interchangeable parts
(sources, verbs, time expressions, sentence frames, fillers), so the same fact
appears in hundreds of surface forms and memorising a form stops paying off.

Every frame states its fields verbatim (names, HH:MM times, places,
temperatures), so the reliability builder's text check still applies. None
of it may reuse the held-out test bank's wording: tests/test_prose.py checks
that no four-word run of a test template appears in text from this grammar.
No frame uses gendered pronouns.
"""

from __future__ import annotations

import random
from typing import Callable


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _low(s: str) -> str:
    """Lower-case a sentence-initial word for mid-sentence use, leaving acronyms (CCTV) alone."""
    return s if s[1:2].isupper() else s[:1].lower() + s[1:]


def span(r: random.Random, a: str, b: str) -> str:
    """A stay from a to b; a always comes before b in the text."""
    return r.choice([
        f"from {a} to {b}", f"from {a} until {b}", f"between {a} and {b}", f"from {a} till {b}",
        f"from {a} right through to {b}", f"from {a} to {b} without leaving",
    ])


def _found(r: random.Random, v: str, room: str, t: str) -> str:
    finder = r.choice(["a cleaner", "a colleague", "the caretaker", "a neighbour", "a delivery driver", "a friend"])
    return r.choice([
        f"{v} was found dead in the {room} at {t}.",
        f"At {t}, {finder} found {v} dead in the {room}.",
        f"{_cap(finder)} discovered the body of {v} in the {room} at {t}.",
        f"The body of {v} was found in the {room} at {t}.",
        f"{v} was found in the {room} at {t}, already dead.",
    ])


def _temp(r: random.Random, t: str, temp: str) -> str:
    return r.choice([
        f"At {t} the body temperature was {temp}°C.",
        f"The body measured {temp}°C at {t}.",
        f"The body's temperature, recorded at {t}, was {temp}°C.",
        f"At {t} the pathologist measured {temp}°C.",
        f"When checked at {t}, the body was at {temp}°C.",
    ])


def _rate(r: random.Random) -> str:
    return r.choice([
        "In that room a body cools by about 1°C per hour from 37°C, and the estimate is reliable to within about an hour.",
        "Bodies there lose roughly 1°C an hour from 37°C; the estimate carries a margin of about one hour.",
        "The pathologist works on 1°C of cooling per hour from 37°C, plus or minus an hour.",
        "Cooling runs at about 1°C per hour from a starting 37°C, with an error of about an hour.",
        "At that room's temperature a body drops about 1°C each hour from 37°C; allow an hour each side.",
    ])


def found_temp(r, v, room, t, temp):
    parts = [_found(r, v, room, t), _temp(r, t, temp)]
    if r.random() < 0.3:
        parts.reverse()
    return " ".join(parts + [_rate(r)])


def found(r, v, room, t):
    return _found(r, v, room, t)


def temp(r, t, temp):
    return _temp(r, t, temp) + " " + _rate(r)


def alive(r, v, t, room):
    return r.choice([
        f"{v}'s phone sent a message at {t}.",
        f"{v} was seen buying milk at the corner shop at {t}.",
        f"{v} paid for a taxi home at {t}.",
        f"{v} logged off the work laptop at {t}.",
        f"A neighbour heard {v} talking on the landing at {t}.",
        f"{v} signed for a parcel at {t}.",
        f"{v} posted a photo online at {t}.",
        f"The lift camera shows {v} going up alone at {t}.",
        f"At {t}, {v} rang a friend, who is sure of the time.",
        f"{v} was alive at {t}: the building's entry log shows a badge-in.",
    ])


def homicide(r):
    cause = r.choice(["a blow to the back of the head", "defensive cuts on both hands", "a heavy blow to the temple",
                      "bruising on both wrists from a struggle", "a stab wound to the chest"])
    return r.choice([
        f"The post-mortem excludes an accident: it found {cause}.",
        f"No fall could explain {cause}.",
        f"The pathologist is clear this was not a fall: {cause}.",
        f"The injury, {cause}, means this was not an accident.",
    ])


def no_entry(r, room):
    return r.choice([
        f"There was no sign of forced entry; the {room} door had been locked and unlocked with a key.",
        f"The {room} door had not been forced; it was opened with a key.",
        f"Nothing was broken: the {room} was entered with a key.",
        f"No door or window of the {room} was forced, so whoever went in had a key.",
        f"The {room} was locked, and the lock had been opened with a key.",
    ])


def holders(r, room, holders):
    return r.choice([
        f"Keys to the {room} are held only by {holders}.",
        f"Only {holders} hold keys to the {room}.",
        f"The key log lists {holders} as holding keys; nobody else has one.",
        f"{holders} are the only people with keys for the {room}.",
        f"Keys exist for {holders} and for no one else.",
    ])


def access(r, room, holders_):
    return no_entry(r, room) + " " + holders(r, room, holders_)


def never_key(r, others):
    return r.choice([f"{others} never had one.", f"{others} have never had a key.",
                     f"No key was ever made for {others}."])


def key_returned(r, x, room):
    return r.choice([
        f"{x} borrowed a spare {room} key for one afternoon last spring and returned it that evening.",
        f"{x} used to hold a {room} key but gave it back last year, as the key log shows.",
        f"A spare {room} key was lent to {x} in the summer; it came back the next day.",
    ])


SOURCES_SPAN = ["CCTV", "The door log", "Card payments", "The staff rota", "The booking system", "Badge records",
                "Phone location data", "The ticket barrier log", "A number-plate camera", "The attendance register"]


def alibi(r, s, p, a, b):
    src = r.choice(SOURCES_SPAN)
    sp = span(r, a, b)
    return r.choice([
        f"{src} at {p} shows {s} there {sp}.",
        f"{s} was at {p} {sp}; {_low(src)} confirms it.",
        f"According to {_low(src)} at {p}, {s} was there {sp}.",
        f"{src} confirms that {s} was at {p} {sp}.",
        f"{s} arrived at {p} at {a} and left at {b}, which {_low(src)} confirms.",
        f"{s} left {p} at {b}, having arrived at {a}; {_low(src)} backs this up.",
        f"{s}'s evening is verified: {p}, {sp}.",
        f"{src} puts {s} at {p} {sp}.",
    ])


def alibi_short(r, s, p, a, e):
    return r.choice([
        f"The door log at {p} has {s} coming in at {a}, and its last entry for {s} is at {e}.",
        f"{s}'s entry card at {p} registers {s} from {a} to {e}.",
    ])


def alibi_extend(r, s, p, a, b, e):
    return r.choice([
        f"But CCTV at {p} shows {s} leaving only at {b}; the earlier record just stopped at {e}. "
        f"So {s} was at {p} from {a} to {b}.",
        f"A second check fixes the gap: {p}'s exit camera shows {s} stayed until {b}, not {e}, "
        f"having been there since {a}.",
    ])


def testimony(r, s, w):
    return r.choice([
        f"{s} says they were at {w} all evening.",
        f"{s} claims to have been at {w} all night; nobody can back this up.",
        f"{s} told detectives they spent the night at {w}. No record supports it.",
        f"According to {s}, they were at {w}; no one else can confirm this.",
        f"{s}'s only alibi is their own word: {w}, all evening.",
        f"{s} says they stayed at {w} the whole night. It cannot be checked.",
    ])


def point(r, s, q, t):
    return r.choice([
        f"A receipt shows {s} buying something at {q} at {t}.",
        f"{s} withdrew cash at {q} at {t}.",
        f"A single CCTV frame shows {s} at {q} at {t}.",
        f"{s} bought a coffee at {q} at {t}.",
        f"{s}'s bank card was used at {q} at {t}.",
    ])


def motive(r, s, role, m):
    return r.choice([
        f"{s}, the {role}, {m}.",
        f"Neighbours say {s}, the {role}, {m}.",
        f"{s} (the {role}) {m}.",
        f"Friends of the victim say that {s}, the {role}, {m}.",
    ])


def witness_alibi(r, w, s, p, a, b):
    sp = span(r, a, b)
    return r.choice([
        f'{w} says: "I was with {s} at {p} from {a} to {b}."',
        f"{w} told officers that {w} and {s} were together at {p} {sp}.",
        f'"{s} was with me at {p} {sp}," says {w}.',
        f"{w} gives {s} an alibi: together at {p} {sp}.",
        f"{w} states that {s} never left {w}'s company at {p} from {a} to {b}.",
        f"According to {w}, {w} and {s} were at {p} together {sp}.",
    ])


SOURCES_POINT = ["Card logs", "CCTV", "A timestamped receipt", "Bank records", "A taxi booking", "Phone location data",
                 "The door log", "A bus ticket", "A loyalty-card scan", "A parking ticket"]


def record(r, n, l, t):
    src = r.choice(SOURCES_POINT)
    low = _low(src)
    return r.choice([
        f"{src} shows {n} at {l} at {t}.",
        f"{src} puts {n} at {l} at {t}.",
        f"{n} was at {l} at {t}, according to {low}.",
        f"At {t}, {low} recorded {n} at {l}.",
        f"{src} from {l} is stamped {t} and shows {n}.",
        f"{src} places {n} at {l} at {t}.",
    ])


def claim(r, s, l, t):
    return r.choice([
        f'{s} says: "I was at {l} at {t}."',
        f'"At {t} I was at {l}," says {s}.',
        f"{s} told the inquiry that at {t} they were at {l}.",
        f"{s} claims to have been at {l} at {t}.",
        f"Asked about {t}, {s} said they were at {l}.",
        f"{s}'s statement: at {t}, at {l}.",
    ])


def sighting(r, s, q, l, t):
    return r.choice([
        f'{s} says: "I saw {q} at {l} at {t}."',
        f'"I ran into {q} at {l} at {t}," {s} says.',
        f"According to {s}, {q} was at {l} at {t}, and so was {s}.",
        f"{s} remembers seeing {q} at {l} at {t}, where {s} was too.",
        f"At {t} {s} was at {l} and spotted {q} there.",
        f"{s} and {q} were at {l} together at {t}, says {s}.",
    ])


def demeanour(r, w):
    return r.choice([
        f"{w} seemed nervous and avoided eye contact during the interview.",
        f"{w} answered every question calmly and in great detail.",
        f"{w} kept glancing at the door during the interview.",
        f"{w} gave short answers and refused a cup of coffee.",
        f"{w} was polite but seemed exhausted.",
        f"{w} became angry when asked about money.",
        f"{w} spoke very fast and repeated themselves.",
    ])


def rule_liar(r):
    return r.choice([
        "Exactly one witness is lying; the others tell the truth. Records are reliable.",
        "One of the witnesses lied and all the others told the truth. The records are all reliable.",
        "Every record below is reliable. Of the witnesses, exactly one is lying.",
    ])


def rule_combo(r):
    return r.choice([
        "Records are reliable. Exactly one witness below is lying; the others tell the truth.",
        "All records are reliable. Among the witnesses who give alibis, exactly one is lying.",
        "The records below are reliable, and exactly one alibi witness lied.",
    ])


FILLERS = ["It had been raining since the afternoon.", "The street was quiet that night.",
           "Detectives checked this twice.", "This was confirmed the next morning.",
           "The information came in late on the first day."]

SALIENT = [
    "A cup of cold tea was on the desk; the victim always left one there.",
    "The radio was still on; the victim fell asleep to it most nights.",
    "A taxi waited outside at {t}; it had been booked by a neighbour.",
    "A light in the stairwell flickered around {t}; the bulb has been faulty for weeks.",
    "An umbrella was found by the door; it belongs to the cleaner.",
]

GRAMMAR: dict[str, Callable] = {
    "found_temp": found_temp, "found": found, "temp": temp, "alive": alive, "homicide": homicide,
    "access": access, "no_entry": no_entry, "holders": holders, "never_key": never_key,
    "key_returned": key_returned, "alibi": alibi, "alibi_short": alibi_short, "alibi_extend": alibi_extend,
    "testimony": testimony, "point": point, "motive": motive, "witness_alibi": witness_alibi,
    "record": record, "claim": claim, "sighting": sighting, "demeanour": demeanour,
    "rule_liar": rule_liar, "rule_combo": rule_combo,
}


def say(r: random.Random, kind: str, **kw) -> str:
    """Compose a sentence for this fact kind from the grammar."""
    fn = GRAMMAR[kind]
    names = fn.__code__.co_varnames[1:fn.__code__.co_argcount]
    alias = {"temp_": "temp", "holders_": "holders"}
    args = {n: kw[alias.get(n, n)] for n in names}
    return fn(r, **args)
