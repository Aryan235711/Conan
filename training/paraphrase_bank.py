"""Grow the prose phrasing bank with a local LLM, keeping only verified paraphrases.

v9.1 reads 76-78% of held-out prose paragraphs exactly; most remaining errors
are sentence structures its training never showed. This script asks a local
model (ollama, qwen2.5-coder:7b by default) to paraphrase each training
sentence pattern, then keeps a paraphrase only if it passes every check:

  1. Rules: every placeholder present, no other placeholders or braces, no
     digits, no gendered pronouns, no first person outside quotes, one line.
  2. No four-word run of a held-out test template (the model never sees the
     test bank, but this rules out chance overlap).
  3. Round trip: the pattern is filled with concrete values and a separate
     extraction prompt (temperature 0) must recover the fact type and every
     field exactly. A paraphrase that turns a claim into a record, drops the
     companion from an alibi, or moves a time is rejected.

Output: data/paraphrase_bank.json (or --out), {kind: [template with {s}-style fields]}.
An existing output file is extended, never overwritten. --gen-model picks the
paraphrasing model; the checker always uses --model, so a second generator
adds new styles under the same verification.

    python3 training/paraphrase_bank.py --rounds 8
    python3 training/paraphrase_bank.py --out data/paraphrase_bank_v2.json --gen-model deepseek-r1:8b
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine import prose  # noqa: E402

URL = "http://localhost:11434/api/generate"
OUT = ROOT / "data" / "paraphrase_bank.json"

# kind: (tokens -> prose.py field names, meaning, seed sentences, expected extraction type)
KINDS: dict[str, dict] = {
    "alibi": {
        "fields": {"PERSON": "s", "PLACE": "p", "START": "a", "END": "b"},
        "meaning": "A reliable record (CCTV, card logs, staff, a booking system) confirms that [PERSON] was at "
                   "[PLACE] for the whole time from [START] to [END]. It is verified, not just a claim.",
        "seeds": ["CCTV at [PLACE] shows [PERSON] there continuously from [START] to [END].",
                  "Staff and the booking system confirm [PERSON] was at [PLACE] from [START] until [END]."],
        "type": "A",
    },
    "witness_alibi": {
        "fields": {"WITNESS": "w", "SUSPECT": "s", "PLACE": "p", "START": "a", "END": "b"},
        "meaning": "[WITNESS] claims that [WITNESS] and [SUSPECT] were together at [PLACE] for the whole time from "
                   "[START] to [END]. It is [WITNESS]'s claim, not a record.",
        "seeds": ['[WITNESS] says: "I was with [SUSPECT] at [PLACE] from [START] to [END]."',
                  "According to [WITNESS], [SUSPECT] was with [WITNESS] at [PLACE] the whole time between [START] and [END]."],
        "type": "B",
    },
    "testimony": {
        "fields": {"PERSON": "s", "PLACE": "w"},
        "meaning": "[PERSON] says they were at [PLACE] all evening, but nothing and nobody confirms it. "
                   "It is only [PERSON]'s own word.",
        "seeds": ["[PERSON] says they were at [PLACE] all evening.",
                  "[PERSON] claims to have stayed at [PLACE] all night, but no one can confirm it."],
        "type": "C",
    },
    "record": {
        "fields": {"PERSON": "n", "PLACE": "l", "TIME": "t"},
        "meaning": "A reliable record (CCTV, a receipt, card logs) shows [PERSON] at [PLACE] at one moment, [TIME].",
        "seeds": ["CCTV shows [PERSON] at [PLACE] at [TIME].", "A timestamped receipt puts [PERSON] at [PLACE] at [TIME]."],
        "type": "D",
    },
    "point": {
        "fields": {"PERSON": "s", "PLACE": "q", "TIME": "t"},
        "meaning": "A receipt or card payment shows [PERSON] at [PLACE] at one single moment, [TIME].",
        "seeds": ["A receipt shows [PERSON] buying something at [PLACE] at [TIME].",
                  "[PERSON]'s bank card was used at [PLACE] at [TIME]."],
        "type": "D",
    },
    "claim": {
        "fields": {"PERSON": "s", "PLACE": "l", "TIME": "t"},
        "meaning": "[PERSON] says that [PERSON] was at [PLACE] at [TIME]. It is [PERSON]'s statement.",
        "seeds": ['[PERSON] says: "I was at [PLACE] at [TIME]."', "[PERSON] told the inquiry that at [TIME] they were at [PLACE]."],
        "type": "E",
    },
    "sighting": {
        "fields": {"PERSON": "s", "OTHER": "q", "PLACE": "l", "TIME": "t"},
        "meaning": "[PERSON] says they saw [OTHER] at [PLACE] at [TIME], so both were there at [TIME]. "
                   "It is [PERSON]'s statement.",
        "seeds": ['[PERSON] says: "I saw [OTHER] at [PLACE] at [TIME]."',
                  "[PERSON] remembers bumping into [OTHER] at [PLACE] at [TIME]."],
        "type": "F",
    },
    "holders": {
        "fields": {"ROOM": "room", "HOLDERS": "holders"},
        "meaning": "Only the people listed in [HOLDERS] hold keys to the [ROOM]; nobody else has one.",
        "seeds": ["Keys to the [ROOM] are held only by [HOLDERS].",
                  "The key register shows keys to the [ROOM] held only by [HOLDERS]."],
        "type": "G",
    },
    "alive": {
        "fields": {"VICTIM": "v", "TIME": "t"},
        "meaning": "Something shows [VICTIM] was still alive at [TIME] (a message, a call, being seen, using a door badge).",
        "seeds": ["[VICTIM]'s phone sent a text message at [TIME].", "A neighbour spoke to [VICTIM] on the landing at [TIME]."],
        "type": "H",
    },
    "no_entry": {
        "fields": {"ROOM": "room"},
        "meaning": "Nobody forced their way into the [ROOM]; the door was opened with a key.",
        "seeds": ["There was no sign of forced entry; the [ROOM] door had been locked and unlocked with a key."],
        "type": "I",
    },
    "homicide": {
        "fields": {},
        "meaning": "The victim's injury shows the death was not an accident (no fall could cause it).",
        "seeds": ["The fatal injury is a blow to the back of the head that no fall in the room could explain."],
        "type": "J",
    },
}

EXTRACT = """You read one sentence from a detective case file and say exactly what it states.
Types:
A = verified presence: a record or staff confirm a person was at a place for a span of time. Fields: person, place, from, to.
B = alibi claim: one person says they were together with another person at a place for a span of time. Fields: speaker, companion, place, from, to.
C = unverified account of a whole evening or night: a person's own word about where they spent it, with no clock time given and nothing confirming it. Fields: person.
D = record of a moment: a record, receipt or payment puts a person at a place at one time. Fields: person, place, time.
E = statement about oneself at a clock time: a person says they were at a place at a specific HH:MM time. Fields: speaker, place, time.
F = statement of a sighting: a person says they saw another person at a place at a time. Fields: speaker, seen, place, time.
G = key holders: who holds keys. Fields: names (list).
H = victim alive: the victim was alive at a time. Fields: time.
I = no forced entry: the door was opened with a key. No fields.
J = not an accident: the injury rules out an accident. No fields.
K = none of these.
Copy names, places and times exactly as written. Times are HH:MM; a span has separate from and to times.
The victim (the dead person) is {victim}. Record sources such as CCTV or card logs are never a person.
Examples:
"Card payments at the cinema put Ivan Costa there from 20:10 to 23:50." -> {{"type": "A", "person": "Ivan Costa", "place": "the cinema", "from": "20:10", "to": "23:50"}}
"Hana Evans told police that Hana Evans and Ben Larsen were at the gym together from 19:30 until 22:05." -> {{"type": "B", "speaker": "Hana Evans", "companion": "Ben Larsen", "place": "the gym", "from": "19:30", "to": "22:05"}}
"Olga Petrov claims to have spent the night at home; nobody can confirm it." -> {{"type": "C", "person": "Olga Petrov"}}
"Card logs show Sami Dube at the station at 21:00." -> {{"type": "D", "person": "Sami Dube", "place": "the station", "time": "21:00"}}
"'I was at the library at 20:00,' says Quinn Adler." -> {{"type": "E", "speaker": "Quinn Adler", "place": "the library", "time": "20:00"}}
"Rosa Moreau says that at 19:00 they saw Elif Garcia at the café." -> {{"type": "F", "speaker": "Rosa Moreau", "seen": "Elif Garcia", "place": "the café", "time": "19:00"}}
"Only Ava Jensen, Kavya Nakamura and Pedro Fischer have keys to the studio." -> {{"type": "G", "names": ["Ava Jensen", "Kavya Nakamura", "Pedro Fischer"]}}
"{victim} paid for a sandwich at 18:45." -> {{"type": "H", "time": "18:45"}}
"The office door was unlocked with a key; nothing was forced." -> {{"type": "I"}}
"A blow to the head that no fall could cause killed the victim." -> {{"type": "J"}}
Sentence: {sentence}
Answer with one JSON object such as {{"type": "D", "person": "...", "place": "...", "time": "..."}} and nothing else."""

GEN = """Rewrite the sentence below in {n} different ways for a detective case file.
Meaning to keep exactly: {meaning}
Rules:
- Keep every bracketed token exactly as written: {tokens}. Each must appear at least once.
- Vary the sentence structure a lot: reported speech, direct quotes, passive voice, fronted time phrases, different verbs and nouns, a short clause added before or after.
- One or two sentences each. Never use he, she, his, her or him.
- Do not add any new facts, numbers, times, names or places.{extra}
Original: {seed}
Answer with a JSON list of {n} strings and nothing else."""

# Generic structural hints for kinds whose roles are easy to confuse.
EXTRA = {
    "witness_alibi": "\n- Vary who is named first: sometimes [SUSPECT] first, sometimes [WITNESS] first, sometimes the "
                     "speaker only at the end (for example '..., says [WITNESS].'). It must stay clear that [WITNESS] "
                     "is the one making the claim and that both were together.",
    "sighting": "\n- Vary who is named first; it must stay clear that [PERSON] is the one who saw [OTHER].",
    "claim": "\n- Sometimes put the time first, sometimes the place first.",
    "record": "\n- Name many kinds of record: tills, ticket gates, door logs, phone data, taxi apps, cameras.",
}

GENDERED = re.compile(r"\b(he|she|his|her|him|himself|herself|hers)\b", re.I)
FIRST = re.compile(r"\b(I|me|my|we|us|our)\b")


def ask(model: str, prompt: str, temperature: float, json_mode: bool = False, max_tokens: int = 1200) -> str:
    body = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": max_tokens}}
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["response"]


def parse_json(text: str):
    m = re.search(r"(\[.*\]|\{.*\})", text, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None


def _grams(words: list[str], n: int = 4) -> set[tuple]:
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def _runs(tpl: str) -> list[list[str]]:
    return [re.findall(r"[a-z0-9°'\-]+", c.lower()) for c in re.split(r"\{[a-z_]+\}|\[[A-Z]+\]", tpl)]


TEST_GRAMS = {g for v in prose.BANK.values() for t in v["test"] for w in _runs(t) for g in _grams(w)}
TRAIN_GRAMS = {g for v in prose.BANK.values() for t in v["train"] for w in _runs(t) for g in _grams(w)}
BANNED = TEST_GRAMS - TRAIN_GRAMS


SPEECH = re.compile(r"\b(say|says|said|saying|claim|claims|claimed|told|tells|state|states|stated|statement|insist|"
                    r"insists|insisted|maintain|maintains|maintained|recount|recounts|recounted|report|reports|"
                    r"reported|declare|declares|declared|testimony|account|swear|swears|swore|recall|recalls|"
                    r"recalled|remember|remembers|remembered|according to \[(PERSON|WITNESS)\]|explain|explains|"
                    r"explained|assert|asserts|asserted|describe|describes|described|\")", re.I)
SOURCE = re.compile(r"\b(cctv|camera|cameras|footage|log|logs|logged|card|cards|receipt|receipts|record|records|"
                    r"recorded|register|registry|booking|scan|scans|scanned|rota|data|ticket|tickets|badge|"
                    r"payment|payments|till|timestamp|timestamped|time-stamped|staff|video|system|gps|phone)\b", re.I)
TOGETHER = re.compile(r"\b(with|together|company|alongside|accompanied|accompanying|joined|both|side)\b", re.I)
ONLY = re.compile(r"\b(only|sole|solely|exclusively|no one else|nobody else|no other|alone|just)\b", re.I)
HEDGE = re.compile(r"\b(among others|including|others|also|such as|some of|several)\b", re.I)

# Meaning rules per kind, on top of the round trip: a claim must sound like a
# claim and a record like a record.
SEMANTIC = {
    "alibi": lambda s: SOURCE.search(s) and not SPEECH.search(s),
    "record": lambda s: SOURCE.search(s) and not SPEECH.search(s),
    "point": lambda s: SOURCE.search(s) and not SPEECH.search(s),
    "witness_alibi": lambda s: SPEECH.search(s) and TOGETHER.search(s) and not SOURCE.search(s),
    "testimony": lambda s: SPEECH.search(s) and not re.search(r"\b(cctv|camera|footage|log|card|receipt|register|"
                                                              r"booking|scan|rota|ticket|badge|payment|till)\b", s, re.I),
    "claim": lambda s: SPEECH.search(s) and not SOURCE.search(s),
    "sighting": lambda s: SPEECH.search(s) and not SOURCE.search(s),
    "holders": lambda s: ONLY.search(s) and not HEDGE.search(s),
}


def rule_ok(kind: str, s: str) -> str | None:
    spec = KINDS[kind]
    if "\n" in s or not 15 <= len(s) <= 280:
        return "length"
    toks = set(re.findall(r"\[([A-Z]+)\]", s))
    if toks != set(spec["fields"]):
        return "tokens"
    if "{" in s or "}" in s or re.search(r"\d", s):
        return "chars"
    if GENDERED.search(s):
        return "pronoun"
    # a sentence must not open with a place (it would start lower-case: "the café was ...")
    if re.match(r"\[(ROOM|PLACE)\]", s) or re.search(r"\bon \[TIME\]", s):
        return "awkward"
    outside = re.sub(r'"[^"]*"', "", s)
    if FIRST.search(outside):
        return "first person"
    if any(g in BANNED for w in _runs(s) for g in _grams(w)):
        return "test overlap"
    if kind in SEMANTIC and not SEMANTIC[kind](s):
        return "meaning rule"
    return None


SAMPLE = {"PERSON": "Leo Brandt", "WITNESS": "Maya Okafor", "SUSPECT": "Leo Brandt", "OTHER": "Rosa Iyer",
          "VICTIM": "Tara Hughes", "PLACE": "the bakery", "ROOM": "workshop", "START": "21:40", "END": "00:15",
          "TIME": "22:35", "HOLDERS": "Tara Hughes, Leo Brandt and Rosa Iyer"}


def expected(kind: str) -> dict:
    s = SAMPLE
    t = KINDS[kind]["type"]
    return {
        "A": {"person": s["PERSON"], "place": s["PLACE"], "from": s["START"], "to": s["END"]},
        "B": {"speaker": s["WITNESS"], "companion": s["SUSPECT"], "place": s["PLACE"], "from": s["START"], "to": s["END"]},
        "C": {"person": s["PERSON"]},
        "D": {"person": s["PERSON"], "place": s["PLACE"], "time": s["TIME"]},
        "E": {"speaker": s["PERSON"], "place": s["PLACE"], "time": s["TIME"]},
        "F": {"speaker": s["PERSON"], "seen": s["OTHER"], "place": s["PLACE"], "time": s["TIME"]},
        "G": {"names": ["Tara Hughes", "Leo Brandt", "Rosa Iyer"]},
        "H": {"time": s["TIME"]},
        "I": {}, "J": {},
    }[t]


def _norm(v):
    if isinstance(v, list):
        return sorted(_norm(x) for x in v)
    v = str(v).strip().lower()
    return re.sub(r"^(the|a|an) ", "", v)


def round_trip(model: str, kind: str, s: str) -> bool:
    filled = s
    for tok, val in SAMPLE.items():
        filled = filled.replace(f"[{tok}]", val)
    got = parse_json(ask(model, EXTRACT.format(sentence=filled, victim=SAMPLE["VICTIM"]), 0.0,
                         json_mode=True, max_tokens=120))
    if not isinstance(got, dict) or got.get("type") != KINDS[kind]["type"]:
        return False
    return all(_norm(got.get(k, "")) == _norm(v) for k, v in expected(kind).items())


def to_template(kind: str, s: str) -> str:
    for tok, field in KINDS[kind]["fields"].items():
        s = s.replace(f"[{tok}]", "{" + field + "}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5-coder:7b", help="checker (and default generator)")
    ap.add_argument("--gen-model", default=None, help="paraphrasing model, if different from --model")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--rounds", type=int, default=8, help="generation requests per kind")
    ap.add_argument("--n", type=int, default=8, help="paraphrases per request")
    ap.add_argument("--kinds", default=",".join(KINDS))
    args = ap.parse_args()
    out_path = Path(args.out)
    gen_model = args.gen_model or args.model
    bank = json.loads(out_path.read_text()) if out_path.exists() else {}
    stats = {}
    rng = random.Random(0)
    for kind in args.kinds.split(","):
        spec = KINDS[kind]
        seen = {t for t in bank.get(kind, [])}
        cand: list[str] = []
        reasons: dict[str, int] = {}
        t0 = time.time()
        for _ in range(args.rounds):
            seed = rng.choice(spec["seeds"])
            prompt = GEN.format(n=args.n, meaning=spec["meaning"], seed=seed, extra=EXTRA.get(kind, ""),
                                tokens=", ".join(f"[{k}]" for k in spec["fields"]) or "(none)")
            raw = ask(gen_model, prompt, 0.9, max_tokens=3000 if "r1" in gen_model else 1200)
            out = parse_json(re.sub(r"<think>.*?</think>", "", raw, flags=re.S))   # reasoning models think first
            for s in out if isinstance(out, list) else []:
                if not isinstance(s, str):
                    continue
                s = s.strip()
                why = rule_ok(kind, s)
                if why:
                    reasons[why] = reasons.get(why, 0) + 1
                elif s not in cand:
                    cand.append(s)
        kept = []
        for s in cand:
            if round_trip(args.model, kind, s):
                kept.append(to_template(kind, s))
            else:
                reasons["round trip"] = reasons.get("round trip", 0) + 1
        bank[kind] = sorted(seen | set(kept))
        stats[kind] = {"candidates": len(cand), "kept": len(kept), "rejected": reasons, "seconds": round(time.time() - t0)}
        print(kind, stats[kind], flush=True)
        out_path.write_text(json.dumps(bank, indent=1, ensure_ascii=False))
    print("bank sizes:", {k: len(v) for k, v in bank.items()})


if __name__ == "__main__":
    main()
