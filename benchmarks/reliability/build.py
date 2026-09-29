"""Build the reliability suite: hand-written hard cases with verified answer keys.

The generated splits share one format with the training data, so scoring
well on them shows a model learned the method, not that it can handle a case
written differently.  This suite holds cases written in prose, several facts
per paragraph, with edge cases and traps, plus out-of-family reasoning types.

Source files (source/*.json) come in two tiers.

Tier A, "verification": "solver".  Each evidence paragraph lists the
structured facts it states, in the same schema the generators use (times as
"HH:MM").  The builder:
  - runs the family's solver and requires exactly one surviving scenario,
    equal to the author's "intended" label;
  - computes key evidence (paragraphs whose removal makes the answer
    ambiguous) and the death window;
  - checks every paragraph flagged as a red herring changes nothing;
  - checks every name, time and place in a paragraph's facts appears in its
    text, so the prose and the facts cannot drift apart.

Tier B, "verification": "manual" or "computed".  Reasoning types no solver
covers yet.  The author supplies the answer key plus a step-by-step "proof"
for review.  If a "bayes" block is given, posteriors are computed from the
stated priors and likelihoods and the most probable scenario must equal the
intended one.

    python3 benchmarks/reliability/build.py            # writes cases.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine import composite, liar  # noqa: E402
from detective_engine.engine.case_validator import CaseQualityValidator  # noqa: E402
from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.generator import Fact, death_window, fmt  # noqa: E402
from detective_engine.generator import solve as timeline_solve  # noqa: E402

SRC = HERE / "source"
OUT = HERE / "cases.jsonl"

TIME_FIELDS = ("time", "start", "end", "discovery")
QUESTIONS = {
    "timeline": ["What window of time does the physical evidence allow for the death?",
                 "Who could have entered, and does that rule matter here?",
                 "Which verified alibis cover the whole window?",
                 "Which details look important but change nothing?"],
    "liar": ["Which statements can be checked against a record?",
             "What does each 'I saw' statement imply about both people?",
             "If a given witness were the liar, would everything else still fit?"],
    "composite": ["What window does the physical evidence allow?",
                  "Which witness claim does a reliable record contradict?",
                  "Once the liar is discounted, whose alibi still covers the whole window?",
                  "Who could have entered the room?"],
}


class BuildError(Exception):
    pass


def minutes(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return ((h * 60 + m) - 12 * 60) % (24 * 60)


def _convert(data: dict) -> dict:
    out = dict(data)
    for k in TIME_FIELDS:
        if isinstance(out.get(k), str):
            out[k] = minutes(out[k])
    if "atoms" in out:
        out["atoms"] = [[p, minutes(t) if isinstance(t, str) else t, place, pos] for p, t, place, pos in out["atoms"]]
    return out


def _tokens(fact: dict) -> list[str]:
    """Names, times and places a paragraph must state for this fact."""
    d = fact["data"]
    toks: list[str] = []
    for k in ("name", "witness", "suspect", "speaker"):
        if isinstance(d.get(k), str):
            toks.append(d[k])
    toks += d.get("names", [])
    for k in TIME_FIELDS:
        if isinstance(d.get(k), str):
            toks.append(d[k])
    if isinstance(d.get("place"), str):
        toks.append(d["place"])
    for p, t, place, _ in d.get("atoms", []):
        toks += [p, t, place]
    if "temp" in d:
        toks.append(f"{d['temp']:.1f}")
    return toks


def _shuffle_scenarios(case_id: str, scenarios: list[dict]) -> list[dict]:
    order = scenarios[:]
    random.Random(case_id).shuffle(order)
    return order


def build_solver_case(src: dict) -> dict:
    family = src["family"]
    items = src["evidence"]
    # prose / fact consistency
    for i, it in enumerate(items, 1):
        for f in it.get("facts", []):
            missing = [t for t in _tokens(f) if t not in it["text"]]
            if missing:
                raise BuildError(f"{src['id']} E{i}: text does not state {missing}")

    def facts_from(keep: list[int]) -> list[Fact]:
        out = []
        for i in keep:
            it = items[i]
            role = "herring" if it.get("herring") else "neutral"
            for f in it.get("facts", []):
                out.append(Fact(f["kind"], "", _convert(f["data"]), role))
        return out

    scen = _shuffle_scenarios(src["id"], src["scenarios"])
    labels = [s["label"] for s in scen]
    suspects = [l for l in labels if l != "accident"]
    with_accident = "accident" in labels

    def solve(keep: list[int]) -> set[str]:
        fs = facts_from(keep)
        if family == "timeline":
            return timeline_solve(fs, suspects, with_accident)
        if family == "composite":
            return composite.solve(fs, suspects, src["witnesses"], with_accident)
        if family == "liar":
            return liar.solve(fs, suspects, src["n_places"])
        raise BuildError(f"unknown family {family}")

    all_idx = list(range(len(items)))
    sol = solve(all_idx)
    if sol != {src["intended"]}:
        raise BuildError(f"{src['id']}: solver finds {sorted(sol)}, author intended {src['intended']}")
    if family == "composite":
        liars = composite.consistent_liars(facts_from(all_idx), src["witnesses"])
        if len(liars) != 1:
            raise BuildError(f"{src['id']}: {len(liars)} consistent liars, need exactly 1")

    key = [f"E{i + 1}" for i in all_idx if solve([j for j in all_idx if j != i]) != sol]
    herrings = [f"E{i + 1}" for i, it in enumerate(items) if it.get("herring")]
    for h in herrings:
        i = int(h[1:]) - 1
        if solve([j for j in all_idx if j != i]) != sol:
            raise BuildError(f"{src['id']} {h}: flagged red herring changes the answer")
    sid = {l: f"S{i}" for i, l in enumerate(labels, 1)}
    answer_key = {
        "true_scenario": sid[src["intended"]],
        "ruled_out": [sid[l] for l in labels if l != src["intended"]],
        "key_evidence": key,
        "red_herrings": herrings,
    }
    # Each fact records its paragraph ("para", 1-based) so trace writers can
    # cite the paragraph, not the fact's position.
    gen_facts = []
    for i in all_idx:
        for f in facts_from([i]):
            gen_facts.append({**f.to_dict(), "para": i + 1})
    gen = {"family": family, "facts": gen_facts}
    if family in ("timeline", "composite"):
        lo, hi = death_window(facts_from(all_idx))
        answer_key["time_window"] = {"label": "time of death", "earliest": fmt(lo), "latest": fmt(hi)}
    if family == "composite":
        gen["witnesses"] = src["witnesses"]
    if family == "liar":
        gen["n_places"] = src["n_places"]
    return _assemble(src, scen, answer_key, gen)


def build_manual_case(src: dict) -> dict:
    scen = _shuffle_scenarios(src["id"], src["scenarios"])
    labels = [s["label"] for s in scen]
    sid = {l: f"S{i}" for i, l in enumerate(labels, 1)}
    ak = src["answer_key"]
    answer_key = {
        "true_scenario": sid[src["intended"]],
        "ruled_out": [sid[l] for l in ak.get("ruled_out", [])],
        "key_evidence": ak.get("key_evidence", []),
        "red_herrings": ak.get("red_herrings", []),
    }
    if ak.get("time_window"):
        answer_key["time_window"] = ak["time_window"]
    if "bayes" in src:
        b = src["bayes"]
        post = {l: b["priors"][l] for l in labels}
        for lk in b["likelihoods"]:
            for l in labels:
                post[l] *= lk["values"][l]
        z = sum(post.values())
        post = {l: v / z for l, v in post.items()}
        best = max(post, key=post.get)
        if best != src["intended"]:
            raise BuildError(f"{src['id']}: computed posterior favours {best}, author intended {src['intended']}")
        answer_key["posteriors"] = {sid[l]: round(v, 4) for l, v in post.items()}
        drift = abs(sum(answer_key["posteriors"].values()) - 1.0)
        if drift > 0.001:   # rounding: push the residue onto the most probable scenario
            answer_key["posteriors"][sid[best]] = round(answer_key["posteriors"][sid[best]] + 1.0 - sum(answer_key["posteriors"].values()), 4)
    return _assemble(src, scen, answer_key, None)


def _assemble(src: dict, scen: list[dict], answer_key: dict, gen: dict | None) -> dict:
    case = {
        "id": src["id"],
        "title": src["title"],
        "category": f"reliability-{src.get('family', src.get('type', 'manual'))}",
        "summary": src["summary"],
        "evidence": [it["text"] for it in src["evidence"]],
        "detective_questions": src.get("questions") or QUESTIONS.get(src.get("family", ""), []),
        "analysis_protocol": src.get("protocol", ["Separate facts from claims.", "Test each scenario against every fact."]),
        "scenarios": [s["text"] for s in scen],
        "hidden_truth": src["hidden_truth"],
        "false_narrative": src.get("false_narrative", ""),
        "requires_all": [],
        "teaches": [],
        "solution": {"unlock_key": "RELIABILITY"},
        "answer_key": answer_key,
        "reliability": {
            "tier": src["tier"],
            "verification": src["verification"],
            "skills": src.get("skills", []),
            "inspired_by": src.get("inspired_by", ""),
            "proof": src.get("proof", []),
        },
    }
    if gen is not None:
        case["generator"] = gen
    problems = CaseQualityValidator._answer_key_problems(CaseDefinition.from_dict(case))
    if problems:
        raise BuildError(f"{src['id']}: {problems}")
    return case


def main() -> int:
    cases, errors = [], []
    for path in sorted(SRC.glob("*.json")):
        src = json.loads(path.read_text(encoding="utf-8"))
        try:
            case = build_solver_case(src) if src["verification"] == "solver" else build_manual_case(src)
            cases.append(case)
            ak = case["answer_key"]
            print(f"ok    {src['id']:6} tier {src['tier']} {src['verification']:8} "
                  f"{len(case['evidence']):2} paragraphs, key {ak['key_evidence']}")
        except BuildError as exc:
            errors.append(str(exc))
            print(f"ERROR {exc}")
    OUT.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
    print(f"\n{len(cases)} cases written to {OUT.relative_to(ROOT)}; {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
