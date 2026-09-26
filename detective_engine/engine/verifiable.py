"""Verifiable scorer (v2) — rewards answers that can be checked, not words that match.

The v1 engine scores keyword overlap, which rewards keyword stuffing and
penalizes correct answers that mention the scenario they reject.  This
module scores a structured final answer against the case's ``answer_key``
with exact, ungameable checks:

    correctness   picked the true scenario                       (binary)
    calibration   Brier skill score vs. a uniform forecast        (0..1)
                  -> hedging with even odds earns 0
    elimination   F1 of ruled-out scenarios vs. the key           (0..1)
    evidence      F1 of cited decisive evidence vs. the key       (0..1)
                  -> citing every clue destroys precision
    red_herrings  F1 of flagged distractors vs. the key           (0..1)
    time_window   interval overlap (IoU) with the true window     (0..1)

Internally inconsistent answers (top pick is not the most probable, top
pick is also ruled out, a ruled-out scenario keeps real probability) are
penalized by halving the reward per violation.

Unparseable output scores 0.0 — it is never silently skipped, so a policy
trained on this reward cannot escape it by emitting garbage.

Expected final-answer format (a JSON object, usually in a ```json block
at the end of the response):

    {
      "most_likely": "S2",
      "probabilities": {"S1": 0.1, "S2": 0.8, "S3": 0.1},
      "ruled_out": ["S3"],
      "key_evidence": ["E7", "E9"],
      "red_herrings": ["E5"],
      "time_window": {"earliest": "23:15", "latest": "00:45"}   # only if asked
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .case_loader import CaseLoader
from .models import AnswerKey, CaseDefinition, TimeWindow

# Component weights (renormalized over the components a case actually has).
WEIGHTS: dict[str, float] = {
    "correctness": 0.40,
    "calibration": 0.20,
    "elimination": 0.15,
    "evidence": 0.15,
    "red_herrings": 0.10,
    "time_window": 0.15,
}

PASS_THRESHOLD = 0.70
RULED_OUT_MAX_PROB = 0.10   # a scenario you rule out should not keep more than this


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VerifiableResult:
    case_id: str
    reward: float                      # 0..1 — the training / eval signal
    passed: bool
    format_ok: bool
    correct: bool = False
    components: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    feedback: list[str] = field(default_factory=list)
    answer: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "reward": round(self.reward, 4),
            "passed": self.passed,
            "format_ok": self.format_ok,
            "correct": self.correct,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "violations": self.violations,
            "feedback": self.feedback,
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)


def parse_final_answer(text: str) -> dict[str, Any] | None:
    """Extract the last JSON object that has a "most_likely" key.

    Reasoning-model <think> blocks are ignored.  Returns None if no such
    object exists.
    """
    if not text:
        return None
    cleaned = _THINK_RE.sub(" ", text)
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for m in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "most_likely" in obj:
            found = obj
    return found


def _norm_id(value: Any, prefix: str) -> str | None:
    """Normalize "s2", "S2", 2, "Scenario 2" -> "S2"."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{prefix}{value}"
    if isinstance(value, str):
        m = re.search(r"(\d+)", value)
        if m and (value.strip().upper().startswith(prefix) or value.strip().isdigit()
                  or value.strip().lower().startswith(("scenario", "evidence"))):
            return f"{prefix}{int(m.group(1))}"
    return None


def _norm_id_list(values: Any, prefix: str, valid: set[str]) -> set[str]:
    if not isinstance(values, list):
        return set()
    out = set()
    for v in values:
        nid = _norm_id(v, prefix)
        if nid in valid:
            out.add(nid)
    return out


# ---------------------------------------------------------------------------
# Component metrics
# ---------------------------------------------------------------------------

def _f1(predicted: set[str], truth: set[str]) -> float:
    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0
    tp = len(predicted & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(truth)
    return 2 * precision * recall / (precision + recall)


def _normalize_probs(raw: Any, valid: list[str]) -> dict[str, float] | None:
    """Map to {Si: p} over all valid scenarios, clipped and renormalized."""
    if not isinstance(raw, dict):
        return None
    probs = {sid: 0.0 for sid in valid}
    for k, v in raw.items():
        sid = _norm_id(k, "S")
        if sid in probs and isinstance(v, (int, float)) and not isinstance(v, bool):
            probs[sid] = max(0.0, float(v))
    total = sum(probs.values())
    if total <= 0:
        return None
    # Accept percentages or probabilities; anything that sums to ~1 or ~100.
    return {k: v / total for k, v in probs.items()}


def brier_skill(probs: dict[str, float], true_id: str) -> float:
    """1 - Brier / Brier(uniform), clipped to [0, 1]."""
    n = len(probs)
    if n < 2:
        return 1.0
    brier = sum((p - (1.0 if sid == true_id else 0.0)) ** 2 for sid, p in probs.items())
    brier_uniform = (n - 1) / n
    return max(0.0, 1.0 - brier / brier_uniform)


def _clock_minutes(hhmm: str) -> int | None:
    """Minutes since 12:00 noon, so overnight windows (23:14-01:00) are contiguous."""
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(hhmm))
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if h > 23 or mi > 59:
        return None
    return ((h * 60 + mi) - 12 * 60) % (24 * 60)


def window_iou(pred: Any, truth: TimeWindow) -> float:
    if not isinstance(pred, dict):
        return 0.0
    a0, a1 = _clock_minutes(pred.get("earliest", "")), _clock_minutes(pred.get("latest", ""))
    b0, b1 = _clock_minutes(truth.earliest), _clock_minutes(truth.latest)
    if None in (a0, a1, b0, b1) or a1 < a0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union == 0:
        return 1.0 if a0 == b0 else 0.0
    return inter / union


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def score_answer(case: CaseDefinition, answer: str | dict[str, Any]) -> VerifiableResult:
    """Score one answer (raw model text or an already-parsed dict) for one case."""
    key: AnswerKey | None = case.answer_key
    if key is None:
        raise ValueError(f"Case {case.id} has no answer_key; it cannot be scored verifiably.")

    parsed = parse_final_answer(answer) if isinstance(answer, str) else answer
    scenario_ids = [f"S{i}" for i in range(1, len(case.scenarios) + 1)]
    evidence_ids = {f"E{i}" for i in range(1, len(case.evidence) + 1)}

    most_likely = _norm_id(parsed.get("most_likely"), "S") if isinstance(parsed, dict) else None
    if not isinstance(parsed, dict) or most_likely not in scenario_ids:
        return VerifiableResult(
            case_id=case.id, reward=0.0, passed=False, format_ok=False,
            feedback=["No valid final answer: expected a JSON object with most_likely set to a scenario ID."],
            answer=parsed if isinstance(parsed, dict) else None,
        )

    comps: dict[str, float] = {}
    violations: list[str] = []
    feedback: list[str] = []

    # correctness
    correct = most_likely == key.true_scenario
    comps["correctness"] = 1.0 if correct else 0.0

    # calibration
    probs = _normalize_probs(parsed.get("probabilities"), scenario_ids)
    if probs is None:
        comps["calibration"] = 0.0
        feedback.append("No usable probabilities: calibration scored 0.")
    else:
        comps["calibration"] = brier_skill(probs, key.true_scenario)

    # elimination
    ruled_out = _norm_id_list(parsed.get("ruled_out"), "S", set(scenario_ids))
    comps["elimination"] = _f1(ruled_out, set(key.ruled_out))

    # evidence
    cited = _norm_id_list(parsed.get("key_evidence"), "E", evidence_ids)
    if key.key_evidence:
        comps["evidence"] = _f1(cited, set(key.key_evidence))

    # red herrings (only scored when the case defines some)
    flagged = _norm_id_list(parsed.get("red_herrings"), "E", evidence_ids)
    if key.red_herrings:
        comps["red_herrings"] = _f1(flagged, set(key.red_herrings))

    # time window
    if key.time_window is not None:
        comps["time_window"] = window_iou(parsed.get("time_window"), key.time_window)

    # consistency
    if probs is not None:
        top = max(probs.values())
        if probs[most_likely] < top - 1e-9:
            violations.append("most_likely is not the scenario given the highest probability")
        leaked = [s for s in ruled_out if probs[s] > RULED_OUT_MAX_PROB]
        if leaked:
            violations.append(f"ruled-out scenarios still hold probability > {RULED_OUT_MAX_PROB}: {sorted(leaked)}")
    if most_likely in ruled_out:
        violations.append("most_likely is also listed as ruled out")
    if cited & flagged:
        violations.append(f"the same evidence is cited as key and as a red herring: {sorted(cited & flagged)}")

    weights = {k: WEIGHTS[k] for k in comps}
    total_w = sum(weights.values())
    base = sum(weights[k] * comps[k] for k in comps) / total_w
    reward = base * (0.5 ** len(violations))

    # feedback (post-hoc; reveals the key, so only show after an attempt)
    if not correct:
        feedback.append(f"Wrong scenario: chose {most_likely}.")
    missed = set(key.key_evidence) - cited
    if missed:
        feedback.append(f"Missed decisive evidence: {sorted(missed, key=lambda x: int(x[1:]))}")
    extra = cited - set(key.key_evidence)
    if extra:
        feedback.append(f"Cited non-decisive evidence as key: {sorted(extra, key=lambda x: int(x[1:]))}")
    wrong_elim = ruled_out - set(key.ruled_out)
    if wrong_elim:
        feedback.append(f"Ruled out scenarios the evidence does not exclude: {sorted(wrong_elim)}")

    return VerifiableResult(
        case_id=case.id,
        reward=reward,
        passed=correct and reward >= PASS_THRESHOLD and not violations,
        format_ok=True,
        correct=correct,
        components=comps,
        weights={k: v / total_w for k, v in weights.items()},
        violations=violations,
        feedback=feedback + [f"Inconsistency: {v}" for v in violations],
        answer=parsed,
    )


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

_INSTRUCTIONS = """\
Work through the case: separate observations from interpretations, list the
explanations each clue allows, find contradictions, and decide which
scenarios the evidence rules out.  Refer to evidence as E1, E2, ... and to
scenarios as S1, S2, ...

End your response with a JSON object in a ```json block:
{fields}
Rules: probabilities cover every scenario and sum to 1. A scenario you rule
out must be physically or logically excluded by the evidence, not merely
unlikely. key_evidence lists only the decisive clues. red_herrings lists
clues that look important but do not discriminate between scenarios."""


def build_prompt(case: CaseDefinition) -> str:
    """Render a case as a model prompt. Never includes the answer key or hidden truth."""
    key = case.answer_key
    lines = [f"CASE {case.id}: {case.title}", "", case.summary, "", "EVIDENCE:"]
    lines += [f"E{i}. {e}" for i, e in enumerate(case.evidence, 1)]
    lines += ["", "SCENARIOS:"]
    lines += [f"S{i}. {s}" for i, s in enumerate(case.scenarios, 1)]
    if case.detective_questions:
        lines += ["", "QUESTIONS TO CONSIDER:"]
        lines += [f"- {q}" for q in case.detective_questions]
    example = {
        "most_likely": "S?",
        "probabilities": {f"S{i}": 0.0 for i in range(1, len(case.scenarios) + 1)},
        "ruled_out": ["S?"],
        "key_evidence": ["E?"],
        "red_herrings": ["E?"],
    }
    if key is not None and key.time_window is not None:
        example["time_window"] = {"earliest": "HH:MM", "latest": "HH:MM"}
        lines += ["", f"Also estimate the {key.time_window.label} as a clock-time window."]
    lines += ["", _INSTRUCTIONS.format(fields=json.dumps(example, indent=2))]
    return "\n".join(lines)


class VerifiableScorer:
    """Loads cases once; scores answers against their answer keys."""

    def __init__(self, cases_dir: str | None = None, cases: list[CaseDefinition] | None = None):
        if cases is None:
            cases, _ = CaseLoader(cases_dir).load_all()
        self._cases = {c.id: c for c in cases if c.answer_key is not None}

    @property
    def case_ids(self) -> list[str]:
        return sorted(self._cases)

    def case(self, case_id: str) -> CaseDefinition:
        return self._cases[case_id]

    def prompt(self, case_id: str) -> str:
        return build_prompt(self._cases[case_id])

    def score(self, case_id: str, answer: str | dict[str, Any]) -> VerifiableResult:
        if case_id not in self._cases:
            raise KeyError(f"Unknown or unkeyed case: {case_id}")
        return score_answer(self._cases[case_id], answer)

    def reward(self, case_id: str, answer: str | dict[str, Any]) -> float:
        """Scalar reward for RL loops."""
        return self.score(case_id, answer).reward
