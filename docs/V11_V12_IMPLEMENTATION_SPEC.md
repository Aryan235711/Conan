# V11 / V12 Implementation Spec

**Focus:** perception integrity, adaptive probing, adversarial perception behavior detection, and longitudinal cognitive-integrity tracking.

This spec extends V10 without replacing it.

---

## Design goals

V10 evaluates reasoning quality and derivation integrity.

V11 / V12 add a new layer:

1. **Perception integrity** — what was noticed, when, and how stably it persisted
2. **Adversarial perception detection** — whether the observation trace looks strategically repaired
3. **Adaptive tutoring** — when to probe, and what to ask next
4. **Longitudinal learning model** — how perceptual weaknesses evolve over time

---

## New module

Create:

- [detective_engine/engine/perception_integrity.py](detective_engine/engine/perception_integrity.py)

This module should remain independent from `validator.py` so the perception layer can be added gradually.

---

## Core principle

Separate three different failure types:

1. **Perception miss** — clue never registered early
2. **Memory loss** — clue was initially observed but later dropped
3. **Strategic repair** — clue appears late in a way inconsistent with genuine early perception

That separation is the core architectural reason for introducing trace-based capture.

---

## Proposed data structures

Use stable evidence IDs, not raw text, as the main key.

Reason: raw text is noisy, paraphrasable, and hard to compare across phases.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ObservationEvent:
    evidence_id: str | None
    text: str
    timestamp: float
    phase: Literal["initial", "later", "revision"]
    order_index: int
    source: Literal["user", "probe", "system"] = "user"
    confidence: float = 1.0


@dataclass
class PerceptionTrace:
    initial: list[ObservationEvent] = field(default_factory=list)
    later: list[ObservationEvent] = field(default_factory=list)
    revisions: list[ObservationEvent] = field(default_factory=list)
    first_hypothesis_timestamp: float | None = None
    first_elimination_timestamp: float | None = None


@dataclass(frozen=True)
class EvidenceMeta:
    id: str
    text: str
    diagnostic_weight: float
    salience_level: float
    misdirection_risk: float
    salience_confidence: float
    bias_tags: list[str] = field(default_factory=list)
    critical: bool = False


@dataclass
class PerceptionIntegrityResult:
    coverage: float
    retention_score: float
    late_injection_score: float
    causal_uptake_score: float
    salience_distortion: float
    narrative_lock_in: float
    timing_consistency: float
    strategic_neutrality_flag: bool
    adversarial_flags: list[str] = field(default_factory=list)
    recommended_probes: list[str] = field(default_factory=list)
```

---

## Case JSON extension

Add optional perception metadata per evidence item.

Recommended shape:

```json
{
  "id": "ev_01",
  "text": "A chair is slightly pulled back from the table.",
  "diagnostic_weight": 0.72,
  "salience_level": 0.65,
  "misdirection_risk": 0.15,
  "salience_confidence": 0.82,
  "bias_tags": ["activity-trace", "contradiction-anchor"],
  "critical": true
}
```

### Notes

- `diagnostic_weight` is how much the clue helps discriminate among hypotheses
- `salience_level` is how naturally human attention is drawn to it
- `misdirection_risk` is how likely the clue is to capture attention in an unhelpful way
- `salience_confidence` prevents false precision in author annotations
- `critical` marks clues that perception scoring should care about most

---

## New capture flow

V11 should add an **early micro-capture** before normal observation elaboration.

### Proposed interaction flow

1. **Micro-capture**
   - “List the first 3 raw details you notice.”
   - short, low-latency, no explanation
2. **Standard observation phase**
   - collect full observations
3. **Reasoning phases**
   - anomalies, hypotheses, contradiction notes, elimination
4. **Adaptive probes** if triggered
   - only after suspicious perception patterns appear

This creates the separation between:

- initial perception
- later recall
- strategic repair

---

## Core metrics

## 1. Retention score

Measures whether early observations persist later.

```python
def retention_score(initial_ids: set[str], later_ids: set[str]) -> float:
    if not initial_ids:
        return 0.0
    retained = initial_ids & later_ids
    return len(retained) / len(initial_ids)
```

Interpretation:

- high = stable perception
- low = memory drop or weak encoding

---

## 2. Late injection score

Measures how many critical clues appear late without early registration.

```python
def late_injection_score(
    initial_ids: set[str],
    later_ids: set[str],
    critical_ids: set[str],
) -> float:
    if not critical_ids:
        return 0.0
    injected = (later_ids - initial_ids) & critical_ids
    return len(injected) / len(critical_ids)
```

Interpretation:

- high = possible strategic repair or memory contamination

---

## 3. Causal uptake score

Measures whether observed clues actually influence later reasoning.

```python
def causal_uptake_score(
    observed_ids: set[str],
    used_in_reasoning_ids: set[str],
) -> float:
    if not observed_ids:
        return 0.0
    used = observed_ids & used_in_reasoning_ids
    return len(used) / len(observed_ids)
```

Interpretation:

- high = observations were cognitively integrated
- low = decorative or checklist-style observation

---

## 4. Coverage

Measures how many critical clues were noticed at all.

```python
def coverage_score(
    observed_ids: set[str],
    critical_ids: set[str],
) -> float:
    if not critical_ids:
        return 1.0
    return len(observed_ids & critical_ids) / len(critical_ids)
```

Interpretation:

- low = likely perceptual omission

---

## 5. Salience distortion

Measures whether the user over-selects vivid / misleading clues while under-selecting critical but quieter clues.

```python
def salience_distortion(
    observed_ids: set[str],
    evidence_map: dict[str, EvidenceMeta],
) -> float:
    if not observed_ids:
        return 0.0

    salient = 0.0
    diagnostic = 0.0
    for eid in observed_ids:
        meta = evidence_map.get(eid)
        if not meta:
            continue
        salient += meta.salience_level * max(meta.misdirection_risk, 0.1)
        diagnostic += meta.diagnostic_weight

    if diagnostic <= 0:
        return min(1.0, salient)
    return max(0.0, min(1.0, (salient - diagnostic) / max(diagnostic, 1e-6)))
```

Interpretation:

- high = likely attraction to vivid but low-value cues

---

## 6. Narrative lock-in

Measures how much hypothesis commitment happened before sufficient critical observation coverage.

```python
def narrative_lock_in(
    trace: PerceptionTrace,
    initial_critical_coverage: float,
) -> float:
    if trace.first_hypothesis_timestamp is None:
        return 0.0

    if initial_critical_coverage >= 0.6:
        return 0.0

    return 1.0 - initial_critical_coverage
```

Interpretation:

- high = early top-down story formation before enough evidence registration

---

## 7. Timing-content consistency

Measures whether early observations are plausibly aligned with what humans naturally notice.

```python
def timing_consistency(
    trace: PerceptionTrace,
    evidence_map: dict[str, EvidenceMeta],
) -> float:
    if not trace.initial:
        return 0.0

    score = 0.0
    for event in trace.initial:
        meta = evidence_map.get(event.evidence_id or "")
        if not meta:
            continue
        score += meta.salience_level

    return score / len(trace.initial)
```

Interpretation:

- extremely low = suspiciously “too perfect” early capture of only quiet clues
- extremely high = likely salience-driven bias
- middle range is often healthiest

---

## Adversarial detection heuristics

## Strategic neutrality

A user may try to look unbiased by inserting all categories later.

```python
def detect_strategic_neutrality(result: PerceptionIntegrityResult) -> bool:
    return (
        result.coverage > 0.80
        and result.late_injection_score > 0.50
        and result.causal_uptake_score < 0.40
    )
```

Meaning:

- looks balanced on paper
- but balance arrived late
- and did not affect reasoning

This is a strong signal of performative fairness.

---

## Additional adversarial flags

Recommended flag rules:

- `late_fairness_injection`
  - many critical disconfirming clues appear only after hypothesis formation
- `decorative_observation`
  - clue observed but never used downstream
- `revision_laundering`
  - revisions repair earlier omission after narrative lock-in
- `checklist_symmetry`
  - suspiciously even category coverage with weak uptake
- `salience_capture`
  - vivid misleading clues dominate early trace
- `premature_commitment`
  - first hypothesis arrived before adequate critical coverage

---

## Adaptive probing engine

Probing should be conditional.

```python
def should_probe(result: PerceptionIntegrityResult) -> bool:
    return (
        result.salience_distortion > 0.60
        or result.late_injection_score > 0.40
        or result.narrative_lock_in > 0.70
        or result.strategic_neutrality_flag
    )
```

### Probe bank

```python
PROBES = [
    "List 2 clues that weaken your current hypothesis.",
    "List 1 detail you almost ignored.",
    "List 2 low-salience clues that may matter.",
    "Which clue did you notice late, and why?",
    "Name 1 observed detail you have not used in your reasoning yet.",
]
```

### Probe policy

Map probe to failure mode:

- high `late_injection_score` → ask about late-noticed clues
- high `salience_distortion` → ask for low-salience clues
- high `narrative_lock_in` → ask for disconfirming clues
- low `causal_uptake_score` → ask which observed clue was unused

---

## Integration with V10

Important: do **not** directly dump raw perception metrics into the main score at first.

Recommended rollout:

### Phase A — observational only

- compute `PerceptionIntegrityResult`
- show it in report
- do not affect pass/fail yet

### Phase B — soft weighting

Add a small bounded bonus/penalty band, for example:

$$
\text{perception adjustment} \in [-2, +2]
$$

Then:

$$
\text{final adjusted score} = \text{V10 score} + \text{perception adjustment}
$$

### Why not use the proposed raw formula directly?

Because these metrics live on different scales and confidence levels.

The following is too unstable initially:

$$
\text{final score} = \text{causality} + 2\cdot\text{coverage} + \text{retention} + \text{uptake} - \text{late injection} - \text{salience distortion} - \text{lock-in}
$$

It should instead be normalized into a bounded adjustment.

### Recommended adjustment

```python
def perception_adjustment(result: PerceptionIntegrityResult) -> float:
    raw = (
        0.8 * result.coverage
        + 0.5 * result.retention_score
        + 0.6 * result.causal_uptake_score
        - 0.7 * result.late_injection_score
        - 0.6 * result.salience_distortion
        - 0.6 * result.narrative_lock_in
    )
    return max(-2.0, min(2.0, 2.0 * (raw - 0.5)))
```

---

## Report integration

Add a new report section in [detective_engine/engine/case_runner.py](detective_engine/engine/case_runner.py):

- `Perception coverage`
- `Retention`
- `Late injection`
- `Causal uptake`
- `Salience distortion`
- `Narrative lock-in`
- `Strategic neutrality flag`
- recommended retry probes

Example output:

```text
— Perception Integrity —
  Coverage:            0.67
  Retention:           0.75
  Late injection:      0.50
  Causal uptake:       0.33
  Salience distortion: 0.71
  Narrative lock-in:   0.62
  ⚠ Strategic neutrality detected
  Suggested probes:
    • List 2 clues that weaken your current hypothesis.
    • Name 1 detail you noticed late and why.
```

---

## Longitudinal profile model

Extend [detective_engine/engine/user_profile.py](detective_engine/engine/user_profile.py).

Suggested additions:

```python
user_profile = {
    "perception_bias_history": [],
    "late_injection_trend": [],
    "causal_uptake_trend": [],
    "salience_distortion_trend": [],
    "narrative_lock_in_trend": [],
    "strategic_flags": [],
    "common_perception_failures": []
}
```

### Track over time

- repeated omission of contradiction anchors
- repeated attraction to emotionally vivid clues
- repeated late insertion of critical evidence
- repeated decorative inclusion without uptake
- repeated early hypothesis lock-in

This should allow recommendations like:

- “You usually notice dramatic clues before diagnostic ones.”
- “You often register disconfirming evidence too late.”
- “Your main weakness is not logic but early selective observation.”

---

## Implementation roadmap

## V11 — foundation

1. Add `EvidenceMeta` support to case schema
2. Add `ObservationEvent` and `PerceptionTrace`
3. Add early micro-capture in `case_runner.py`
4. Implement `perception_integrity.py`
5. Print perception metrics in report
6. Store metrics in user profile

## V11.5 — tutoring loop

1. Add `should_probe()`
2. Add adaptive probes after risky traces
3. Allow retry focused on missed perception patterns
4. Record whether retry improved coverage or uptake

## V12 — adversarial perception detection

1. Add strategic-neutrality and revision-laundering flags
2. Add bounded perception adjustment to final evaluation
3. Add longitudinal cognitive-integrity trends
4. Add case authoring support for bias-tag annotations

---

## Key safeguards

### 1. Do not overclaim

This system cannot prove inner honesty.

It can only detect whether the **trace of perception** is consistent or suspicious.

### 2. Keep perception separate from logic early

A user can reason well after perceiving poorly, and vice versa.

These should be separate metrics first.

### 3. Prefer stable IDs over text

Evidence matching must be ID-based wherever possible.

### 4. Keep human annotation uncertainty explicit

Use `salience_confidence` to avoid treating author metadata as objective truth.

---

## Final summary

V10 evaluates:

- reasoning integrity
- causality
- counter-evidence
- trap resistance

V11 / V12 should evaluate:

- what was noticed first
- what was forgotten
- what was inserted late
- what actually influenced reasoning
- whether “balanced observation” was genuine or performative

That moves the engine from **reasoning evaluator** toward **cognitive integrity evaluator**.
