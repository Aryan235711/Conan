# From keyword matching to a verifiable reward: rebuilding Conan

This report describes how Conan went from a keyword-matching reasoning
scorer to a verifiable environment that can be used as a reinforcement
learning reward, what was measured along the way, and what is still open.

## 1. The starting point and the audit

Conan began as a detective-reasoning engine. A user or model worked through
phases (observations, anomalies, hypotheses, elimination) and a rule engine
scored the text across four weighted pillars. Around it sat 494 generated
cases built from 15 templates and 1,500 training examples.

A forensic audit tested the scorer adversarially rather than reading it. The
main findings:

- **Keyword soup won.** A dump of case keywords with no reasoning scored
  higher than the project's own reference answer and passed with a B.
- **Correct answers were penalized.** Rules matched words anywhere in the
  text with no handling of negation. Stating the required conclusion, such as
  eliminating "no one touched the object", triggered both a forbidden-pattern
  penalty and an inference trap.
- **The ceiling was too low to pass.** LLM-judge points were always in the
  denominator even when the judge was off, and empty answers earned free
  points. Careful expert answers failed all six gold cases.
- **The generated cases leaked the answer.** In every generated case the
  hidden truth appeared word for word as the second scenario, and it was
  always the longest one.
- **The test split measured memorization.** 46% of test completions appeared
  word for word in training, and 93% matched once names were masked.
- **Two gold cases were wrong.** In C004, the stated rigor mortis pointed to
  a death after 01:00, supporting the witness the case said had lied. In
  C005, the "single plate" clue was exactly what the suspect's story
  predicted.
- **The LLM judge flipped verdicts.** Its fallback parser read "INCOHERENT"
  as COHERENT and "INVALID" as VALID.

A scorer with these properties cannot be used as a training reward: a policy
would learn to stuff keywords and avoid naming the scenarios it rejects.

## 2. Design principle: reward what can be checked

Detective cases have a useful property: most of what matters is checkable.
Who did it, which scenarios are impossible, which clues are decisive and when
the death happened all have exact answers. Version 2 asks the model to reason
freely and then end with a structured answer, and scores only that answer.

Each component was chosen to close a specific exploit:

- **Calibration** uses a Brier skill score relative to a uniform forecast, so
  hedging with even odds earns exactly zero.
- **Evidence and elimination** use F1, so citing every clue or ruling out
  every scenario loses precision.
- **Consistency checks** halve the reward when an answer contradicts itself,
  for example when the top pick is not the most probable scenario.
- **Unparseable output scores zero** instead of being skipped, so the policy
  cannot escape the reward through malformed output.

On the six gold cases, with identical answer types given to both scorers:

| | v1 keyword engine | v2 verifiable scorer |
|---|---|---|
| Correct pairwise orderings of expert, echo and wrong answers | 14 of 18 | 18 of 18 |
| Expert answers passing | 1 of 6 | 6 of 6 |
| Keyword dump | outscores experts | scores 0 |

The honest caveat is circularity: the gold answer keys and the expert
reference answers were written by the same author. The generated cases and
the model evaluations below do not have that problem.

## 3. Cases that are correct by construction

Fixing C004 and C005 by hand showed that hand-written logic is fragile. The
generator therefore works from a simulated world instead of templates:

1. Sample a victim, suspects, a true time of death, key holders and where
   each suspect verifiably was.
2. Write evidence from that world: body temperature with a stated cooling
   rate, the last sign of life, lock state, key holders, verified alibis,
   unverified testimony, motives, and deliberate red herrings.
3. Run a solver over the structured facts and keep the case only if exactly
   one scenario survives.

The answer key is derived, not written. A clue is "key evidence" if removing
it makes the answer ambiguous, which is a precise definition of decisive. Red
herrings are verified to change nothing. A second, independent solver that
checks every minute of the window re-verifies every case in the test suite;
all 2,600 cases in the four splits pass.

The out-of-distribution split draws names, places, alibi locations and
phrasings from a pool that never appears in training, at the hardest level.

## 4. Leakage probes

A benchmark is only as good as its resistance to shortcuts, so the harness
includes agents designed to exploit surface patterns.

The first probe picks the suspect named most often in the evidence. On the
first generator it scored about 60% against 23% chance: the culprit
systematically had more evidence lines. The generator was changed so every
suspect is named exactly three times (key-holder sentence, whereabouts,
motive), and the culprit and key-less suspects draw their whereabouts from the
same mix of unverified testimony, partial alibis and alibis that end before
the window. The probe now scores at chance, and a regression test enforces it.

The second probe treats any verified alibi as clearing a suspect. It still
beats chance because it applies one genuine reasoning step while ignoring
whether the alibi covers the whole window. It earns low reward and never
passes a case, which is the intended behaviour.

## 5. Results

All runs were on an Apple M2 laptop with 16 GB of memory, with greedy
decoding (temperature 0). Each model saw the same first 20 cases of each
generated split and the six gold cases. Accuracy is shown with a Wilson 95%
interval. Chance accuracy on the generated splits is about 23%.

### Baselines and probes (200 cases per split)

| Agent | In-distribution accuracy | Held-out accuracy | Mean reward, held-out |
|---|---|---|---|
| Solver upper bound | 100% (98-100) | 100% (98-100) | 1.00 |
| Uniform odds | 23% (18-29) | 20% (15-27) | 0.07 |
| Random | 20% (15-27) | 15% (11-21) | 0.06 |
| Most-mentioned suspect | 28% (22-35) | 23% (18-29) | 0.11 |
| Unverified-alibi heuristic | 39% (33-46) | 35% (29-42) | 0.17 |

No baseline passes a single case. The solver scoring 1.00 confirms that a
perfect answer reaches the scorer's ceiling.

### Models (20 cases per split)

| Model | Gold accuracy | In-distribution accuracy | Held-out accuracy | Held-out reward | Held-out valid format |
|---|---|---|---|---|---|
| Qwen2.5 0.5B Instruct, untrained | 33% (10-70) | 30% (15-52) | 5% (1-24) | 0.06 | 55% |
| Qwen2.5 0.5B, SFT | 17% (3-56) | 55% (34-74) | 45% (26-66) | 0.41 | 100% |
| Qwen2.5 Coder 7B, untrained | 67% (30-90) | 25% (11-47) | 5% (1-24) | 0.16 | 85% |
| DeepSeek-R1 8B | not completed | | | | |

**The warm-up transfers to held-out vocabulary.** The fine-tuned 0.5B model
went from 5% to 45% on the held-out split, and the intervals do not overlap.
It learned the method, not the names: held-out cases use names, places and
alibi phrasings that never appear in training. Valid output went from 55% to
100%, and the time-window score rose from 0.00 to 0.41.

**The 7B model reasons fluently but not correctly.** Qwen2.5 Coder 7B did
best on the six gold cases, whose prose resembles ordinary detective
fiction, but was at chance on the generated cases. Its time-of-death windows
scored near zero; in one case it subtracted eight hours from 23:14 and gave
a sixteen-hour window. In another it refused to commit, set its answer to
null and gave even odds after contradicting itself. Both earn zero or near
zero, as they should.

**The fine-tuned model's errors are bookkeeping, not arithmetic.** It usually
gets the time window exactly right. Its typical failure is losing track of
which name belongs to which scenario ID: it rules a suspect out twice under
two IDs and then names the same suspect as the culprit under a third. All 16
consistency violations in its test answers were of one kind, the same clue
cited both as decisive and as a red herring. This is the behaviour the RL
stage should target, since the reward pays only for correct conclusions and
halves self-contradictory answers.

**The warm-up is narrow.** On the six hand-written gold cases the fine-tuned
model did slightly worse than the untrained one. The method it learned is
specific to the generated case family.

**DeepSeek-R1 ran out of budget.** On its first case it spent its whole
3,072-token budget thinking and never produced an answer. A fair evaluation
needs about four times the budget, roughly 20 minutes per case on this
machine, so it was stopped and is reported as not completed.

### GRPO

GRPO was started from the fine-tuned adapter with 4 samples per case. It ran
for 12 steps before the laptop's memory was exhausted and each step slowed
from about 2 minutes to about 28 minutes, so it was stopped. The logged steps
show the loop working as intended: rewards ranged from 0.20 to 0.45 within
steps, which is the variance GRPO needs, and every sample produced valid
format. Twelve steps with one case each are far too few to show learning, and
no adapter from this run was evaluated. A real run needs a single GPU with
24 GB or more; the script is ready for it and now checkpoints frequently so
an interrupted run keeps its progress.

## 6. Training pipeline

- **SFT warm-up.** Reasoning traces are written from the solver's facts in a
  fixed method: time window, accident, access, alibis, red herrings,
  conclusion, final answer. All 2,000 training traces score 1.0.
- **GRPO.** TRL's GRPO trainer optimizes the verifiable reward with LoRA,
  plus a small format reward so early batches have variance. Each score
  component, the accuracy, the format rate and the violation rate are logged
  every step. Reward hacking would appear as a component falling while the
  total rises.
- **Evaluation.** Trained adapters are scored by the same harness as every
  other agent, on the same held-out splits.

## 7. Limitations and next steps

- One reasoning family. Time windows, access and alibis are covered; motive
  chains, physical mechanisms and testimony consistency need their own
  simulators.
- Scale. Local runs used small models and small samples on a laptop. A full
  GRPO run needs a GPU; the scripts are ready for one.
- Process quality. The verifiable reward checks conclusions and cited
  evidence, not the prose in between. Scoring the prose needs an LLM judge
  validated against human labels, which has not been done yet.
- Gold cases. The six hand-written cases are useful as a readable showcase
  but are too few for statistics.
