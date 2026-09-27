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

Two more families use the same recipe. In **which witness is lying**,
several witnesses describe where they were and whom they saw, reliable
records pin some people to places, and exactly one witness lies. The lie is
either contradicted by the liar's own record or hidden in an "I saw X" claim
that only a record about X contradicts. An independent backtracking search
over concrete worlds re-checks every case. The **combined** family chains
both: every key holder's alibi comes from a witness, the witness covering for
the culprit is lying, and a record exposes them. Solving it takes four steps
in order: fix the window, catch the liar, void their alibi, then apply access
and the remaining alibis. Combined cases average 29 evidence lines and nine
decisive clues. In every family each person is named the same number of
times, so counting names reveals nothing.

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
decoding (temperature 0). Accuracy is shown with a Wilson 95% interval.
Chance accuracy is about 23% on the in-distribution split and 16.7% on the
held-out split, which has six scenarios per case.

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

### Models on the held-out split (200 cases)

| Model | Accuracy | Accuracy when the answer parses | Valid format | Time-window score | Red-herring score | Mean reward |
|---|---|---|---|---|---|---|
| Qwen2.5 0.5B Instruct, untrained | 12.5% (9-18) | 22.3% | 56% | 0.01 | 0.08 | 0.05 |
| Qwen2.5 0.5B, SFT | 24.0% (19-30) | 24.7% | 97% | 0.41 | 0.41 | 0.26 |

### Models on smaller samples (20 cases per split, 6 gold cases)

| Model | Gold accuracy | In-distribution accuracy | Held-out accuracy |
|---|---|---|---|
| Qwen2.5 0.5B Instruct, untrained | 33% (10-70) | 30% (15-52) | 5% (1-24) |
| Qwen2.5 0.5B, SFT | 17% (3-56) | 55% (34-74) | 45% (26-66) |
| Qwen2.5 Coder 7B, untrained | 67% (30-90) | 25% (11-47) | 5% (1-24) |
| DeepSeek-R1 8B | not completed | | |

**A 20-case sample overstated the gain.** On the first 20 held-out cases the
fine-tuned model scored 45% against 5% for the untrained one, and an earlier
version of this report presented that as the result. On all 200 cases it
scores 24% against 12.5%. The first 20 were an unusually favourable sample
for the fine-tuned model and an unusually unfavourable one for the base
model. The lesson is to size evaluations before drawing conclusions.

**SFT teaches the format and the arithmetic, not the deduction.** Most of the
accuracy gain comes from producing a valid answer: valid format rises from
56% to 97%. Among answers that parse, accuracy moves only from 22.3% to
24.7%, a little above the 16.7% chance level. What the warm-up does teach
shows in the components: the time-window score rises from 0.01 to 0.41 and
the red-herring score from 0.08 to 0.41. The model learns to compute the
window and to set aside motives and noise, but not to carry the chain all the
way to the right suspect. Held-out names, places and phrasings do not hurt
it, so what it learned is the method rather than the vocabulary.

**The 7B model reasons fluently but not correctly.** Qwen2.5 Coder 7B did
best on the six gold cases, whose prose resembles ordinary detective
fiction, but was at chance on the generated cases. Its time-of-death windows
scored near zero; in one case it subtracted eight hours from 23:14 and gave
a sixteen-hour window. In another it refused to commit, set its answer to
null and gave even odds after contradicting itself. Both earn zero or near
zero, as they should.

**The fine-tuned model's errors are mostly bookkeeping.** Its windows usually
overlap the true window, but only 24 of its 194 parseable answers get the
window exactly. A typical failure is losing track of which name belongs to
which scenario ID: in one case it ruled a suspect out twice under two IDs and
then named the same suspect as the culprit under a third. 123 of its 200
answers break a consistency rule, and every one of them the same way: a clue
cited both as decisive and as a red herring. The untrained model breaks the
rules differently, most often by naming a culprit it has also ruled out
(104 of 200). This is the behaviour the RL stage should target, since the
reward pays only for correct conclusions and halves self-contradictory
answers.

**The warm-up is narrow.** On the six hand-written gold cases the fine-tuned
model did slightly worse than the untrained one. The method it learned is
specific to the generated case family.

**It does not transfer to a different reasoning type.** On 50 cases of the
which-witness-is-lying family, which uses the same vocabulary as training but
a different kind of deduction, the timeline-trained model scored 30% (19-44)
against 26% (16-40) untrained, with chance at 26%. Both are at chance. The
untrained model already produces valid answers on this family, so there is no
format gain to find either. Training on several families at once is the next
test of transfer.

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

### Expert iteration on the Mac

Without a GPU, reward-driven training was done with expert iteration: the
model samples four answers per training case, the verifiable scorer keeps the
best one only if it passes (correct, consistent, reward at least 0.7), and a
fresh LoRA is trained from the SFT model on every accepted answer so far.
Rounds 1 and 2 used 200 mixed-level training cases each; round 3 used 200
level-3 cases only, matching the held-out difficulty. Each round took about
five hours on the M2.

| Model (held-out, 200 cases) | Accuracy | Pass rate | Self-contradicting | Window score |
|---|---|---|---|---|
| SFT start | 24.0% (19-30) | 10.5% | 61% | 0.40 |
| Round 1 (85 accepted) | 21.5% (16-28) | 11.5% | 49% | 0.42 |
| Round 2 (194 accepted) | 24.0% (19-30) | 14.5% | 45% | 0.38 |
| Round 3, level 3 only (272 accepted) | 20.0% (15-26) | 10.0% | 48% | 0.32 |

Expert iteration made the model more consistent and, in round 2, fully right
more often, but it did not raise culprit accuracy on held-out cases.
Training only on hard cases made things slightly worse. On training cases the
model did improve: acceptance on unseen training cases rose from 43% in
round 1 to 55% in round 2.

### Where the reasoning breaks

`benchmarks/step_breakdown.py` checks every step of every answer against the
case's facts. For the SFT model and the best expert-iteration round:

| Step | SFT start | Round 2 |
|---|---|---|
| Time window right | 21% | 20% |
| Accident ruled out | 95% | 98% |
| All key-less suspects ruled out | 69% | 76% |
| All alibi-covered suspects ruled out | 55% | 46% |
| True culprit kept (not ruled out) | 27% | 25% |
| Final pick right, when every step was right | 100% | 100% |

The final pick is never the problem. The main failure is ruling out the true
culprit, and the breakdown shows why. When the culprit has a verified alibi
that does not cover the window, the SFT model still rules them out 74-83% of
the time, and after round 2 this rises to 84-85%. Getting the time window
right barely changes it (culprit kept 25% with the right window, 27% with a
wrong one). The model is not comparing alibi times with the window at all; it
has learned the shortcut "a verified alibi clears the suspect". Expert
iteration reinforced that shortcut, because it works on many easier training
cases.

This is the same shortcut as the unverified-alibi leakage probe, which scores
35% on this split. The models learned a weaker version of a shortcut rather
than the method. The next experiments target this directly: a micro-skill
test of whether the model can compare two time intervals at all, and
structured per-suspect fields so the reward can check each coverage decision
instead of only the final answer.

### Can the models compare two time intervals at all?

`benchmarks/interval_probe.py` isolates the one comparison the cases hinge
on: given a verified alibi and a death window, does the alibi cover the whole
window? Half the items are "yes"; the "no" items fail to cover in three ways
(ending before the window, ending inside it, starting inside it), and about
half cross midnight. Chance is 50%.

| Model | One-word answer, clock times | One-word answer, plain minutes | Step by step, clock times (120 items) |
|---|---|---|---|
| Qwen2.5 0.5B, untrained | 50%, always "yes" | 50%, always "yes" | 45% (36-54) |
| Qwen2.5 0.5B, SFT | 50%, always "yes" | 50%, always "yes" | 42% (34-51) |
| Qwen2.5 0.5B, expert iteration round 2 | 50%, always "yes" | 50%, always "yes" | 54% (45-63) |
| Qwen2.5 Coder 7B | 50%, always "no" | 50%, always "no" | 56% (47-64) |

No model can do it. Asked for one word, each answers the same way every
time. Given room to reason, all four stay near chance; the 7B model is no
better than the 0.5B ones. A typical step-by-step error inverts the rule:
"the alibi starts at 21:09, before the window opens at 23:13, so it does not
cover the window", when starting earlier is exactly what covering requires.

**The skill can be taught quickly.** `training/make_interval_data.py` writes
interval questions with worked answers that show every step: add 24 to the
hour of any time after midnight, check the alibi starts at or before the
window opens, check it ends at or after the window closes. The training
questions use three phrasings the probe never uses, and none of the probe's
items. After 18 minutes of LoRA training on 1,200 of them, the SFT 0.5B model
scores 99% (97-100) on the probe's 200 step-by-step items, up from 42%, with
no drop across midnight. A 0.5B model that is shown the work beats an
untrained 7B model by more than 40 points on this skill.

Two conclusions follow. First, a larger model alone will not fix the case
failures; the skill has to be taught. Second, the SFT traces were part of the
problem: they state conclusions such as "covering the whole window, so S4 is
ruled out" without showing the comparison, so a model can copy the
conclusion without learning to reach it. The next step is traces that show
every comparison explicitly (start before the window opens, end after it
closes, with times converted to minutes across midnight), a short curriculum
on the comparison itself, and per-suspect structured answers so the reward
checks each coverage decision.

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

- Three reasoning families. Timeline and access, which witness is lying,
  and a combination of the two are covered. Physical mechanisms and
  probabilistic cases like C006 still need their own simulators.
- Scale. Local runs used small models on a laptop. Only the held-out split
  was evaluated on 200 cases; other model numbers use 20 cases. A full GRPO
  run needs a GPU, and `training/gpu_run.sh` runs it end to end.
- Evaluation size. A 20-case sample overstated the SFT gain by nearly a
  factor of two. Claims should come from 200 cases or more.
- Process quality. The verifiable reward checks conclusions and cited
  evidence, not the prose in between. Scoring the prose needs an LLM judge
  validated against human labels, which has not been done yet.
- Gold cases. The six hand-written cases are useful as a readable showcase
  but are too few for statistics.
