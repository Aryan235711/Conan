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

### Explicit comparisons inside full cases

SFT v2 was trained from the base model on 600 curriculum items plus 400 case
traces that show every alibi comparison. Its answers are longer, so at the
usual 1,024-token budget 68% were cut off before the final answer; the
numbers below use a 2,048-token budget.

| Held-out, 200 cases | SFT v1 | SFT v2, explicit comparisons |
|---|---|---|
| Accuracy | 24.0% (19-30) | 23.5% (18-30) |
| Pass rate | 10.5% | 16.5% |
| Mean reward | 0.26 | 0.38 |
| Time window right | 21% | 59% |
| Self-contradicting answers | 61% | 32% |
| Trace keeps the culprit | 8% | 44% |
| Trace keeps the culprit, final answer drops them | 4% | 31% |
| Probe, step by step | 42% | 100% |

The explicit comparisons fixed the reasoning: the written trace now keeps the
culprit in 44% of answers instead of 8%, and the window is right three times
as often. Accuracy did not move because the gain is lost between the trace
and the final answer. In 31% of answers the trace says the culprit is not
cleared and the final answer rules them out anyway, often after a name mix-up
such as combining the first name of one suspect with the surname of another.
The next version adds explicit state tracking: after each step the trace
lists the suspects still possible, so the conclusion reads off the one left.

**SFT v4: compact traces with state tracking.** v4 used a shorter comparison
("Start 21:38 vs window 22:08: 21 < 22, so earlier. Starts in time: yes"),
listed red herrings by ID, and added "Still possible: ..." after each step.
Evaluated with a 1,280-token budget:

| Held-out, 200 cases | SFT v1 | SFT v2 | SFT v4 |
|---|---|---|---|
| Accuracy | 24.0% (19-30) | 23.5% (18-30) | 28.5% (23-35) |
| Pass rate | 10.5% | 16.5% | 19.0% |
| Valid format | 97% | 97% | 100% |
| Culprit kept in the final answer | 27% | 24% | 46% |
| Every elimination step right | 24% | 24% | 1% |
| Probe, step by step | 42% | 100% | 69% |

State tracking nearly doubled how often the culprit survives to the final
answer, and accuracy is the best so far, though the interval still overlaps
v1's. It also exposed new failures. The model invents alibi lines from other
facts (once treating the victim and the discovery time as an alibi), lets
the "still possible" list drift, and writes a final ruled-out list that
matches its own last list in only 12 of 200 answers. On the probe, 40 of its
61 misses are copying errors (the window start 23:05 written as 22:05): the
terse comparison is easier to miscopy than the verbose one, which scored
100%. With only 400 case traces in a mix that was 60% curriculum, the model
learned the format of tracking but not consistent bookkeeping. SFT v5 keeps
tracking, returns to the verbose comparison, and uses all 2,000 timeline
traces.

### SFT v5: verbose comparisons, state tracking, and enough examples

v5 combines what worked: the verbose comparison that names each value (100%
on the probe), the "still possible" list from v4, and all 2,000 timeline
traces instead of 400, plus 600 curriculum items. It trained for 650 steps
(4.5 hours on the M2) and was evaluated with a 1,792-token budget.

| Held-out, 200 cases | SFT v1 | SFT v4 | SFT v5 |
|---|---|---|---|
| Accuracy | 24.0% (19-30) | 28.5% (23-35) | **83.5% (78-88)** |
| Pass rate | 10.5% | 19.0% | 81.0% |
| Mean reward | 0.26 | 0.35 | 0.86 |
| Time-window score | 0.40 | 0.51 | 0.97 |
| Evidence score | 0.47 | 0.44 | 0.90 |
| Red-herring score | 0.39 | 0.35 | 0.81 |
| Self-contradicting answers | 61% | 23% | 4% |
| Every elimination step right | 24% | 1% | 83% |
| Culprit kept in the final answer | 27% | 46% | 83% |
| Probe, step by step | 42% | 69% | 100% |

This is the first decisive improvement: the intervals are far apart, and
every component of the reward rose together. Whenever v5 gets every
elimination step right, its final answer is right, as for every earlier
model; the difference is that it now gets the steps right 83% of the time.
The remaining errors are mostly alibi eliminations (84% of answers rule out
every covered suspect) and a residual 10% of answers where the trace keeps
the culprit and the final answer drops them.

Two checks guard against a false result. No test-set name appears anywhere
in the training data (0 of 191), and the only evidence lines shared between
training and test are generic sentences used by every case, such as the
injury that rules out an accident. A read of the answers confirms the model
follows the method: it fixes the window, applies access, works through each
alibi comparison, updates the list of remaining suspects, and concludes from
the last one standing.

What made the difference was diagnosis, not more training of the same kind.
Expert iteration on the original traces could not fix a skill the traces
never showed; the error breakdown located the failing step, the probe showed
the skill was missing in every model tested, and each later version changed
the training data to target a specific measured failure.

### SFT v6: the combined family

The combined family chains two skills: fix the time window, then catch the
witness who is lying to cover for the culprit before applying the alibis.
v5, trained only on timeline cases, carried the time skill over (window
score 0.91) but almost never looked for a liar: 3 of 100 answers did, and its
accuracy of 22% was near the 18% chance level.

v6 continued from v5 on 1,000 combined-family traces plus 500 timeline
traces as replay. The combined traces show the liar step explicitly: every
witness claim is checked against each reliable record about the witness or
the person they vouch for ("21:52 <= 22:27 <= 24:29, so inside the claimed
time; different place, so the claim is false"). Training took about 5 hours
on the M2 after two fixes for memory: gradient checkpointing and clearing the
MPS cache every step. The first attempt without them slowed from 50 to 400
seconds per step as memory filled.

| Held-out | v5 | v6 |
|---|---|---|
| Combined family: accuracy | 22% (15-31), 100 cases | **70.5% (64-76), 200 cases** |
| Combined family: pass rate | 8% | 67.5% |
| Combined family: self-contradicting | 65% | 6% |
| Combined family: answers that find the liar | 3 of 100 | 194 of 200 |
| Timeline family, same 100 cases: accuracy | 83% (74-89) | 89% (81-94) |

v6 learned the new skill and kept the old one: timeline accuracy on the same
cases did not fall, and if anything rose. Its errors on the combined family
are almost always complete misses rather than partial ones; its pass rate
(67.5%) is close to its accuracy (70.5%).

### The reliability suite

Twelve hand-written cases (see `benchmarks/reliability/README.md`): six in
the trained families but written as prose with traps, and six with reasoning
types never trained on.

| Model | Tier A, trained skills in prose | Tier B, new reasoning types | Valid format |
|---|---|---|---|
| SFT v5 | 0 of 6 | 1 of 6 | 10 of 12 |
| SFT v6 | 2 of 6 | 3 of 6 | 12 of 12 |

v6 does better, but prose is its weakest point. In three of its four tier A
misses it got the time window wrong: once it took a distracting 01:30
timestamp from a paragraph as the last sign of life, and twice it slipped by
an hour when adding the error margin. The fourth miss was a lying-witness case,
a family v6 was never trained on, where it picked the witness who behaved
nervously. With twelve cases these are qualitative observations, not
measurements, but they point directly at the next two versions: training on
the lying-witness family, and training on evidence written as varied prose.

### SFT v7: one model for all three families

v7 continued from v6 on 1,000 lying-witness traces, plus 300 combined and 300
timeline traces as replay. The lying-witness traces now expand every
statement into the places it implies ("I saw X at the café" places both the
speaker and X there) and check each against a record for the same person and
time.

| Held-out | v6 | v7 |
|---|---|---|
| Lying-witness family | 24% (17-33), 100 cases, chance 26% | **40% (33-47)**, 200 cases |
| Combined family, same 100 cases | 67% | **77% (68-84)** |
| Timeline family, 100 cases | 89% (81-94) | 84% (76-90) |
| Reliability suite | 5 of 12 | 2 of 12 |

v7 is one model that handles all three families, and it improved on both
witness families. The timeline dip is within the intervals. The reliability
suite fell from 5 to 2 of 12; with twelve cases a swing of three is largely
noise, but prose remains the weakest area for every version.

**Where the lying-witness family fails.** v7 never contradicts itself (0%
violations) and its pass rate equals its accuracy, so it follows the method
cleanly. Of its 120 wrong answers, 55 find no contradiction at all, reporting
"no record" for every statement, and 62 pair a statement with a record about a
different person or time. Only 2 find the right contradiction and then name
the wrong liar, and 1 invents a record. Accuracy is flat across difficulty
levels. The failing step is not reasoning but lookup: for each claim, finding
the one record about the same person at the same time among 15-20 evidence
lines. This is the same pattern as the alibi comparison, and the same remedy
applies: a short lookup curriculum with worked answers, and traces that first
index the records by person so each check consults a short list.

### SFT v7.1: teaching record lookup

The v7 diagnosis pointed at one step, so v7.1 targeted it the same way the
interval comparison was fixed. A lookup curriculum (3,000 items: a list of
reliable records with decoys, and a worked answer that lists the person's
records before deciding confirmed, contradicted or no record) was added, and
the lying-witness traces now group the records by person and quote that
person's short list at every claim check. v7.1 continued from v7 on 600
curriculum items, 1,000 indexed lying-witness traces, and 150 combined plus
150 timeline traces as replay.

| Held-out | v7 | v7.1 |
|---|---|---|
| Lookup probe, 100 items (chance about 33%) | 30% (22-40) | **82% (73-88)** |
| Lying-witness family, 200 cases | 40% (33-47) | **94.5% (90-97)** |
| Combined family, 100 cases | 77% (68-84) | 80% (71-87) |
| Timeline family, 100 cases | 84% (76-90) | 84% (76-90) |
| Reliability suite | 2 of 12 | 1 of 12 |

Teaching the one missing step took lying-witness accuracy from 40% to 94.5%,
with no loss on the other two families. On the probe, v7.1 almost always
confirms a true match (37 of 41) and catches a contradiction (40 of 41), but
it now over-finds records: when none exists it says so in only 5 of 18 items.

The reliability suite has fallen for three versions (v6 5, v7 2, v7.1 1 of
12). Twelve cases are noisy, but a steady decline is a signal: each round of
training on the generated format makes the model a little less robust to
evidence written as prose. Prose robustness is now the priority.

### SFT v9: prose robustness

v7.1's failures on the hand-written cases were misreadings, not reasoning
errors: a suspect's receipt time taken as the victim's last sign of life, a
claim checked against a record about someone else, a fact lost inside a
paragraph that held several. So v9 teaches reading.

- **Prose cases** (`detective_engine/prose.py`). Worlds from all three
  families are written as prose with traps: several facts per paragraph, an
  earlier sign of life beside the one that counts, an alibi extended by a
  second source, a receipt that is a single moment, a returned key. Each case
  is rebuilt by the reliability builder, which re-checks that the text states
  every fact and re-solves the answer. Phrasings come in two disjoint banks:
  training uses one, the held-out `prose_test` (150 cases, held-out names)
  uses the other.
- **A reading step.** Every trace now starts by listing what each paragraph
  states, then reasons as before. Combined-family traces also look up each
  person's records before checking a claim, as lying-witness traces do.
- **Lookup items with more "no record" answers**, the gap v7.1 left.

v9 continued from v7.1 on 1,600 prose traces, 450 templated traces and 300
lookup items (587 steps, about 9 hours on the laptop).

| Held-out | v7.1 | v9 |
|---|---|---|
| Prose test, 150 cases, unseen phrasing | 29% (22-36) | **43% (35-51)** |
| &nbsp;&nbsp;timeline / lying witness / combined, 50 each | 30% / 34% / 22% | 58% / 42% / 28% |
| Reliability suite, hand-written | 1 of 12 | **5 of 12** |
| Lookup probe | 82% ("no record" 5/18) | **90%** ("no record" 16/18) |
| Lying-witness family, 100 cases | 96% | 97% |
| Combined family, 100 cases | 80% | **92% (85-96)** |
| Timeline family, 100 cases | 84% | 85% |

v9 is the best model on every split, and the combined family gains most,
probably from the record lookup its traces now show. But prose is still far
behind the templated format (43% against 85-97%). Checking the reading step
against the correct reading shows why: only 39-49% of fact paragraphs are read
exactly right, and the errors concentrate on wording the training bank never
used ("vouches for", "season ticket scanned"), on the second fact of a merged
paragraph, and on losing track of paragraph numbers. With three to five
phrasings per fact type, v9 learned the phrasings rather than reading.

A caveat on the reliability suite: the traps were designed after reading
v7.1's failures on its tier A cases, so those cases are no longer fully blind.
The prose test, written in a bank the model never saw, is the clean measure.

Next: far more varied phrasing (a local model paraphrasing each paragraph,
kept only if every name, time and place survives), with the test bank still
held out.

### SFT v9.1: varied phrasing

Before changing anything, a diagnostic split asked whether v9's prose gap was
wording or something else: 60 new prose cases in v9's *training* phrasings,
with held-out names. v9 scored **85% (74-92)** there and read 91-99% of
paragraphs exactly, against 43% and 39-49% on the held-out phrasings. The gap
was entirely unseen wording.

v9.1 adds a phrasing grammar (`detective_engine/prose_grammar.py`) that builds
each sentence from interchangeable sources, verbs, time expressions and frames
(some give the end time before the start), merges up to three facts per
paragraph and adds filler sentences. It gives about five times as many
distinct paragraph shapes as v9's templates. A test forbids any four-word run
of a held-out test template in the new text, so the prose test stays unseen.
v9.1 continued from v9 on 1,800 such prose traces, 240 templated traces and
100 lookup items (535 steps, about 8.7 hours).

| Held-out | v9 | v9.1 |
|---|---|---|
| Prose test, 150 cases, unseen phrasing | 43% (35-51) | **66% (58-73)** |
| &nbsp;&nbsp;timeline / lying witness / combined, 50 each | 58% / 42% / 28% | 74% / 74% / 50% |
| &nbsp;&nbsp;paragraphs read exactly | 39-49% | **76-78%** |
| Prose in training phrasings, 60 cases | 85% | 87% |
| Reliability suite, tier A (prose, trained families) | 2 of 6 | **5 of 6** |
| Reliability suite, tier B (other reasoning types) | 3 of 6 | 0 of 6 |
| Lying-witness family, 100 cases | 97% | 98% |
| Combined family, 100 cases | 92% | 92% |
| Timeline family, 100 cases | 85% | 85% |
| Lookup probe | 90% | 90% |

Varied phrasing worked: the model now reads most paragraphs of wording it never
saw, and held-out prose accuracy rose by 23 points with no loss anywhere on the
templated families. The combined family, with 25 or more paragraphs per case,
is still the weakest at 50%.

Tier B went from 3 of 6 to 0 of 6. Those cases need reasoning no generator
teaches (lividity, timetables, stomach contents, two liars, probability), so
v9 was likely solving them by chance or general knowledge that further
training on the three families has pushed out. That is the job of the new
reasoning types planned for v11. Tier A's traps were designed after seeing
v7.1 fail them, so the prose test remains the cleaner measure.

### SFT v9.2: paraphrases from a second source

v9.1 solved about 97% of the prose cases whose decisive paragraphs it read
correctly, so reading was still the bottleneck. Its errors were sentence
structures the grammar never produced, plus a format problem: the reading step
listed key holders alphabetically, which a small model does badly.

v9.2 adds paraphrases from a local model (qwen2.5-coder 7B, run with ollama on
the laptop; `training/paraphrase_bank.py`). The model is never shown the held-out
test wording. A paraphrase is kept only if it passes rule checks (every
placeholder, no numbers, pronouns or stray first person, no four-word run of a
test template, claims must sound like claims and records like records) and a
round trip: filled with real values, a separate extraction prompt must recover
the fact type and every field exactly. On a control set of deliberately wrong
paraphrases the checker accepted none of 15. 497 of 694 candidates survived,
across 11 fact types. Key holders are now listed in the order the text names
them. v9.2 continued from v9.1 (547 steps, about 8 hours).

| Held-out | v9.1 | v9.2 |
|---|---|---|
| Prose test, 150 cases, unseen phrasing | 66% (58-73) | **72% (64-79)** |
| &nbsp;&nbsp;timeline / lying witness / combined, 50 each | 74% / 74% / 50% | 84% / 78% / 54% |
| &nbsp;&nbsp;paragraphs read right | 77-78% | 80-83% |
| Prose in training phrasings, 60 cases | 87% | **97%** |
| Reliability suite | 5 of 12 | 5 of 12 |
| Lying-witness family, 100 cases | 98% | 99% |
| Combined family, 100 cases | 92% | **97% (92-99)** |
| Timeline family, 100 cases | 85% | **91% (84-95)** |
| Lookup probe | 90% | 93% ("no record" 18/18) |

v9.2 is the best model on every split. The prose gain is smaller than v9.1's
and not yet certain: on the same 150 cases v9.2 alone was right on 25 and v9.1
alone on 16 (exact two-sided p = 0.21), though every family moved up. The
combined family is still the weakest on prose (54%) even though 81% of its
paragraphs are read right: with 25 or more paragraphs per case, one misread
decisive paragraph is enough to fail. Wording variety is giving less each
round (v9.1 +23 points, v9.2 about +6), so the next step should change
approach rather than add more wording.

### SFT v9.3: a reading curriculum

v9.2 solved 94% of prose cases whose decisive paragraphs it read right, and
its commonest misreading was an alibi claim read as the speaker's own claim
(the companion dropped) or with speaker and companion swapped. Reading is also
only about a tenth of the tokens in a full trace. So v9.3 gives reading its own
curriculum: 4,000 short items of one to four paragraphs with their reading
lines, weighted towards alibi claims, sightings and statements, plus a larger
verified paraphrase bank (a second generator, deepseek-r1 8B, under the same
checks; alibi claims 27 -> 161 patterns).

| Held-out | v9.2 | v9.3 |
|---|---|---|
| Prose test, 150 cases | 72% (64-79) | **75% (68-82)** |
| &nbsp;&nbsp;timeline / lying witness / combined | 84% / 78% / 54% | 84% / **88%** / 54% |
| &nbsp;&nbsp;alibi claims read right | 65% | **84%** |
| Prose in training phrasings, 60 cases | 97% | 92% |
| Lying-witness / combined / timeline, 100 each | 99% / 97% / 91% | 98% / 94% / 91% |
| Lookup probe | 93% | **71%** |
| Reliability suite | 5 of 12 | 3 of 12 |

The targeted skill improved a lot: alibi claims are read right 84% of the time
instead of 65%, and lying-witness prose rose to 88%. The overall prose gain is
small and uncertain (paired 19 vs 14, p = 0.49). Combined cases did not move
(54%): each has about nine decisive paragraphs, and at 90% per paragraph all
nine are read right only about a third of the time (16 of 50 here). Combined
prose needs near-perfect reading per paragraph, not just better reading.

v9.3 also regressed on record lookup (93% -> 71%): it lists the right records,
then answers "no record" even when one matches the time. Its training mix held
only 60 lookup items, all from a batch weighted towards "no record". v9.4 is a
short repair run with balanced lookup items and replay.

### SFT v9.4: repairing lookup

v9.4 is a short run from v9.3 (260 steps, half the learning rate) on 400 lookup
items at the original balance (20% "no record") with replay of reading items,
prose and templated traces.

| Held-out | v9.2 | v9.3 | v9.4 |
|---|---|---|---|
| Lookup probe | 93% | 71% | **93%** |
| Prose test, 150 cases | 72% | 75% | **77% (69-83)** |
| &nbsp;&nbsp;timeline / lying witness / combined | 84 / 78 / 54% | 84 / 88 / 54% | 80 / 84 / **66%** |
| Lying-witness family, 100 cases | 99% | 98% | 99% |
| Combined family, 100 cases | **97%** | 94% | 90% (83-94) |
| Timeline family, 100 cases | 91% | 91% | 94% |
| Prose in training phrasings, 60 cases | 97% | 92% | 92% |
| Reliability suite | 5 of 12 | 3 of 12 | 4 of 12 |

The repair worked: lookup is back to 93%, and v9.4 is the best model on
held-out prose (77%), with combined prose up from 54% to 66%. It is not a free
upgrade: on templated combined cases it lost 7 that v9.2 solved and gained none
(exact paired p = 0.016). The prose gains over v9.3 are within noise (14 vs 12
paired). Across v9.x, held-out prose rose from 29% (v7.1) to 77% while the
templated families stayed at 90-99%.

### SFT v9.5: consolidation, and a fresh prose test

v9.5 is a short replay-heavy run from v9.4 (300 templated combined traces plus
replay of every other skill, 287 steps, half learning rate) to win back the
templated combined cases v9.4 lost. Because prose_test had guided five rounds of
decisions, a second held-out prose test (prose_test2: new worlds and names, the
same held-out phrasing bank) was added before v9.5 was evaluated.

| Held-out | v9.4 | v9.5 |
|---|---|---|
| prose_test2, fresh, 150 cases | **80% (73-86)** | 76% (69-82) |
| &nbsp;&nbsp;timeline / lying witness / combined | 88 / 76 / 76% | 88 / 72 / 68% |
| prose_test (used for earlier decisions) | 77% | 83% |
| Combined family, 100 cases | 90% | **96% (90-98)** |
| Lying-witness / timeline, 100 each | 99% / 94% | 99% / 93% |
| Lookup probe | 93% | 92% |
| Reliability suite | 4 of 12 | 5 of 12 |

On the fresh test the two are equivalent (paired 12 vs 18, p = 0.36); v9.5's
83% on the original test did not replicate, which is what a test used for
selection does. The honest prose number for the v9 series is about 76-80% on
phrasing never trained on (v7.1: 29%). v9.5 recovers the templated combined
family, so it is the best all-round model and the base for v11.

### v11: how many witnesses are lying?

The lying-witness family always has exactly one liar, so a model can stop at
the first contradiction. The first new reasoning type (`detective_engine/multi_liar.py`)
states the number of liars in the rule, one or two, half the cases each, and
keeps a case only if exactly one set of that many witnesses fits the records
and the other statements. A two-liar case with five witnesses has ten
scenarios. It was designed from general principles; the private reliability
cases were not read.

| mliar_test (100 per kind) | one liar | two liars |
|---|---|---|
| v9.5, before training (50 each) | 49/50 | 11/50 |
| v11a: 1,000 k-liar traces + replay | 99 | 42 |
| v11b: one-step scenario lookup in traces | 99 | 36 |
| v11c: each scenario checked on its own line | 98 | **98 (93-99)** |

v11a found both liars in every two-liar case it got wrong; it failed to map the
pair onto the right scenario, because the names come out in a different order
than the scenario list. A trace step that listed "the scenarios naming X" in one
go made it worse: the model got that list wrong about three times in four.
Checking each scenario on its own line ("S3 (A and B): A yes, C no") took two
liars from 42% to 98%. The same lesson as the alibi comparison and the record
lookup: a small model fails at a search done in one step and succeeds when each
comparison is written out.

v11c kept the other families (combined 97%, lying witness 100%, lookup 89%,
fresh prose 76%) but leaned worse on timeline cases (87% vs v9.5's 93%, paired
4 vs 10, p = 0.18; timeline prose 38 vs 44 of 50). A short timeline-heavy
consolidation (v11d: 240 steps, half learning rate, 400 timeline traces plus
replay including 150 k-liar traces) recovered it and kept the new skill:

| Held-out | v9.5 | v11d |
|---|---|---|
| Two liars, 100 cases | 11 of 50 (before training) | **97** |
| One liar, 100 cases | 49 of 50 | 99 |
| Timeline family, 100 cases | 93% | **95% (89-98)** |
| Combined family, 100 cases | 96% | **98% (93-99)** |
| Lying-witness family, 100 cases | 99% | **100%** |
| prose_test2, fresh, 150 cases | 76% | **79% (71-84)** |
| Lookup probe | 92% | **95%** |
| Reliability suite | 5 of 12 | 4 of 12 |

v11d matches or beats v9.5 on every split (none of the paired differences is
large enough to be certain on its own) and adds the two-liar skill. It is the
best model so far.

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

Lessons carried forward from v1 to v6: show every decision's working in the
traces, target one measured failure per version, track the remaining suspects
explicitly, use enough examples, continue from the last good model with
replay of earlier skills, and evaluate on 200+ held-out cases plus the
reliability suite.

Planned versions:

- **v7:** one model for all three families, with record-by-record traces for
  the lying-witness family as well.
- **v8:** expert iteration on top of v7. On v1 it reinforced a shortcut; now
  that most answers reason correctly, filtering for correct and consistent
  answers should reinforce the right method.
- **v9:** robustness to prose: training on evidence rewritten as varied
  paragraphs with several facts each, checked so every fact survives.
- **v10:** knowing when not to conclude: imperfect records, mistaken
  witnesses, and a rewarded "evidence insufficient" answer.
- **v11:** new reasoning types (physical mechanisms, computed probabilities),
  then the same recipe on a 1.5B model with free GPU time, and a field test on
  historical solved cases used for evaluation only.


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
