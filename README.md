# Conan: a verifiable reasoning environment for detective cases

[![tests](https://github.com/Aryan235711/Conan/actions/workflows/tests.yml/badge.svg)](https://github.com/Aryan235711/Conan/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![MIT License](https://img.shields.io/badge/license-MIT-green.svg)

Conan is a benchmark and reinforcement-learning environment for multi-step
deductive reasoning. A model reads a case (evidence, suspects, scenarios),
reasons in free text, and ends with a structured answer. The answer is scored
against a machine-checkable key: which scenario is true, which ones the
evidence rules out, which clues are decisive, which are red herrings, and
when the death happened.

The core library is pure Python standard library. Training uses PyTorch,
Transformers, TRL and PEFT in a separate environment.

## Results

Held-out split: 200 cases with names, places and phrasing never seen in
training, at the hardest difficulty. Chance accuracy is 16.7%. Everything
ran on an Apple M2 laptop with 16 GB of memory; 95% Wilson intervals in
brackets.

| Model | Accuracy | Pass rate | Mean reward | Self-contradicting |
|---|---|---|---|---|
| Solver upper bound | 100% | 100% | 1.00 | 0% |
| Qwen2.5 0.5B Instruct, untrained | 12.5% (9-18) | 0% | 0.05 | 53% |
| Qwen2.5 0.5B, first SFT | 24.0% (19-30) | 10.5% | 0.26 | 61% |
| Qwen2.5 0.5B, three rounds of expert iteration | 24.0% (19-30) | 14.5% | 0.29 | 45% |
| **Qwen2.5 0.5B, SFT v5** | **83.5% (78-88)** | **81.0%** | **0.86** | **4%** |
| Qwen2.5 0.5B, SFT v6 (timeline, 100 cases) | 89% (81-94) | 86% | 0.90 | 3% |
| Qwen2.5 0.5B, SFT v7 (timeline, 100 cases) | 84% (76-90) | 82% | | 3% |
| Qwen2.5 0.5B, SFT v7.1 (timeline, 100 cases) | 84% (76-90) | 82% | | 4% |
| Qwen2.5 0.5B, SFT v9 (timeline, 100 cases) | 85% (77-91) | 83% | | 3% |
| Qwen2.5 0.5B, SFT v9.1 (timeline, 100 cases) | 85% (77-91) | 85% | | 0% |
| Qwen2.5 0.5B, SFT v9.2 (timeline, 100 cases) | 91% (84-95) | 87% | | 4% |
| Qwen2.5 0.5B, SFT v9.4 (timeline, 100 cases) | 94% (88-97) | | | 1% |
| Qwen2.5 Coder 7B, untrained (20 cases) | 5% (1-24) | 0% | 0.16 | |

On the combined family, where the model must catch a lying witness before
the timeline works, v5 scored 22% (chance 18%) and looked for a liar in 3 of
100 answers. SFT v6, trained on traces that check every witness claim against
the records, scores **70.5% (64-76)** on 200 held-out cases and finds the liar
in 194 of 200, without losing timeline accuracy. On twelve hand-written prose
cases with traps it solves 5, up from 1 for v5; prose remains the weakest point.

SFT v7 is one model for all three families: lying-witness cases 40% (33-47),
up from 24% (chance 26%); combined cases 77% on the same 100 cases v6 scored
67% on; timeline 84%. On lying-witness cases it follows the method without
contradicting itself, but fails to look up the right record for a claim, the
next skill to teach.

SFT v7.1 teaches that lookup with a short curriculum and traces that group
records by person: a lookup probe rises from 30% to 82%, and lying-witness
accuracy from 40% to **94.5% (90-97)** on 200 held-out cases, with combined
(80%) and timeline (84%) unchanged or better. One model now scores 80-95% on
all three generated families. The hand-written prose cases are the open
problem: 1 of 12 for v7.1, down from 5 for v6.

SFT v9 trains on the same families written as prose (verified by the same
builder) and starts every answer with a reading step. On 150 held-out prose
cases in phrasing it never saw, accuracy rises from 29% to **43% (35-51)**, and
the hand-written cases from 1 to 5 of 12. It is also the best model on the
templated families: lying witness 97%, combined **92%**, timeline 85% (100
cases each). Prose remains the gap: the reading step is exact on fewer than
half of the paragraphs, failing mostly on unfamiliar wording.

A diagnostic showed the gap was wording alone: in its training phrasings v9
scores 85% on new prose cases. SFT v9.1 adds a phrasing grammar (about five
times more sentence shapes, test phrasing excluded) and raises held-out prose
accuracy to **66% (58-73)**, reading 76-78% of unseen paragraphs exactly, with
the templated families unchanged (lying witness 98%, combined 92%, timeline
85%). It solves 5 of 6 hand-written prose cases in the trained families but
none of the 6 that need other reasoning types.

SFT v9.2 adds paraphrases written by a local 7B model and kept only after a
round-trip check recovers every fact exactly (the checker rejected all 15
deliberately wrong paraphrases in a control set). Held-out prose rises to
**72% (64-79)**, and v9.2 is the best model on every split: lying witness 99%,
combined **97%**, timeline **91%** (100 cases each). Long combined cases in
prose (54%) are the remaining gap.

SFT v9.3 adds a reading curriculum (short paragraph-reading items) and a larger
verified paraphrase bank: alibi claims are read right 84% of the time instead
of 65%, held-out prose reaches **75% (68-82)**, and lying-witness prose 88%.
Combined prose stays at 54%, and record lookup regressed (93% -> 71%); a short
repair run (v9.4) restores lookup to 93% and reaches **77% (69-83)** on
held-out prose (combined prose 66%), at a cost on templated combined cases
(97% -> 90%, paired p = 0.016). Held-out prose has gone from 29% (v7.1) to 77%.
SFT v9.5, a short consolidation run, recovers templated combined cases (96%)
and keeps prose; on a fresh, untouched prose test v9.4 and v9.5 score 80% and
76% (a tie within noise). That 76-80% is the honest prose figure for v9.

v11 adds the first new reasoning type: the rule states how many witnesses lie
(one or two). Before training, two-liar cases scored 11 of 50; after v11c, whose
traces check each scenario on its own line, **98 of 100**, with one-liar cases
at 98 and the other families held (combined 97%, lying witness 100%). Timeline
cases dipped (87% vs 93%); a short consolidation run is next.

The path to 83.5% came from diagnosis rather than more training:

1. A step-by-step error analysis showed models ruled out the real culprit
   about three times in four, because they treated any verified alibi as
   clearing a suspect without checking its times against the death window.
2. A micro-skill probe showed no model tested, up to 7B, could decide
   whether an alibi covers a time window (42-56%, chance 50%).
3. A short curriculum of worked comparisons taught the skill: 42% to 99% on
   the probe after 18 minutes of training.
4. Case traces were rewritten to show every comparison, and to list the
   suspects still possible after each step so the conclusion reads off the
   last one standing.
5. With 2,000 such traces the model gets every elimination step right in 83%
   of held-out cases, and when every step is right the answer is right.

Expert iteration, a reward-filtered self-training loop that fits on a laptop,
made the first model more consistent but did not raise accuracy: it
reinforced the alibi shortcut instead of fixing it. Full details, including
a 20-case result that was later corrected, are in [docs/REPORT.md](docs/REPORT.md).

## Why a verifiable reward

The first version of this project scored reasoning by keyword overlap. An
audit showed that a dump of case keywords with no reasoning outscored careful
answers, that correct answers were penalized for naming the scenario they
rejected, and that generated cases leaked their answer in the scenario list.
A policy trained on that signal would learn to stuff keywords.

Version 2 scores what can be checked exactly:

| Component | What it measures | Why it resists gaming |
|---|---|---|
| Correctness | picked the true scenario | binary |
| Calibration | Brier skill score against a uniform forecast | even odds score 0 |
| Elimination | F1 of ruled-out scenarios | ruling out everything destroys precision |
| Evidence | F1 of cited decisive clues | citing every clue destroys precision |
| Red herrings | F1 of flagged distractors | only scored where the case defines them |
| Time window | overlap with the true time-of-death window | a vague window scores near 0 |

Answers that contradict themselves are halved per violation, for example a
top pick that is not the most probable scenario. Output with no valid final
answer scores 0 rather than being skipped, so a trained policy cannot escape
the reward by emitting garbage.

## Cases that are correct by construction

Hand-written cases are few and can be wrong: two of the six original gold
cases had logic that pointed to a different answer than their own key. The
generators therefore simulate a ground-truth world, write evidence from it,
and run a solver. A case is kept only if exactly one scenario survives. The
answer key is computed rather than written: a clue is "key evidence" if
removing it makes the answer ambiguous, and every red herring is verified to
change nothing. Every family has a second, independent solver in the tests.

| Family | What it takes | Example trap |
|---|---|---|
| Timeline and access | fix the death window from body cooling, apply key access, check which verified alibis cover the whole window | an alibi that ends inside the window |
| Which witness is lying | check every statement against reliable records and the other statements | a lie hidden in an "I saw X" claim that only a record about X contradicts |
| Combined | fix the window, catch the lying witness, void their alibi, then apply access and alibis | the liar's own receipt half-corroborates the lie |

Combined cases average 29 evidence lines and nine decisive clues, which is
closer to the hand-written gold cases than either family alone.

**Leakage probes.** An early version let a model win by picking the suspect
named most often, which scored about 60% against 23% chance. Every person is
now named the same number of times, and innocent and guilty suspects draw
their whereabouts from the same mix. The probe now scores at chance in every
family, and regression tests keep it there.

## Quick start

```bash
python3 main.py --validate                                # check all case files
python3 tests/run_all.py                                  # run every test script
python3 benchmarks/scorer_ranking.py                      # v1 vs v2 scorer on gold cases

python3 -m detective_engine.generator --out data/generated   # all families and splits
python3 -m detective_engine.evaluate --agent solver --splits test_ood,liar_test,combo_test
python3 -m detective_engine.evaluate --agent ollama:qwen2.5-coder:7b --splits gold,test_ood --limit 20
python3 -m detective_engine.evaluate --report             # markdown results table
```

Scoring an answer from Python:

```python
from detective_engine import VerifiableScorer

scorer = VerifiableScorer()
prompt = scorer.prompt("C004")          # evidence E1..En, scenarios S1..Sn, answer format
result = scorer.score("C004", model_output_text)
print(result.reward, result.correct, result.components, result.feedback)
```

## Training

The whole pipeline runs from one script on a single GPU with 24 GB or more:
data for all three families, the base-model evaluation, SFT, evaluation,
GRPO, merging the adapters, evaluation, and the results table.

```bash
bash training/gpu_run.sh                 # full run on a GPU, several hours
SMOKE=1 bash training/gpu_run.sh         # tiny version that checks every stage
MODEL=Qwen/Qwen2.5-3B-Instruct NAME=qwen3b bash training/gpu_run.sh
```

Without a GPU, `training/expert_iteration.py` runs reward-filtered
self-training on a 16 GB Mac: the model samples several answers per case, the
verifiable scorer keeps only correct and consistent ones, and the model is
fine-tuned on them, round after round. Each phase runs in its own process so
memory is released between phases.

The individual steps are `training/make_sft_data.py`, `sft_train.py`,
`grpo_train.py` and `merge_adapters.py`. A short supervised warm-up teaches
the answer format so GRPO starts with reward variance; GRPO then optimizes
the verifiable reward directly. Every score component, accuracy, format rate
and violation rate is logged per step, so reward hacking shows up as a
component falling while the total rises.

## Repository layout

```
detective_engine/
  engine/verifiable.py      v2 scorer, prompt builder, answer parser
  engine/models.py          case and answer-key data model
  generator.py              timeline family, split builder
  liar.py                   witness-consistency family
  composite.py              combined family
  evaluate.py               agents, run logging, Wilson intervals, report
  engine/*.py               v1 process-feedback engine (keyword based; not used as reward)
  cases/C001-C006           hand-written gold cases with answer keys
training/                   traces, SFT, GRPO, adapter merging, gpu_run.sh
benchmarks/                 v1 vs v2 scorer ranking benchmark
tests/                      run with tests/run_all.py
docs/REPORT.md              write-up: audit, design, results, limitations
```

## Limitations

- The 83.5% result is for the timeline family, which the model was trained on
  with new names, places and phrasing held out. It has not yet been trained or
  tested on the lying-witness or combined families.
- The model learned a procedure from solver-written traces. That is the point
  of the exercise, but it means the traces define the method it follows.
- Three reasoning families so far. Physical mechanisms and probabilistic
  cases like C006 still need their own simulators.
- The six gold cases are hand-written and their answer keys were written by
  the same person who wrote the reference answers in the benchmark.
- The v1 engine still scores reasoning process by keywords. It is kept for
  written feedback only and is not a reliable measure.
- Local results come from a laptop and small models. The 7B and in-distribution
  numbers use 20 cases and carry wide intervals.

## License

MIT. See [LICENSE](LICENSE).
