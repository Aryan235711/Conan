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
training, at the hardest difficulty. Chance accuracy is 16.7%. Local runs on
an Apple M2 laptop with 16 GB of memory; 95% Wilson intervals in brackets.

| Agent | Cases | Accuracy | Valid format | Mean reward |
|---|---|---|---|---|
| Solver upper bound | 200 | 100% | 100% | 1.00 |
| Most-mentioned suspect, leakage probe | 200 | 23% (18-29) | 100% | 0.11 |
| Qwen2.5 0.5B Instruct, untrained | 200 | 12.5% (9-18) | 56% | 0.05 |
| Qwen2.5 0.5B, SFT on solver traces | 200 | 24% (19-30) | 97% | 0.26 |
| Qwen2.5 Coder 7B via Ollama, untrained | 20 | 5% (1-24) | 85% | 0.16 |

Supervised fine-tuning on solver-written traces doubles accuracy, but most
of that gain is format: among answers that parse, accuracy moves only from
22% to 25%, a little above chance. What SFT does teach is the time
arithmetic (time-window score 0.01 to 0.41) and spotting red herrings (0.08
to 0.41). Picking the culprit, which needs every step to be right, is still
unsolved at this size. That is the job of the RL stage, since the verifiable
reward pays for the correct conclusion directly. The timeline-trained model
also shows no transfer to the which-witness-is-lying family, where it scores
30% against 26% chance on 50 cases. A first run on 20 cases
suggested 45% for the SFT model; the 200-case run corrected it, and both are
documented in the report.

GRPO ran for 12 steps before the laptop ran out of memory. The pipeline works
end to end; a full run needs a GPU. See [docs/REPORT.md](docs/REPORT.md).

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
