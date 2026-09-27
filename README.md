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

Local runs on an Apple M2 laptop with 16 GB of memory. Generated splits use
20 cases per model, so 95% intervals are wide and are shown in the report.
Chance accuracy is about 23%.

| Agent | In-distribution accuracy | Held-out accuracy | Held-out reward |
|---|---|---|---|
| Solver upper bound | 100% | 100% | 1.00 |
| Qwen2.5 0.5B Instruct, untrained | 30% | 5% | 0.06 |
| Qwen2.5 0.5B, SFT on solver traces | **55%** | **45%** | **0.41** |
| Qwen2.5 Coder 7B via Ollama, untrained | 25% | 5% | 0.16 |
| Most-mentioned suspect, leakage probe | 28% | 23% | 0.11 |

A short supervised warm-up takes a 0.5B model from 5% to 45% on held-out
cases with unseen names, places and phrasing, and beats an untrained 7B model.
The remaining errors are mostly bookkeeping: the model computes the time
window correctly but loses track of which name belongs to which scenario.
GRPO on top of the warm-up ran for 12 steps before the laptop ran out of
memory; the pipeline works, and a full run needs a GPU. Details, intervals
and failure analysis are in [docs/REPORT.md](docs/REPORT.md).

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
generator therefore simulates a ground-truth world (victim, suspects, true
time of death, key holders, verified and unverified alibis), writes evidence
from it, and runs a solver. A case is kept only if exactly one scenario
survives the evidence. The answer key is computed rather than written:

- **Ruled out** is every scenario the solver excludes.
- **Key evidence** is every clue whose removal makes the answer ambiguous.
- **Red herrings** are generated distractors, each verified to change nothing.
- **Time window** comes from body cooling and the last sign of life.

A second, independent minute-by-minute solver re-checks every case in the
test suite. Three difficulty levels vary the number of suspects, red
herrings, an accident scenario, a partial-alibi trap and how tight alibis
are. The out-of-distribution test split uses names, places and phrasings
that never appear in training.

**Leakage probes.** An early version let a model win by picking the suspect
named most often, which scored about 60% against 23% chance. Every suspect is
now named exactly three times, and innocent and guilty suspects draw their
whereabouts from the same mix, so only the timing separates them. The
most-mentioned probe now scores at chance, and a regression test keeps it
there.

## Quick start

```bash
python3 main.py --validate                                # check all case files
python3 tests/run_all.py                                  # run every test script
python3 benchmarks/scorer_ranking.py                      # v1 vs v2 scorer on gold cases

python3 -m detective_engine.generator --out data/generated   # 2,600 verified cases
python3 -m detective_engine.evaluate --agent solver --splits test_id,test_ood
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

```bash
python3 -m venv .venv-train
.venv-train/bin/pip install torch transformers trl peft datasets accelerate

python3 training/make_sft_data.py                          # solver-written reasoning traces
.venv-train/bin/python training/sft_train.py --output runs/sft/qwen0.5b
.venv-train/bin/python training/grpo_train.py --init-adapter runs/sft/qwen0.5b --levels 1,2,3
.venv-train/bin/python -m detective_engine.evaluate --agent hf:runs/sft/qwen0.5b --splits test_id,test_ood
```

A short supervised warm-up on solver-written traces teaches a small model the
method and answer format. GRPO then optimizes the verifiable reward directly;
it needs reward variance within each group of samples, which a model that
never produces a valid answer cannot provide. Every score component,
accuracy, format rate and violation rate is logged per step, so reward
hacking shows up as a component falling while the total rises.

## Repository layout

```
detective_engine/
  engine/verifiable.py      v2 scorer, prompt builder, answer parser
  engine/models.py          case and answer-key data model
  generator.py              world simulator, solver, split builder
  evaluate.py               agents, run logging, Wilson intervals, report
  engine/*.py               v1 process-feedback engine (keyword based; not used as reward)
  cases/C001-C006           hand-written gold cases with answer keys
training/                   SFT trace builder, SFT and GRPO scripts
benchmarks/                 v1 vs v2 scorer ranking benchmark
tests/                      run with tests/run_all.py
docs/REPORT.md              write-up: audit, design, results, limitations
```

## Limitations

- Generated cases cover one reasoning family: time windows, access and
  alibis. Other deduction types need their own simulators.
- The six gold cases are hand-written and their answer keys were written by
  the same person who wrote the reference answers in the benchmark.
- The v1 engine still scores reasoning process by keywords. It is kept for
  written feedback only and is not a reliable measure.
- Local model results come from small samples on a laptop and carry wide
  confidence intervals, which are reported.

## License

MIT. See [LICENSE](LICENSE).
