# Reliability suite

Hard, hand-written cases for checking whether a trained model is reliable
outside the generated format it was trained on.

The generated test splits share one template with the training data, so a
high score on them shows a model learned the method. It does not show the
model can handle a case written differently: prose paragraphs holding several
facts, distracting details, edge cases, or reasoning types it never trained
on. This suite covers that gap.

## Two tiers

**Tier A: verified by solver.** Cases in the trained families (timeline and
access, which witness is lying, and the combined family), written as prose
with deliberate traps. Every paragraph lists the structured facts it states,
and the builder:

- runs the family's solver and requires exactly one surviving scenario, equal
  to the author's intended answer;
- computes the decisive clues (paragraphs whose removal makes the answer
  ambiguous) and the death window;
- checks every paragraph flagged as a red herring changes nothing;
- checks every name, time and place in a paragraph's facts appears in its
  text, so the prose and the facts cannot drift apart.

**Tier B: manual or computed.** Reasoning types no solver covers yet, drawn
from real investigative techniques: lividity showing a body was moved, a dry
patch under a car dating its arrival, a timetable alibi that cannot work, time
of death from stomach contents, two liars at once, and a probability case.
Each carries a step-by-step proof for review. When a case states priors and
likelihoods, the builder computes the posteriors and requires the most
probable scenario to match the intended one.

All people and places are fictional. Real cases are used only as inspiration
for the kind of reasoning, never reproduced, because a case written from
memory could misstate facts about real victims and accused people.

## Build and run

```bash
python3 benchmarks/reliability/build.py                    # verify sources, write cases.jsonl
python3 -m detective_engine.evaluate --agent solver --splits reliability
.venv-train/bin/python -m detective_engine.evaluate --agent hf:runs/sft/qwen0.5b-v5 --splits reliability --max-new-tokens 2048
python3 tests/test_reliability.py                           # builder tests (run in CI)
```

On this split the `solver` agent answers from the verified key, as an oracle
ceiling. It scores 1.0 on every case except the probability case, where a
correctly calibrated 58% cannot earn full calibration credit.

## Adding a case

Create `source/<ID>.json`. Tier A fields:

| Field | Meaning |
|---|---|
| `id`, `title`, `summary`, `hidden_truth` | Case identity and the explanation |
| `tier`: `"A"`, `verification`: `"solver"` | Marks it for solver checks |
| `family` | `"timeline"`, `"liar"` or `"composite"` |
| `evidence` | List of `{"text", "facts", "herring"}` paragraphs |
| `scenarios` | List of `{"text", "label"}`; labels are suspect names or `"accident"`, witness names for `"liar"` |
| `intended` | Label of the true scenario |
| `witnesses` | Composite family only: the witness names |
| `n_places` | Liar family only: number of distinct places in the world |

Facts use the generator schema with times as `"HH:MM"`, for example
`{"kind": "alibi", "data": {"name": "Ann Lee", "start": "21:30", "end": "00:30"}}`.
State every fact's names, times and places in the paragraph text exactly as
written in the fact.

Tier B uses `verification: "manual"` or `"computed"`, an `answer_key` with
labels (`ruled_out`, `key_evidence`, `red_herrings`, optional `time_window`), a
`proof` list, and optionally a `bayes` block of priors and likelihoods.

Then run the builder. It refuses a case whose facts, prose or answer do not
line up.

## Keeping it private

`source/` and `cases.jsonl` are gitignored. A test set published on GitHub can
end up in future models' training data, which would make it useless as a test.
The builder, this guide and the tests are public; the cases stay local. Back
them up somewhere private.
