# Running the 1.5B model on a free Kaggle GPU

The 0.5B model learns each reasoning type but reads unfamiliar wording at about
75-80%. This run repeats the same recipe once, from the base model, on
Qwen2.5 1.5B, to see how much of that gap is model size.

You need a Kaggle account with phone verification (required for GPU and
internet access in notebooks). The free quota is 30 GPU hours a week and 12
hours per session; this run needs roughly 8-10 hours of training and 2-3 of
evaluation, so it takes two sessions.

## 1. Build the bundle (on the Mac)

```bash
python3 training/make_consolidated.py      # data/kaggle/sft_all.jsonl, 6,900 examples
python3 training/make_kaggle_bundle.py     # data/kaggle/conan_kaggle.zip
```

`conan_private_eval.zip` (the hand-written reliability and spot-check cases) is
optional. Upload it only as a **private** dataset, or skip it and run those two
checks on the Mac afterwards.

## 2. Upload it

Kaggle → Datasets → New Dataset → drop `conan_kaggle.zip` → name it
`conan-kaggle` → keep visibility **Private** → Create.

## 3. Create the notebook

Kaggle → Code → New Notebook. In the right-hand panel:

- **Accelerator:** GPU T4 x2
- **Internet:** On
- **Add Input:** your `conan-kaggle` dataset

Paste these three cells:

```python
# Cell 1: copy the code to a writable folder
!cp -r /kaggle/input/conan-kaggle/Conan /kaggle/working/Conan 2>/dev/null || \
  (cd /kaggle/working && unzip -q /kaggle/input/conan-kaggle/conan_kaggle.zip)
%cd /kaggle/working/Conan
!pip install -q -U transformers trl peft datasets accelerate
```

```python
# Cell 2: five-minute check that everything runs (optional but recommended)
!SMOKE=1 python training/kaggle_run.py
```

```python
# Cell 3: the real run
!python training/kaggle_run.py
```

Run cells 1 and 2 interactively. If the check finishes with a results table,
click **Save Version → Save & Run All (Commit)**. That runs the notebook in the
background for up to 12 hours; you can close the browser.

## 4. Second session (finish training, evaluate)

Training stops by itself after 9.5 hours and saves a checkpoint. When the
committed run finishes:

1. Open the notebook again and click **Add Input → Notebook Output →** this
   notebook's latest version.
2. **Save Version → Save & Run All** again. The script restores the
   checkpoints, finishes training if needed, and evaluates.

Repeat once more if the log ends with "rerun in a new session".

## 5. Get the results

In the finished version's **Output** tab, download `conan_results.zip`. It
holds `results.md` (the table), every raw answer and score, and the trained
adapter. Unzip it into the repository on the Mac (it only adds files under
`runs/`) and the usual report and analysis scripts work on it.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` in training | Add `MAX_LENGTH=3600` before `python` in cell 3 (drops the longest fifth of combined cases from full length) |
| Model download fails | Check Internet is On and the account is phone-verified |
| `unexpected keyword argument 'dtype'` | The `pip install -U` in cell 1 did not run; rerun it |
| Evaluation did not finish | Run another session as in step 4; scored cases are skipped |

Settings can be changed with environment variables in cell 3, for example
`!HOURS=8 EVAL_N=60 python training/kaggle_run.py` (see the top of
`training/kaggle_run.py`).
