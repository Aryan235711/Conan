"""Train and evaluate a larger model in one stage on a free Kaggle GPU (T4, 16 GB).

Run from the repository root inside a Kaggle notebook (see docs/KAGGLE.md):

    !python training/kaggle_run.py                # train, then evaluate
    !SMOKE=1 python training/kaggle_run.py        # five-minute check of the whole pipeline

What it does:
  1. Restores checkpoints and results from an earlier session, if that
     session's output was added to the notebook as an input.
  2. Trains Qwen2.5 1.5B once, from the base model, on the consolidated set
     (data/kaggle/sft_all.jsonl) with 16-bit precision. Training stops cleanly
     before the session limit and resumes in the next session.
  3. Evaluates on the held-out splits, eight cases at a time. Evaluation skips
     cases already scored, so it also resumes.
  4. Writes results.md and conan_results.zip next to the repository.

Environment variables: MODEL, NAME, STEPS, HOURS (training budget this
session, default 9.5), EVAL_N (cases per split, default 100), STAGE
(train | eval | all), SMOKE.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = os.environ.get("SMOKE") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct" if SMOKE else "Qwen/Qwen2.5-1.5B-Instruct")
NAME = os.environ.get("NAME", "smoke" if SMOKE else "qwen1.5b-all")
STEPS = int(os.environ.get("STEPS", 3 if SMOKE else 860))
HOURS = float(os.environ.get("HOURS", 9.5))
EVAL_N = int(os.environ.get("EVAL_N", 2 if SMOKE else 100))
STAGE = os.environ.get("STAGE", "all")
OUT = ROOT / "runs" / "sft" / NAME
DATA = ROOT / "data" / "kaggle" / "sft_all.jsonl"
SPLITS = ["mixed_test", "combo_test", "test_ood", "liar_test", "mliar_test", "prose_test2", "prose_mixed_test",
          "spotcheck", "reliability"]          # the last two only if the private bundle was added
START = time.time()


def sh(*cmd: str) -> int:
    print("+", " ".join(cmd), flush=True)
    return subprocess.call(list(cmd), cwd=ROOT)


def restore() -> None:
    """Copy runs/ from an earlier session's output (added as a notebook input), without overwriting."""
    for src in Path("/kaggle/input").glob("**/runs") if Path("/kaggle/input").exists() else []:
        n = 0
        for f in src.rglob("*"):
            dst = ROOT / "runs" / f.relative_to(src)
            if f.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
        print(f"restored {n} files from {src}")


def split_file(name: str) -> Path:
    if name == "reliability":
        return ROOT / "benchmarks" / "reliability" / "cases.jsonl"
    return ROOT / "data" / "generated" / f"{name}.jsonl"


def train() -> bool:
    if (OUT / "adapter_config.json").exists():
        print("training already finished")
        return True
    sh(sys.executable, "training/sft_train.py", "--model", MODEL, "--data", str(DATA),
       "--max-steps", str(STEPS), "--batch-size", "1", "--grad-accum", "8", "--lr", "1.5e-4",
       "--max-length", os.environ.get("MAX_LENGTH", "4200"), "--gradient-checkpointing", "--fp16",
       "--save-steps", "2" if SMOKE else "50", "--resume", "--max-hours", str(HOURS), "--output", str(OUT),
       *(["--limit", "24"] if SMOKE else []))
    done = (OUT / "adapter_config.json").exists()
    if not done:
        print("\nTraining is not finished. Save this version, then start a new session with this "
              "notebook's output added as an input, and run again.")
    return done


def evaluate() -> None:
    agent = f"hf:runs/sft/{NAME}"
    for s in SPLITS:
        if not split_file(s).exists():
            print(f"skip {s}: file not in this bundle")
            continue
        if time.time() - START > 11.2 * 3600:
            print("session time nearly used; rerun in a new session to finish evaluation")
            break
        n = 150 if s == "prose_test2" and not SMOKE else EVAL_N
        sh(sys.executable, "-m", "detective_engine.evaluate", "--agent", agent, "--splits", s, "--limit", str(n),
           "--max-new-tokens", "64" if SMOKE else ("2560" if s == "mliar_test" else "3072"), "--batch-size", "8")
    if not SMOKE:
        sh(sys.executable, "benchmarks/lookup_probe.py", "--agent", agent)


def package() -> None:
    report = subprocess.run([sys.executable, "-m", "detective_engine.evaluate", "--report"], cwd=ROOT,
                            capture_output=True, text=True).stdout
    (ROOT / "results.md").write_text(report)
    print(report)
    target = ROOT.parent / "conan_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "results.md", "results.md")
        for f in (ROOT / "runs").rglob("*"):
            # results, the final adapter and probe files; checkpoints stay in the notebook output for resuming
            if f.is_file() and "checkpoint-" not in str(f):
                z.write(f, f.relative_to(ROOT))
    print(f"wrote {target}")


def main() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")     # Kaggle offers two T4s; one process uses one
    restore()
    trained = train() if STAGE in ("train", "all") else (OUT / "adapter_config.json").exists()
    if trained and STAGE in ("eval", "all"):
        evaluate()
    package()


if __name__ == "__main__":
    main()
