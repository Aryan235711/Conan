"""Drive the Kaggle run from the Mac with the Kaggle CLI (no browser steps after setup).

One-time setup (the account owner does this in a browser):
  1. Kaggle → Settings → phone verification (needed for GPUs and internet).
  2. Kaggle → Settings → API → Create New Token, then move the downloaded
     kaggle.json to ~/.kaggle/kaggle.json and run: chmod 600 ~/.kaggle/kaggle.json

Then, from the repository root:

    .venv-kaggle/bin/python training/kaggle_push.py dataset    # upload data/kaggle/conan_kaggle.zip (private)
    .venv-kaggle/bin/python training/kaggle_push.py run        # start session 1 (smoke check, then train/eval)
    .venv-kaggle/bin/python training/kaggle_push.py status
    .venv-kaggle/bin/python training/kaggle_push.py resume     # next session: restores session N's output
    .venv-kaggle/bin/python training/kaggle_push.py fetch      # download results into runs/

Everything is created private. The private hand-written cases are never uploaded.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data" / "kaggle" / "push"
KAGGLE = str(Path(sys.executable).parent / "kaggle")
DATASET_SLUG = "conan-kaggle"
WHEELS_SLUG = "conan-wheels"          # Linux wheels of the exact library versions (data/kaggle/wheels)
ACCEL = os.environ.get("KAGGLE_ACCEL", "")        # empty: use enable_gpu from the metadata
MODEL_HANDLE = "qwen-lm/qwen2.5/transformers/1.5b-instruct/1"
STATE = WORK / "state.json"


def user() -> str:
    """The account name, from the Kaggle CLI's own configuration (token files are never read here)."""
    out = subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            return line.split(":", 1)[1].strip()
    sys.exit("no Kaggle credentials: put an API token in ~/.kaggle/access_token (Kaggle → Settings → API)")


def kaggle(*args: str) -> str:
    out = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    text = (out.stdout + out.stderr).strip()
    print(text)
    if out.returncode:
        sys.exit(f"kaggle {' '.join(args[:2])} failed")
    return text


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"session": 0}


def wheels() -> None:
    folder = WORK / "wheels"
    shutil.rmtree(folder, ignore_errors=True)
    shutil.copytree(ROOT / "data" / "kaggle" / "wheels", folder)
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": WHEELS_SLUG, "id": f"{user()}/{WHEELS_SLUG}", "licenses": [{"name": "other"}]}))
    mine = subprocess.run([KAGGLE, "datasets", "list", "--mine"], capture_output=True, text=True).stdout
    if f"{user()}/{WHEELS_SLUG}" in mine:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update")
    else:
        kaggle("datasets", "create", "-p", str(folder))


def dataset() -> None:
    folder = WORK / "dataset"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    with zipfile.ZipFile(ROOT / "data" / "kaggle" / "conan_kaggle.zip") as z:
        z.extractall(folder)                           # upload the folder tree, so paths are predictable
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": "conan-kaggle", "id": f"{user()}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}))
    mine = subprocess.run([KAGGLE, "datasets", "list", "--mine"], capture_output=True, text=True).stdout
    exists = f"{user()}/{DATASET_SLUG}" in mine
    if exists:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update", "-r", "tar")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-r", "tar")   # private by default


RUNNER = '''
import os, shutil, subprocess, sys, glob
# Offline by design: libraries come from the conan-wheels dataset and the model from Kaggle's model library.
os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
found = glob.glob("/kaggle/input/**/training/kaggle_run.py", recursive=True)
if not found:
    for root, dirs, files in os.walk("/kaggle/input"):
        print(root, dirs[:5], files[:5])
    sys.exit("training/kaggle_run.py not found under /kaggle/input")
repo = os.path.dirname(os.path.dirname(found[0]))
wheels = sorted(glob.glob("/kaggle/input/**/*.whl", recursive=True))
models = [os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True) if "1.5b-instruct" in c.lower()]
print("code:", repo, "| wheels:", len(wheels), "| model:", models[:1], flush=True)
if not wheels or not models:
    sys.exit("wheels or model missing from the inputs")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", *wheels])
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])  # Kaggle's old torchao breaks peft; unused here
subprocess.call([sys.executable, "-m", "pip", "check"])
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    sys.exit("no GPU in this session (is the account phone-verified and the accelerator set?)")
os.environ["MODEL"] = models[0]
shutil.copytree(repo, "/kaggle/working/Conan", dirs_exist_ok=True)
os.chdir("/kaggle/working/Conan")
if os.environ.get("SKIP_SMOKE") != "1" and not os.path.exists("runs/sft/qwen1.5b-all"):
    smoke = subprocess.call([sys.executable, "training/kaggle_run.py"], env={**os.environ, "SMOKE": "1"})
    if smoke:
        sys.exit("smoke check failed; see the log above")
    shutil.rmtree("runs/sft/smoke", ignore_errors=True)
    for d in glob.glob("runs/hf_runs_sft_smoke*"):
        shutil.rmtree(d)
    if os.path.exists("results.md"):
        os.remove("results.md")
subprocess.check_call([sys.executable, "training/kaggle_run.py"])
'''


def push(session: int, previous: str | None) -> None:
    folder = WORK / f"kernel{session}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    slug = f"conan15b-run-s{session}"   # fresh name: the first notebook was pinned to a CPU-only image
    (folder / "run.py").write_text(RUNNER)
    meta = {"id": f"{user()}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{user()}/{DATASET_SLUG}", f"{user()}/{WHEELS_SLUG}"],
            "kernel_sources": [previous] if previous else [], "competition_sources": [],
            "model_sources": [MODEL_HANDLE],
            "docker_image": "gcr.io/kaggle-gpu-images/python"}     # ask for the GPU image explicitly
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    kaggle("kernels", "push", "-p", str(folder), *(["--accelerator", ACCEL] if ACCEL else []))
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"session": session, "kernel": f"{user()}/{slug}"}))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    s = state()
    if cmd == "dataset":
        dataset()
    elif cmd == "wheels":
        wheels()
    elif cmd == "run":
        push(1, None)
    elif cmd == "resume":
        push(s["session"] + 1, s["kernel"])
    elif cmd == "status":
        kaggle("kernels", "status", s["kernel"])
    elif cmd == "fetch":
        out = WORK / f"output{s['session']}"
        out.mkdir(parents=True, exist_ok=True)
        kaggle("kernels", "output", s["kernel"], "-p", str(out))
        z = next(out.rglob("conan_results.zip"), None)
        if z:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ROOT)
            print(f"results unpacked into {ROOT / 'runs'}; table in {ROOT / 'results.md'}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
