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
WHEELS_SLUG = "conan-wheels"
PRIVATE_SLUG = "conan-private-eval"   # hand-written reliability and spot-check cases (private dataset)          # Linux wheels of the exact library versions (data/kaggle/wheels)
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


def private() -> None:
    """Upload the hand-written cases as their own private dataset, so they are evaluated on Kaggle too."""
    folder = WORK / "private"
    shutil.rmtree(folder, ignore_errors=True)
    (folder / "eval").mkdir(parents=True)
    for src in (ROOT / "benchmarks" / "reliability" / "cases.jsonl", ROOT / "data" / "generated" / "spotcheck.jsonl"):
        shutil.copy2(src, folder / "eval" / src.name)
    (folder / "dataset-metadata.json").write_text(json.dumps(
        {"title": PRIVATE_SLUG, "id": f"{user()}/{PRIVATE_SLUG}", "licenses": [{"name": "other"}]}))
    mine = subprocess.run([KAGGLE, "datasets", "list", "--mine"], capture_output=True, text=True).stdout
    if f"{user()}/{PRIVATE_SLUG}" in mine:
        kaggle("datasets", "version", "-p", str(folder), "-m", "update", "-r", "tar")
    else:
        kaggle("datasets", "create", "-p", str(folder), "-r", "tar")     # private by default


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
import json
os.environ.update(json.loads('__RUN_ENV__'))      # settings for training/kaggle_run.py, filled in at push time
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
for name, dest in (("cases.jsonl", "benchmarks/reliability/cases.jsonl"), ("spotcheck.jsonl", "data/generated/spotcheck.jsonl")):
    hits = glob.glob(f"/kaggle/input/**/eval/{name}", recursive=True)    # private hand-written cases, if attached
    if hits:
        os.makedirs(os.path.dirname("/kaggle/working/Conan/" + dest), exist_ok=True)
        shutil.copy2(hits[0], "/kaggle/working/Conan/" + dest)
        print("hand-written cases attached:", name, flush=True)
os.chdir("/kaggle/working/Conan")
if os.environ.get("SKIP_SMOKE") != "1" and not os.path.exists("runs/sft/" + os.environ.get("NAME", "qwen1.5b-all")):
    smoke = subprocess.call([sys.executable, "training/kaggle_run.py"],
                            env={**os.environ, "SMOKE": "1", "NAME": "smoke", "STEPS": "3", "BASELINE_SPLITS": ""})
    if smoke:
        sys.exit("smoke check failed; see the log above")
    shutil.rmtree("runs/sft/smoke", ignore_errors=True)
    for d in glob.glob("runs/hf_runs_sft_smoke*"):
        shutil.rmtree(d)
    if os.path.exists("results.md"):
        os.remove("results.md")
subprocess.check_call([sys.executable, "training/kaggle_run.py"])
'''


PARAPHRASE_RUNNER = '''
import glob, json, os, shutil, subprocess, sys, time
# Paraphrase generation on Kaggle's GPU: install ollama, pull the models, run training/paraphrase_bank.py.
args = json.loads('__ARGS__')
src = glob.glob("/kaggle/input/**/training/paraphrase_bank.py", recursive=True)[0]
repo = os.path.dirname(os.path.dirname(src))
shutil.copytree(repo, "/kaggle/working/Conan", dirs_exist_ok=True)
os.chdir("/kaggle/working/Conan")
subprocess.check_call("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
server = subprocess.Popen(["ollama", "serve"])
time.sleep(10)
for model in {"qwen2.5-coder:7b", *[a for a in args if a.startswith(("deepseek", "qwen"))]}:
    subprocess.check_call(["ollama", "pull", model])
for run in args["runs"]:
    subprocess.check_call([sys.executable, "training/paraphrase_bank.py", "--out", args["out"], *run])
server.terminate()
print("PARAPHRASE_DONE")
'''


def paraphrase(out: str, runs: list[list[str]]) -> None:
    """Run paraphrase generation on Kaggle (internet on, GPU), starting from the bank already at `out`."""
    folder = WORK / "paraphrase"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    models = sorted({r[r.index("--gen-model") + 1] for r in runs if "--gen-model" in r})
    payload = {"out": out, "runs": runs}
    payload.update({m: True for m in models})
    (folder / "run.py").write_text(PARAPHRASE_RUNNER.replace("__ARGS__", json.dumps(payload)))
    slug = os.environ.get("KAGGLE_NAME", "conan-paraphrase")
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user()}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [f"{user()}/{DATASET_SLUG}"],
        "kernel_sources": [], "competition_sources": [], "model_sources": []}, indent=1))
    kaggle("kernels", "push", "-p", str(folder))
    STATE.write_text(json.dumps({"session": 0, "kernel": f"{user()}/{slug}", "paraphrase_out": out}))


def push(session: int, previous: str | None) -> None:
    folder = WORK / f"kernel{session}"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    slug = f"{os.environ.get('KAGGLE_NAME', 'conan15b-run')}-s{session}"   # fresh name: the first notebook was pinned to a CPU-only image
    env = {k[4:]: v for k, v in os.environ.items() if k.startswith("RUN_")}   # RUN_NAME=... -> NAME=...
    (folder / "run.py").write_text(RUNNER.replace("__RUN_ENV__", json.dumps(env)))
    meta = {"id": f"{user()}/{slug}", "title": slug, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{user()}/{DATASET_SLUG}", f"{user()}/{WHEELS_SLUG}"]
                               + ([f"{user()}/{PRIVATE_SLUG}"] if os.environ.get("WITH_PRIVATE") == "1" else []),
            "kernel_sources": [previous] if previous else [], "competition_sources": [],
            "model_sources": [MODEL_HANDLE],
            "docker_image": "gcr.io/kaggle-gpu-images/python"}     # ask for the GPU image explicitly
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    kaggle("kernels", "push", "-p", str(folder), *(["--accelerator", ACCEL] if ACCEL else []))
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"session": session, "kernel": f"{user()}/{slug}", "env": env}))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    s = state()
    if cmd == "dataset":
        dataset()
    elif cmd == "wheels":
        wheels()
    elif cmd == "private":
        private()
    elif cmd == "paraphrase":
        # usage: kaggle_push.py paraphrase OUT_PATH "ARGS FOR RUN 1" ["ARGS FOR RUN 2" ...]
        paraphrase(sys.argv[2], [a.split() for a in sys.argv[3:]])
    elif cmd == "run":
        push(1, None)
    elif cmd == "resume":
        push(s["session"] + 1, s["kernel"])
    elif cmd == "status":
        kaggle("kernels", "status", s["kernel"])
    elif cmd == "fetch" and "paraphrase_out" in s:
        out = WORK / "paraphrase_output"
        out.mkdir(parents=True, exist_ok=True)
        name = Path(s["paraphrase_out"]).name
        kaggle("kernels", "output", s["kernel"], "-p", str(out), "--file-pattern", f".*{name}$")
        got = next(out.rglob(name), None)
        if got:
            shutil.copy2(got, ROOT / s["paraphrase_out"])
            print(f"paraphrase bank updated: {s['paraphrase_out']}")
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
