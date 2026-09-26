"""Evaluation harness — run agents on gold and generated cases, score with v2.

Agents:
    uniform           even odds, picks S1, commits to nothing
    random            random scenario with random probabilities (seeded)
    mentions          picks the suspect named most often in the evidence;
                      a leakage probe: if it wins, surface frequency gives the answer away
    unverified        picks a suspect whose alibi is only testimony; a second leakage probe
    solver            reads the generator's structured facts (generated cases
                      only); an upper bound that validates the scorer's ceiling
    ollama:<model>    a local model through the Ollama HTTP API
    hf:<path|repo>    a Hugging Face model or trained LoRA adapter (needs torch)

Every raw output and score is saved to runs/<agent>/<split>.jsonl so results
can be audited and interrupted runs resume where they stopped.

Examples:
    python3 -m detective_engine.evaluate --agent uniform --splits gold,test_id,test_ood
    python3 -m detective_engine.evaluate --agent ollama:qwen2.5-coder:7b --splits gold,test_ood --limit 20
    python3 -m detective_engine.evaluate --report
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .engine.case_loader import CaseLoader
from .engine.models import CaseDefinition
from .engine.verifiable import build_prompt, score_answer
from .generator import death_window, fmt, load_cases, solve, Fact

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DATA = ROOT / "data" / "generated"


# ---------------------------------------------------------------------------
# Case sources
# ---------------------------------------------------------------------------

def load_split(name: str) -> list[CaseDefinition]:
    if name == "gold":
        cases, _ = CaseLoader().load_all()
        return sorted([c for c in cases if c.answer_key is not None], key=lambda c: c.id)
    path = DATA / f"{name}.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found. Run: python3 -m detective_engine.generator")
    return load_cases(path)


def _raw_generated(name: str) -> dict[str, dict]:
    path = DATA / f"{name}.jsonl"
    if name == "gold" or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {d["id"]: d for d in (json.loads(l) for l in f if l.strip())}


# ---------------------------------------------------------------------------
# Agents: callable(case, prompt, raw_case_dict) -> answer (str or dict)
# ---------------------------------------------------------------------------

def _n(case: CaseDefinition) -> int:
    return len(case.scenarios)


def agent_uniform(case, prompt, raw):
    n = _n(case)
    return {"most_likely": "S1", "probabilities": {f"S{i}": 1 / n for i in range(1, n + 1)},
            "ruled_out": [], "key_evidence": [], "red_herrings": []}


def make_agent_random(seed: int = 0):
    rng = random.Random(seed)

    def agent(case, prompt, raw):
        n = _n(case)
        w = [rng.random() for _ in range(n)]
        s = sum(w)
        probs = {f"S{i}": w[i - 1] / s for i in range(1, n + 1)}
        top = max(probs, key=probs.get)
        return {"most_likely": top, "probabilities": probs, "ruled_out": [],
                "key_evidence": [], "red_herrings": []}
    return agent


def agent_mentions(case, prompt, raw):
    """Pick the scenario whose named person appears in the most evidence lines."""
    counts = {}
    for i, s in enumerate(case.scenarios, 1):
        name = s.split(",")[0].strip()
        counts[f"S{i}"] = sum(1 for e in case.evidence if name and name in e) if " " in name else -1
    top = max(counts, key=counts.get)
    n = _n(case)
    probs = {f"S{i}": (0.6 if f"S{i}" == top else 0.4 / (n - 1)) for i in range(1, n + 1)}
    return {"most_likely": top, "probabilities": probs, "ruled_out": [], "key_evidence": [], "red_herrings": []}


def agent_unverified(case, prompt, raw):
    """Leakage probe: pick a suspect whose only whereabouts line is unverified testimony."""
    says = []
    for i, s in enumerate(case.scenarios, 1):
        name = s.split(",")[0].strip()
        if " " in name and any(e.startswith(f"{name} says") for e in case.evidence):
            says.append(f"S{i}")
    top = says[0] if says else "S1"
    n = _n(case)
    probs = {f"S{i}": (0.6 if f"S{i}" == top else 0.4 / (n - 1)) for i in range(1, n + 1)}
    return {"most_likely": top, "probabilities": probs, "ruled_out": [], "key_evidence": [], "red_herrings": []}


def agent_solver(case, prompt, raw):
    """Upper bound: solves from the generator's structured facts."""
    if not raw:
        return agent_uniform(case, prompt, raw)
    facts = [Fact(f["kind"], f["text"], f["data"], f["role"]) for f in raw["generator"]["facts"]]
    labels = []
    for s in case.scenarios:
        labels.append("accident" if s.endswith("was an accident.") else s.split(",")[0].strip())
    suspects = [l for l in labels if l != "accident"]
    sol = solve(facts, suspects, "accident" in labels)
    top_label = next(iter(sol))
    top = f"S{labels.index(top_label) + 1}"
    lo, hi = death_window(facts)
    key = []
    for i, f in enumerate(facts, 1):
        if solve([g for g in facts if g is not f], suspects, "accident" in labels) != sol:
            key.append(f"E{i}")
    herr = [f"E{i}" for i, f in enumerate(facts, 1) if f.kind == "salient"
            or (f.kind == "motive" and f.data.get("name") != top_label)]
    return {"most_likely": top, "probabilities": {f"S{i}": float(f"S{i}" == top) for i in range(1, len(labels) + 1)},
            "ruled_out": [f"S{i}" for i in range(1, len(labels) + 1) if f"S{i}" != top],
            "key_evidence": key, "red_herrings": herr,
            "time_window": {"earliest": fmt(lo), "latest": fmt(hi)}}


def make_agent_ollama(model: str, host: str = "http://localhost:11434", timeout: int = 900,
                      num_predict: int = 3072):
    def agent(case, prompt, raw):
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": num_predict, "seed": 0},
        }).encode()
        req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        msg = data.get("message", {})
        text = msg.get("content", "")
        if msg.get("thinking"):
            text = f"<think>{msg['thinking']}</think>\n{text}"
        return text
    return agent


def make_agent_hf(path: str, max_new_tokens: int = 1024):
    """A local Hugging Face model or a LoRA adapter directory (greedy decoding).

    Needs torch + transformers (+ peft for adapters), e.g. the .venv-train
    environment described in training/grpo_train.py.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    adapter_cfg = Path(path) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel
        base = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(base)
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), path).merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
    model.to(device).eval()

    def agent(case, prompt, raw):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return agent


def get_agent(spec: str) -> Callable:
    if spec == "uniform":
        return agent_uniform
    if spec == "random":
        return make_agent_random()
    if spec == "mentions":
        return agent_mentions
    if spec == "unverified":
        return agent_unverified
    if spec == "solver":
        return agent_solver
    if spec.startswith("ollama:"):
        return make_agent_ollama(spec.split(":", 1)[1])
    if spec.startswith("hf:"):
        return make_agent_hf(spec.split(":", 1)[1])
    sys.exit(f"unknown agent: {spec}")


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------

def _slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec)


def run(spec: str, splits: list[str], limit: int | None) -> None:
    agent = get_agent(spec)
    for split in splits:
        cases = load_split(split)
        if limit:
            cases = cases[:limit]
        raw_by_id = _raw_generated(split)
        out = RUNS / _slug(spec) / f"{split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            with open(out, encoding="utf-8") as f:
                done = {json.loads(l)["case_id"] for l in f if l.strip()}
        todo = [c for c in cases if c.id not in done]
        if spec == "solver" and not raw_by_id:
            print(f"[{spec}] {split}: skipped (solver needs generated facts)")
            continue
        print(f"[{spec}] {split}: {len(todo)} to run ({len(done)} already done)", flush=True)
        for i, case in enumerate(todo, 1):
            prompt = build_prompt(case)
            t0 = time.time()
            try:
                answer = agent(case, prompt, raw_by_id.get(case.id))
                error = None
            except Exception as exc:  # network / model failures are recorded, scored 0
                answer, error = "", f"{type(exc).__name__}: {exc}"
            elapsed = time.time() - t0
            res = score_answer(case, answer)
            rec = {"case_id": case.id, "split": split, "agent": spec,
                   "level": (raw_by_id.get(case.id) or {}).get("generator", {}).get("level"),
                   "seconds": round(elapsed, 2), "error": error,
                   "raw_output": answer if isinstance(answer, str) else json.dumps(answer),
                   **res.to_dict()}
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if spec.startswith(("ollama:", "hf:")):
                print(f"  {i}/{len(todo)} {case.id} reward={res.reward:.2f} correct={res.correct} "
                      f"format_ok={res.format_ok} {elapsed:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(records: list[dict]) -> dict[str, Any]:
    n = len(records)
    k = sum(r["correct"] for r in records)
    lo, hi = wilson(k, n)
    comps = collections.defaultdict(list)
    for r in records:
        for name, v in r["components"].items():
            comps[name].append(v)
    return {
        "n": n, "accuracy": k / n if n else 0.0, "acc_ci": (lo, hi),
        "reward": sum(r["reward"] for r in records) / n if n else 0.0,
        "passed": sum(r["passed"] for r in records) / n if n else 0.0,
        "format_ok": sum(r["format_ok"] for r in records) / n if n else 0.0,
        "components": {c: sum(v) / len(v) for c, v in comps.items()},
    }


def report(runs_dir: Path = RUNS) -> str:
    rows = []
    for agent_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else []:
        for f in sorted(agent_dir.glob("*.jsonl")):
            with open(f, encoding="utf-8") as fh:
                recs = [json.loads(l) for l in fh if l.strip()]
            if recs:
                rows.append((recs[0]["agent"], f.stem, summarize(recs)))
    order = {"gold": 0, "test_id": 1, "test_ood": 2, "val": 3, "train": 4}
    rows.sort(key=lambda r: (order.get(r[1], 9), r[0]))
    lines = ["| split | agent | n | accuracy (95% CI) | mean reward | pass rate | valid format |",
             "|---|---|---|---|---|---|---|"]
    for agent, split, s in rows:
        lo, hi = s["acc_ci"]
        lines.append(f"| {split} | {agent} | {s['n']} | {s['accuracy']:.2f} ({lo:.2f}-{hi:.2f}) | "
                     f"{s['reward']:.3f} | {s['passed']:.2f} | {s['format_ok']:.2f} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate agents on detective cases with the verifiable scorer.")
    ap.add_argument("--agent", help="uniform | random | mentions | unverified | solver | ollama:<model> | hf:<path>")
    ap.add_argument("--splits", default="gold,test_id,test_ood")
    ap.add_argument("--limit", type=int, default=None, help="max cases per split")
    ap.add_argument("--report", action="store_true", help="print a markdown summary of runs/")
    args = ap.parse_args()
    if args.agent:
        run(args.agent, [s.strip() for s in args.splits.split(",") if s.strip()], args.limit)
    if args.report or not args.agent:
        print(report())


if __name__ == "__main__":
    main()
