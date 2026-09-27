"""Expert iteration: reward-filtered self-training that fits on a 16 GB Mac.

GRPO holds generation and training in memory at once, which does not fit on
an M2 with 16 GB.  Expert iteration (also called rejection-sampling
fine-tuning) gets the core benefit of RL with a verifiable reward in separate,
memory-friendly phases:

    round r:
      1. sample   the current model writes K answers per training case
      2. filter   keep the best answer per case only if the verifiable scorer
                  passes it: correct, no consistency violations, reward >= 0.7
      3. train    fine-tune a fresh LoRA on all answers accepted so far,
                  starting from the round-0 model (STaR-style restart)
      4. merge    fold the adapter into a standalone model for the next round
      5. eval     score the new model on the fixed held-out split

The model only learns from its own answers that the reward says were right,
so it is pushed toward correct conclusions rather than just the format.

Each phase runs in its own process so memory is fully released between
phases, and every phase is resumable: rerun the same command after an
interruption and it continues where it stopped.

    caffeinate -i .venv-train/bin/python training/expert_iteration.py \\
        --init-adapter runs/sft/qwen0.5b --rounds 3 --cases-per-round 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine.engine.models import CaseDefinition  # noqa: E402
from detective_engine.engine.verifiable import build_prompt, score_answer  # noqa: E402

PY = sys.executable


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str]) -> None:
    log("$ " + " ".join(cmd))
    env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="1")
    subprocess.run(cmd, check=True, env=env, cwd=ROOT)


def load_raw(paths: list[str]) -> list[dict]:
    return [json.loads(l) for p in paths for l in open(p, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# Phase: sample + filter (runs in a subprocess)
# ---------------------------------------------------------------------------

def phase_sample(model_dir: str, cases_file: str, out_file: str, k: int, temperature: float,
                 max_new_tokens: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)          # CPU load first (MPS loader bug)
    model.to(device).eval()

    raws = load_raw([cases_file])
    done = set()
    if Path(out_file).exists():
        done = {json.loads(l)["case_id"] for l in open(out_file, encoding="utf-8") if l.strip()}
    todo = [r for r in raws if r["id"] not in done]
    log(f"sampling {len(todo)} cases x {k} ({len(done)} already done) with {model_dir}")

    for n, raw in enumerate(todo, 1):
        case = CaseDefinition.from_dict(raw)
        prompt = build_prompt(case)
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**ids, do_sample=True, temperature=temperature, top_p=0.95,
                                 num_return_sequences=k, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        texts = [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]
        results = [score_answer(case, t) for t in texts]
        best = max(range(k), key=lambda i: results[i].reward)
        rec = {
            "case_id": raw["id"],
            "family": raw["generator"].get("family", "timeline"),
            "rewards": [round(r.reward, 4) for r in results],
            "correct": [r.correct for r in results],
            "accepted": bool(results[best].passed),
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": texts[best]}],
            "seconds": round(time.time() - t0, 1),
        }
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if n % 10 == 0 or n == len(todo):
            recs = [json.loads(l) for l in open(out_file, encoding="utf-8")]
            acc = sum(r["accepted"] for r in recs)
            log(f"  {len(recs)} cases sampled, {acc} accepted ({acc / len(recs):.0%}), last case {rec['seconds']}s")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--init-adapter", default=str(ROOT / "runs" / "sft" / "qwen0.5b"),
                    help="SFT adapter that defines the round-0 model")
    ap.add_argument("--train", default=str(ROOT / "data" / "generated" / "train.jsonl"),
                    help="comma-separated training case files")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--cases-per-round", type=int, default=200)
    ap.add_argument("--k", type=int, default=4, help="samples per case")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=800)
    ap.add_argument("--epochs", type=float, default=2.0, help="passes over accepted answers per round")
    ap.add_argument("--eval-split", default="test_ood")
    ap.add_argument("--eval-limit", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "runs" / "ei" / "qwen0.5b"))
    ap.add_argument("--seed", type=int, default=0)
    # internal: single phase run in a subprocess
    ap.add_argument("--phase", choices=["sample"], default=None)
    ap.add_argument("--model-dir")
    ap.add_argument("--cases-file")
    ap.add_argument("--out-file")
    args = ap.parse_args()

    if args.phase == "sample":
        phase_sample(args.model_dir, args.cases_file, args.out_file, args.k, args.temperature, args.max_new_tokens)
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m0 = out / "m0"
    if not (m0 / "config.json").exists():
        run([PY, "training/merge_adapters.py", "--model", args.model, "--adapters", args.init_adapter, "--output", str(m0)])

    # Fixed, disjoint slices of training cases per round.
    raws = load_raw(args.train.split(","))
    random.Random(args.seed).shuffle(raws)
    current = m0
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"rounds": []}

    for r in range(1, args.rounds + 1):
        rdir = out / f"round{r}"
        rdir.mkdir(exist_ok=True)
        cases_file = rdir / "cases.jsonl"
        if not cases_file.exists():
            chunk = raws[(r - 1) * args.cases_per_round: r * args.cases_per_round]
            cases_file.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunk))

        # 1-2. sample and filter
        samples = rdir / "samples.jsonl"
        n_done = sum(1 for _ in open(samples)) if samples.exists() else 0
        if n_done < args.cases_per_round:
            run([PY, "training/expert_iteration.py", "--phase", "sample", "--model-dir", str(current),
                 "--cases-file", str(cases_file), "--out-file", str(samples), "--k", str(args.k),
                 "--temperature", str(args.temperature), "--max-new-tokens", str(args.max_new_tokens)])

        # accepted answers from all rounds so far
        accepted = []
        for rr in range(1, r + 1):
            f = out / f"round{rr}" / "samples.jsonl"
            accepted += [json.loads(l) for l in open(f) if l.strip() and json.loads(l)["accepted"]]
        acc_file = rdir / "accepted_all.jsonl"
        acc_file.write_text("".join(json.dumps({"prompt": a["prompt"], "completion": a["completion"]},
                                               ensure_ascii=False) + "\n" for a in accepted))
        recs = [json.loads(l) for l in open(samples)]
        stats = {
            "round": r,
            "cases": len(recs),
            "accepted_this_round": sum(x["accepted"] for x in recs),
            "any_correct_of_k": sum(any(x["correct"]) for x in recs),
            "mean_best_reward": round(sum(max(x["rewards"]) for x in recs) / len(recs), 4),
            "accepted_total": len(accepted),
        }
        log(f"round {r}: {stats}")
        if not accepted:
            log("no accepted answers; stopping. Try more samples per case (--k) or easier levels.")
            break

        # 3. train a fresh LoRA on all accepted answers, from the round-0 model
        adapter = rdir / "adapter"
        if not (adapter / "adapter_config.json").exists():
            steps = max(5, math.ceil(args.epochs * len(accepted) / 4))
            run([PY, "training/sft_train.py", "--model", str(m0), "--data", str(acc_file),
                 "--max-steps", str(steps), "--grad-accum", "4", "--lr", "1e-4", "--output", str(adapter)])

        # 4. merge
        merged = rdir / "merged"
        if not (merged / "config.json").exists():
            run([PY, "training/merge_adapters.py", "--model", str(m0), "--adapters", str(adapter),
                 "--output", str(merged)])

        # 5. evaluate on the fixed held-out split with the shared harness
        run([PY, "-m", "detective_engine.evaluate", "--agent", f"hf:{merged}", "--splits", args.eval_split,
             "--limit", str(args.eval_limit)])

        from detective_engine.evaluate import RUNS, _slug
        eval_file = RUNS / _slug(f"hf:{merged}") / f"{args.eval_split}.jsonl"
        if eval_file.exists():
            ev = [json.loads(l) for l in open(eval_file)]
            stats["eval_accuracy"] = round(sum(x["correct"] for x in ev) / len(ev), 4)
            stats["eval_reward"] = round(sum(x["reward"] for x in ev) / len(ev), 4)
            stats["eval_valid_format"] = round(sum(x["format_ok"] for x in ev) / len(ev), 4)
            stats["eval_n"] = len(ev)
        summary["rounds"] = [s for s in summary["rounds"] if s["round"] != r] + [stats]
        summary_path.write_text(json.dumps(summary, indent=2))
        log(f"round {r} done: {stats}")
        current = merged

    log("expert iteration finished; see " + str(summary_path))


if __name__ == "__main__":
    main()
