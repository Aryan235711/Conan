"""GRPO fine-tuning against the verifiable detective reward.

Trains a small instruction model with LoRA using TRL's GRPOTrainer on
solver-verified generated cases.  The reward is the v2 verifiable scorer
(correctness, calibration, elimination, evidence, red herrings, time
window), plus a small format reward so early training has signal.

Every score component, the violation rate and the format rate are logged
as training metrics, so reward hacking shows up as a component moving the
wrong way while the total rises.

Setup (isolated env; the engine itself is stdlib-only):
    python3 -m venv .venv-train
    .venv-train/bin/pip install torch transformers trl peft datasets accelerate
    python3 -m detective_engine.generator --out data/generated

Local smoke test (Apple Silicon or CPU, minutes):
    .venv-train/bin/python training/grpo_train.py --smoke

Real run (one GPU with >= 24 GB, e.g. a rented A10/L4/A100):
    .venv-train/bin/python training/grpo_train.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --levels 1,2,3 --max-steps 600 \\
        --num-generations 8 --max-completion-length 1024 --output runs/grpo/qwen1.5b

Then evaluate the adapter with the same harness as every other agent:
    .venv-train/bin/python -m detective_engine.evaluate \\
        --agent hf:runs/grpo/qwen1.5b --splits test_id,test_ood --limit 100
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine.engine.verifiable import build_prompt, parse_final_answer, score_answer  # noqa: E402
from detective_engine.generator import load_cases  # noqa: E402

COMPONENTS = ("correctness", "calibration", "elimination", "evidence", "red_herrings", "time_window")


def _text(completion) -> str:
    """TRL passes plain strings or chat messages depending on the dataset format."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        return completion[-1].get("content", "")
    return str(completion)


def make_reward_functions(cases_by_id: dict):
    def verifiable_reward(completions, case_id, log_metric=None, **kwargs):
        rewards, stats = [], {c: [] for c in COMPONENTS}
        violations = fmt_ok = correct = 0
        for comp, cid in zip(completions, case_id):
            res = score_answer(cases_by_id[cid], _text(comp))
            rewards.append(res.reward)
            fmt_ok += res.format_ok
            correct += res.correct
            violations += bool(res.violations)
            for c in COMPONENTS:
                if c in res.components:
                    stats[c].append(res.components[c])
        if log_metric is not None and rewards:
            n = len(rewards)
            log_metric("verifiable/format_ok", fmt_ok / n)
            log_metric("verifiable/accuracy", correct / n)
            log_metric("verifiable/violation_rate", violations / n)
            for c, v in stats.items():
                if v:
                    log_metric(f"verifiable/{c}", sum(v) / len(v))
        return rewards

    def format_reward(completions, **kwargs):
        out = []
        for comp in completions:
            parsed = parse_final_answer(_text(comp))
            out.append(1.0 if isinstance(parsed, dict) and parsed.get("most_likely") else 0.0)
        return out

    return verifiable_reward, format_reward


def build_dataset(path: Path, levels: set[int], limit: int | None, seed: int):
    from datasets import Dataset

    raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    raw = [r for r in raw if r.get("generator", {}).get("level") in levels]
    random.Random(seed).shuffle(raw)
    if limit:
        raw = raw[:limit]
    cases = {c.id: c for c in load_cases(path)}
    rows = [{"prompt": [{"role": "user", "content": build_prompt(cases[r["id"]])}],
             "case_id": r["id"], "level": r["generator"]["level"]} for r in raw]
    return Dataset.from_list(rows), cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--init-adapter", default=None, help="merge this SFT LoRA adapter into the base before GRPO")
    ap.add_argument("--train", default=str(ROOT / "data" / "generated" / "train.jsonl"))
    ap.add_argument("--levels", default="1", help="comma-separated difficulty levels, e.g. 1 or 1,2,3")
    ap.add_argument("--limit", type=int, default=None, help="max training cases")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8, help="per-device prompts x generations per step")
    ap.add_argument("--max-completion-length", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.0, help="KL coefficient")
    ap.add_argument("--format-weight", type=float, default=0.1)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--output", default=str(ROOT / "runs" / "grpo" / "run"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny local run to check the pipeline end to end")
    args = ap.parse_args()

    if args.smoke:
        args.max_steps, args.num_generations, args.batch_size = 2, 2, 2
        args.max_completion_length, args.limit = 96, 8
        args.output = str(ROOT / "runs" / "grpo" / "smoke")

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    levels = {int(x) for x in args.levels.split(",")}
    dataset, cases = build_dataset(Path(args.train), levels, args.limit, args.seed)
    verifiable_reward, format_reward = make_reward_functions(cases)

    cuda = torch.cuda.is_available()
    config = GRPOConfig(
        output_dir=args.output,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        beta=args.beta,
        temperature=1.0,
        reward_weights=[1.0, args.format_weight],
        logging_steps=1,
        save_steps=max(50, args.max_steps // 4),
        bf16=cuda,
        use_cpu=not cuda and not torch.backends.mps.is_available(),
        log_completions=True,
        report_to=[],
        seed=args.seed,
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")

    # Load on CPU first and let the trainer place it: transformers' threaded
    # loader segfaults when materializing weights directly onto Apple MPS.
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if cuda else torch.float32)
    processing_class = None
    if args.init_adapter:
        from peft import PeftModel
        from transformers import AutoTokenizer
        model = PeftModel.from_pretrained(model, args.init_adapter).merge_and_unload()
        processing_class = AutoTokenizer.from_pretrained(args.model)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[verifiable_reward, format_reward],
        args=config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processing_class,
    )
    trainer.train()
    trainer.save_model(args.output)

    train_bytes = Path(args.train).read_bytes()
    manifest = {
        "model": args.model, "init_adapter": args.init_adapter, "levels": sorted(levels), "cases": len(dataset),
        "train_sha256": hashlib.sha256(train_bytes).hexdigest(),
        "args": vars(args),
        "log_history": trainer.state.log_history,
    }
    Path(args.output, "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"saved adapter and manifest to {args.output}")


if __name__ == "__main__":
    main()
