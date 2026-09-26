"""Supervised warm-up (LoRA) on solver-written reasoning traces.

Teaches a small model the method and the final-answer format so that GRPO
starts with non-zero reward variance.  Loss is computed on the assistant
completion only.

    python3 training/make_sft_data.py
    .venv-train/bin/python training/sft_train.py --max-steps 150 --output runs/sft/qwen0.5b
    .venv-train/bin/python -m detective_engine.evaluate --agent hf:runs/sft/qwen0.5b --splits test_id,test_ood --limit 20
    .venv-train/bin/python training/grpo_train.py --init-adapter runs/sft/qwen0.5b ...
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(ROOT / "data" / "generated" / "sft_train.jsonl"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--output", default=str(ROOT / "runs" / "sft" / "run"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.grad_accum, args.limit = 2, 1, 4
        args.output = str(ROOT / "runs" / "sft" / "smoke")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

    cuda = torch.cuda.is_available()
    # Load on CPU first: transformers' threaded loader segfaults on direct MPS placement.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if cuda else torch.float32)
    config = SFTConfig(
        output_dir=args.output,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, args.max_steps // 20),
        max_length=args.max_length,
        logging_steps=5,
        save_steps=max(50, args.max_steps // 2),
        bf16=cuda,
        report_to=[],
        seed=args.seed,
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds, peft_config=peft_config)
    trainer.train()
    trainer.save_model(args.output)
    Path(args.output, "training_manifest.json").write_text(json.dumps(
        {"model": args.model, "examples": len(ds), "args": vars(args), "log_history": trainer.state.log_history},
        indent=2, default=str))
    print(f"saved adapter to {args.output}")


if __name__ == "__main__":
    main()
