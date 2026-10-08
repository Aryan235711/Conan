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
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="recompute activations instead of storing them; needed for long examples on a 16 GB Mac")
    ap.add_argument("--save-steps", type=int, default=None, help="checkpoint interval (default: half the run)")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --output")
    ap.add_argument("--fp16", action="store_true",
                    help="16-bit mixed precision instead of bf16, for GPUs without bf16 such as the T4")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly after this many hours and save a checkpoint (rerun with --resume to continue)")
    args = ap.parse_args()
    if args.smoke:
        args.max_steps, args.grad_accum, args.limit = 2, 1, 4
        args.output = str(ROOT / "runs" / "sft" / "smoke")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()]
    random.Random(args.seed).shuffle(rows)
    if args.limit:
        rows = rows[:args.limit]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])

    cuda = torch.cuda.is_available()
    # Load on CPU first: transformers' threaded loader segfaults on direct MPS placement.
    bf16 = cuda and not args.fp16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float32)
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
        save_steps=args.save_steps or max(50, args.max_steps // 2),
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=bf16,
        fp16=cuda and args.fp16,
        report_to=[],
        seed=args.seed,
    )
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    callbacks = []
    if torch.backends.mps.is_available():
        class _EmptyMPSCache(TrainerCallback):
            """Release cached MPS memory every step; without it memory builds up on long runs."""
            def on_step_end(self, *a, **k):
                torch.mps.empty_cache()
        callbacks.append(_EmptyMPSCache())
    if args.max_hours:
        import time
        deadline = time.time() + args.max_hours * 3600

        class _TimeLimit(TrainerCallback):
            """Stop at a step boundary once the time budget is spent (free GPU sessions are capped)."""
            def on_step_end(self, a, state, control, **k):
                if time.time() > deadline:
                    control.should_save = True
                    control.should_training_stop = True
        callbacks.append(_TimeLimit())
    trainer = SFTTrainer(model=model, args=config, train_dataset=ds, peft_config=peft_config, callbacks=callbacks)
    last = None
    if args.resume:
        ckpts = sorted(Path(args.output).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
        last = str(ckpts[-1]) if ckpts else None
        print(f"resuming from {last}" if last else "no checkpoint found; starting fresh")
    trainer.train(resume_from_checkpoint=last)
    if trainer.state.global_step < args.max_steps:
        print(f"stopped at step {trainer.state.global_step} of {args.max_steps} (time limit); rerun with --resume")
        return
    trainer.save_model(args.output)
    Path(args.output, "training_manifest.json").write_text(json.dumps(
        {"model": args.model, "examples": len(ds), "args": vars(args), "log_history": trainer.state.log_history},
        indent=2, default=str))
    print(f"saved adapter to {args.output}")


if __name__ == "__main__":
    main()
