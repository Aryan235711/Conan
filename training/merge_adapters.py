"""Merge a chain of LoRA adapters into a base model and save a standalone model.

GRPO trains its adapter on top of the SFT-merged weights, but the saved
adapter records the original base model as its parent.  Loading it alone
would silently drop the SFT stage, so evaluation merges the whole chain in
order first:

    .venv-train/bin/python training/merge_adapters.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --adapters runs/sft/qwen1.5b,runs/grpo/qwen1.5b \\
        --output runs/merged/qwen1.5b-grpo
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapters", required=True, help="comma-separated adapter dirs, applied in order")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)   # CPU load; see sft_train.py
    for path in [p for p in args.adapters.split(",") if p]:
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
        print(f"merged {path}")
    model.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.model).save_pretrained(args.output)
    print(f"saved merged model to {args.output}")


if __name__ == "__main__":
    main()
