#!/usr/bin/env bash
# End-to-end training and evaluation run for one GPU (24 GB or more).
#
#   bash training/gpu_run.sh            # full run, several hours
#   SMOKE=1 bash training/gpu_run.sh    # tiny version to check the pipeline
#
# Steps: set up an isolated env, generate data for both reasoning families,
# evaluate the base model, SFT on solver traces from BOTH families, evaluate,
# GRPO from the SFT adapter, evaluate, and print the results table.
# Every model is scored by the same harness on the same held-out splits.
# Results land in runs/; paste the report into docs/REPORT.md.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
NAME="${NAME:-qwen1.5b}"
EVAL_N="${EVAL_N:-200}"
SFT_STEPS="${SFT_STEPS:-300}"
GRPO_STEPS="${GRPO_STEPS:-400}"
GENS="${GENS:-8}"
COMPLETION="${COMPLETION:-1024}"
SPLITS="test_id,test_ood,liar_test"

if [[ "${SMOKE:-0}" == "1" ]]; then
  MODEL="Qwen/Qwen2.5-0.5B-Instruct"; NAME="smoke"; EVAL_N=2
  SFT_STEPS=2; GRPO_STEPS=1; GENS=2; COMPLETION=64
fi

PY=.venv-train/bin/python
if [[ ! -x "$PY" ]]; then
  python3 -m venv .venv-train
  .venv-train/bin/pip install -q torch transformers trl peft datasets accelerate
fi
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "== data"
python3 -m detective_engine.generator --out data/generated
python3 training/make_sft_data.py \
  --train data/generated/train.jsonl,data/generated/liar_train.jsonl \
  --out data/generated/sft_both.jsonl

echo "== base model: $MODEL"
$PY -m detective_engine.evaluate --agent "hf:$MODEL" --splits "$SPLITS" --limit "$EVAL_N"

echo "== SFT on both families"
$PY training/sft_train.py --model "$MODEL" --data data/generated/sft_both.jsonl \
  --max-steps "$SFT_STEPS" --output "runs/sft/$NAME"
$PY -m detective_engine.evaluate --agent "hf:runs/sft/$NAME" --splits "$SPLITS" --limit "$EVAL_N"

echo "== GRPO from the SFT adapter"
TRAIN_MIX=data/generated/grpo_mix.jsonl
cat data/generated/train.jsonl data/generated/liar_train.jsonl > "$TRAIN_MIX"
$PY training/grpo_train.py --model "$MODEL" --init-adapter "runs/sft/$NAME" --train "$TRAIN_MIX" \
  --levels 1,2,3 --max-steps "$GRPO_STEPS" --num-generations "$GENS" --batch-size "$GENS" \
  --max-completion-length "$COMPLETION" --output "runs/grpo/$NAME"
# The GRPO adapter sits on top of the merged SFT weights; merge both for evaluation.
$PY training/merge_adapters.py --model "$MODEL" --adapters "runs/sft/$NAME,runs/grpo/$NAME" \
  --output "runs/merged/$NAME-grpo"
$PY -m detective_engine.evaluate --agent "hf:runs/merged/$NAME-grpo" --splits "$SPLITS" --limit "$EVAL_N"

echo "== results"
python3 -m detective_engine.evaluate --report
