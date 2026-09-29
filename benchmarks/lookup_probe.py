"""Micro-skill probe: can a model find the record about a person at a time?

SFT v7's lying-witness failures are mostly lookup errors. This probe asks 100
held-out lookup questions (a different seed from the training curriculum):
given a list of records, is a claim about person X at time T confirmed,
contradicted, or untested? Chance is about 1/3.

    .venv-train/bin/python benchmarks/lookup_probe.py --agent hf:runs/sft/qwen0.5b-v7
    python3 benchmarks/lookup_probe.py --report
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from detective_engine.evaluate import _slug, wilson  # noqa: E402
from make_lookup_data import build  # noqa: E402

OUT = ROOT / "runs" / "probes"


def parse(text: str) -> str | None:
    m = re.findall(r"answer:\s*\**\s*(confirmed|contradicted|no record)", text.lower())
    return m[-1] if m else None


def run(agent_spec: str, n: int) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = agent_spec.split(":", 1)[1]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = Path(path) / "adapter_config.json"
    if cfg.exists():
        from peft import PeftModel
        base = json.loads(cfg.read_text())["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(base)
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base), path).merge_and_unload()
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path)
    model.to(device).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"lookup_{_slug(agent_spec)}.jsonl"
    done = {json.loads(l)["i"] for l in open(out)} if out.exists() else set()
    for i, item in enumerate(build(n, seed=99)):
        if i in done:
            continue
        ids = tok.apply_chat_template(item["prompt"], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(device)
        with torch.no_grad():
            o = model.generate(**ids, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        with open(out, "a") as f:
            f.write(json.dumps({"i": i, "agent": agent_spec, "answer": item["answer"], "pred": parse(text), "raw": text}) + "\n")
    print(f"{agent_spec}: done", flush=True)


def report() -> None:
    for p in sorted(OUT.glob("lookup_*.jsonl")):
        r = [json.loads(l) for l in open(p)]
        k = sum(x["pred"] == x["answer"] for x in r)
        lo, hi = wilson(k, len(r))
        by = {}
        for a in ("confirmed", "contradicted", "no record"):
            s = [x for x in r if x["answer"] == a]
            by[a] = f"{sum(x['pred'] == a for x in s)}/{len(s)}"
        print(f"{r[0]['agent'][-40:]:40} n={len(r)} accuracy {k / len(r):.2f} ({lo:.2f}-{hi:.2f})  {by}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.agent:
        run(args.agent, args.n)
    if args.report or not args.agent:
        report()


if __name__ == "__main__":
    main()
