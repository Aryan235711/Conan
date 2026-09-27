"""Micro-skill probe: can a model tell whether an alibi covers a time window?

The step breakdown showed models treat any verified alibi as clearing a
suspect. This probe isolates the one comparison that matters: given an alibi
interval and a death window, does the alibi cover the whole window?

Items are balanced: half "yes" (covers), half "no" split across three ways
of failing to cover (ends before the window opens, ends inside it, starts
inside it). About half the windows cross midnight. Two formats are asked:

    clock     "from 23:10 to 00:45"                 (as in the real cases)
    minutes   "from minute 670 to minute 765"       (no clock arithmetic)

If a model fails both, it lacks the comparison skill.  If it passes minutes
but fails clock, the problem is reading clock times, especially across
midnight.  If it passes both, the skill exists and the cases fail because it
is not applied, which points to training fixes rather than a bigger model.

    .venv-train/bin/python benchmarks/interval_probe.py --agent hf:runs/sft/qwen0.5b
    python3 benchmarks/interval_probe.py --agent ollama:qwen2.5-coder:7b
    python3 benchmarks/interval_probe.py --report
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detective_engine.evaluate import _slug, wilson  # noqa: E402
from detective_engine.generator import fmt  # noqa: E402

OUT = ROOT / "runs" / "probes"


def make_items(n: int = 200, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    kinds = ["covers"] * (n // 2) + ["ends_before"] * (n // 6) + ["ends_inside"] * (n // 6) + ["starts_inside"] * (n - n // 2 - 2 * (n // 6))
    rng.shuffle(kinds)
    items = []
    for i, kind in enumerate(kinds):
        midnight = i % 2 == 0
        lo = rng.randint(11 * 60 + 5, 12 * 60 - 20) if midnight else rng.randint(8 * 60, 10 * 60)   # minutes since noon
        hi = lo + rng.randint(40, 120)
        if kind == "covers":
            a, b = lo - rng.randint(5, 90), hi + rng.randint(5, 90)
        elif kind == "ends_before":
            b = lo - rng.randint(10, 90); a = b - rng.randint(60, 180)
        elif kind == "ends_inside":
            b = rng.randint(lo + 10, hi - 10); a = b - rng.randint(60, 180)
        else:
            a = rng.randint(lo + 10, hi - 10); b = a + rng.randint(60, 180)
        items.append({"id": i, "kind": kind, "answer": "yes" if kind == "covers" else "no",
                      "midnight": (a + 12 * 60) // (24 * 60) != (b + 12 * 60) // (24 * 60) or
                                  (lo + 12 * 60) // (24 * 60) != (hi + 12 * 60) // (24 * 60),
                      "a": a, "b": b, "lo": lo, "hi": hi})
    return items


def question(item: dict, style: str) -> str:
    if style in ("clock", "clock_cot"):
        alibi = f"from {fmt(item['a'])} to {fmt(item['b'])}"
        window = f"from {fmt(item['lo'])} to {fmt(item['hi'])}"
        note = " Times are on one night and may pass midnight."
    else:
        alibi = f"from minute {item['a']} to minute {item['b']}"
        window = f"from minute {item['lo']} to minute {item['hi']}"
        note = ""
    ask = ("Does the alibi cover the entire window, so that the suspect could not have committed the crime "
           "at any time in it? ")
    if style == "clock_cot":
        tail = ("Compare the start and end times step by step, then give your final answer on the last line "
                "as 'Answer: yes' or 'Answer: no'.")
    else:
        tail = "Answer with one word: yes or no."
    return f"A suspect has a verified alibi {alibi}. The death happened at some time {window}.{note} " + ask + tail


def parse(text: str) -> str | None:
    low = text.lower()
    tagged = re.findall(r"answer:\s*\**\s*(yes|no)\b", low)
    if tagged:
        return tagged[-1]
    found = re.findall(r"\b(yes|no)\b", low)
    return found[-1] if found else None


def run(agent_spec: str, n: int, cot_n: int = 120) -> None:
    from detective_engine.evaluate import get_agent

    items = make_items(n)
    if agent_spec.startswith("hf:"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = agent_spec[3:]
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

        def ask(q: str, max_tokens: int) -> str:
            ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    else:
        import urllib.request
        model_name = agent_spec.split(":", 1)[1]

        def ask(q: str, max_tokens: int) -> str:
            body = json.dumps({"model": model_name, "messages": [{"role": "user", "content": q}], "stream": False,
                               "options": {"temperature": 0, "num_predict": max_tokens, "seed": 0}}).encode()
            req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())["message"]["content"]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"interval_{_slug(agent_spec)}.jsonl"
    done = set()
    if path.exists():
        done = {(json.loads(l)["id"], json.loads(l)["style"]) for l in open(path)}
    for style in ("clock", "minutes", "clock_cot"):
        for it in (items[:cot_n] if style == "clock_cot" else items):
            if (it["id"], style) in done:
                continue
            text = ask(question(it, style), 700 if style == "clock_cot" else 8)
            rec = {"agent": agent_spec, "style": style, "id": it["id"], "kind": it["kind"],
                   "midnight": it["midnight"], "answer": it["answer"], "raw": text, "pred": parse(text)}
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print(f"{agent_spec} {style}: done", flush=True)


def report() -> None:
    rows = []
    for p in sorted(OUT.glob("interval_*.jsonl")):
        recs = [json.loads(l) for l in open(p)]
        for style in ("clock", "minutes", "clock_cot"):
            r = [x for x in recs if x["style"] == style]
            if not r:
                continue
            k = sum(x["pred"] == x["answer"] for x in r)
            lo, hi = wilson(k, len(r))
            by = collections.defaultdict(lambda: [0, 0])
            for x in r:
                by[x["kind"]][0] += x["pred"] == x["answer"]; by[x["kind"]][1] += 1
                key = "midnight" if x["midnight"] else "no_midnight"
                by[key][0] += x["pred"] == x["answer"]; by[key][1] += 1
            yes_rate = sum(x["pred"] == "yes" for x in r) / len(r)
            no_answer = sum(x["pred"] is None for x in r) / len(r)
            rows.append((recs[0]["agent"], style, len(r), k / len(r), lo, hi, yes_rate, no_answer, by))
    print(f"{'agent':42} {'format':9} {'n':>4} {'accuracy':>16} {'says yes':>9} {'no answer':>9}  covers ends_before ends_inside starts_inside | midnight other")
    for agent, style, n, acc, lo, hi, yes, noans, by in rows:
        f = lambda k: f"{by[k][0] / max(1, by[k][1]):.2f}"  # noqa: E731
        print(f"{agent[-42:]:42} {style:9} {n:>4} {acc:6.2f} ({lo:.2f}-{hi:.2f}) {yes:9.2f} {noans:9.2f}  "
              f"{f('covers'):>6} {f('ends_before'):>11} {f('ends_inside'):>11} {f('starts_inside'):>13} | {f('midnight'):>8} {f('no_midnight'):>5}")
    print("\nChance is 0.50. A model that always says yes scores 1.00 on 'covers' and 0.00 on the other three.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--cot-n", type=int, default=120, help="items for the step-by-step format")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.agent:
        run(args.agent, args.n, args.cot_n)
    if args.report or not args.agent:
        report()


if __name__ == "__main__":
    main()
