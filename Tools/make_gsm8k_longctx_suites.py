#!/usr/bin/env python3
"""make_gsm8k_longctx_suites.py — k-shot GSM8K prompt suites at controlled context lengths (E-008).

    python3 Tools/make_gsm8k_longctx_suites.py --test-start 300 --count 100 \
        --budgets 0,1000,2000,3500 --tokenizer models/llada2-1-mini/tokenizer.json

For each budget B, writes PromptSuites/gsm8k_ctx<B>.json: the SAME test questions (GSM8K test rows
test-start … test-start+count−1, ids `gsm8k-<row>`) each preceded by worked examples from the
GSM8K *train* split, in a fixed order (train rows 0, 1, 2, …), added while the user-message token
count stays ≤ B for EVERY question (so all prompts of a suite share the same shot count; B = 0 is
zero-shot). Calculator annotations `<<…>>` are stripped from the train solutions. Token counts are
of the user-message text with the model's tokenizer (the chat template adds a fixed ~20 tokens).

Prints sha256 of both upstream files, shot counts, and min/median/max user-token counts per suite.
Needs the `tokenizers` package (in ~/.venvs/neodiffusion-mlx). The engine's context ceiling is
4096 tokens (sliding_window, inert below it): keep B + chat template + gen length < 4096.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import urllib.request

from tokenizers import Tokenizer

BASE = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
SUITES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusion-bench", "PromptSuites")
INSTR = "Solve the math word problem. Show your reasoning, then give the final answer after '####'.\n\n"


def fetch(name: str) -> list[dict]:
    raw = urllib.request.urlopen(BASE + name).read()
    print(f"{name} sha256 {hashlib.sha256(raw).hexdigest()} ({len(raw)} bytes)")
    return [json.loads(l) for l in raw.decode("utf-8").splitlines()]


def shot(ex: dict) -> str:
    sol = re.sub(r"<<[^>]*>>", "", ex["answer"]).strip()
    return f"Question: {ex['question'].strip()}\nAnswer: {sol}\n\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--budgets", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--name-prefix", default="gsm8k_ctx", help="suite name prefix (default gsm8k_ctx)")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    n = lambda s: len(tok.encode(s, add_special_tokens=False).ids)
    test, train = fetch("test.jsonl"), fetch("train.jsonl")
    rows = list(range(a.test_start, a.test_start + a.count))
    tails = {i: f"Question: {test[i]['question'].strip()}\nAnswer:" for i in rows}
    answers = {i: (re.search(r"####\s*(-?\d+)", test[i]["answer"]) or [None, ""])[1] for i in rows}
    for b in map(int, a.budgets.split(",")):
        k, prefix = 0, ""
        if b > 0:
            while True:
                cand = prefix + shot(train[k])
                if max(n(INSTR + cand + tails[i]) for i in rows) > b:
                    break
                prefix, k = cand, k + 1
        prompts = [{"id": f"gsm8k-{i:03d}", "user": INSTR + prefix + tails[i], "answer": answers[i]} for i in rows]
        lens = [n(p["user"]) for p in prompts]
        out = os.path.join(SUITES, f"{a.name_prefix}{b}.json")
        if os.path.exists(out):
            print(f"ABORT: {out} exists"); return 1
        with open(out, "w") as f:
            json.dump({"name": f"{a.name_prefix}{b}", "prompts": prompts}, f, indent=2)
        print(f"{a.name_prefix}{b}: {k} shots, user tokens min {min(lens)} median {statistics.median(lens)} max {max(lens)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
