#!/usr/bin/env python3
"""longctx_blind_sheet.py — render a blind sheet for LONG-context prompts (blinds/bp_blind, 2026-10-20).

    python3 Tools/longctx_blind_sheet.py --pairs blinds/bp_blind/pairs.json --out blinds/bp_blind

Same pairs schema, per-prompt seeded A/B shuffle, key.json and checks.md as Tools/m8_blind_sheet.py
(it calls that module's build() first, so the key is identical), then rewrites sheet.md for long
prompts: the task (last paragraph of the prompt) is shown up front and the context goes into a
collapsed <details> block; prompt and output text are fenced with ~~~~ and indented by two spaces,
so markdown headings inside documents or outputs ("## ...") cannot break the scorer's per-prompt
split. Scoring is unchanged:

    python3 Tools/m8_blind_sheet.py --pairs blinds/bp_blind/pairs.json --out blinds/bp_blind --score blinds/bp_blind/sheet.md
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m8_blind_sheet  # noqa: E402


def fence(text: str) -> str:
    body = "\n".join("  " + line for line in text.splitlines())
    return f"~~~~\n{body}\n~~~~\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Batched prefill (chunk 128 vs off) — long-context blind quality sheet")
    a = ap.parse_args()
    m8_blind_sheet.build(argparse.Namespace(pairs=a.pairs, out=a.out))  # writes key.json + checks.md (+ a sheet we replace)
    key = json.load(open(os.path.join(a.out, "key.json")))
    pairs = json.load(open(a.pairs))["pairs"]
    by = {}
    for p in pairs:
        by.setdefault(p["prompt"], {})[p["mask"]] = p
    s = [f"# {a.title}\n",
         "Score each output 1–5 (5 best) for: coherence (C), instruction adherence (I), factual "
         "consistency with the provided context (F). Then the overall verdict A/B/tie. Do NOT open "
         "key.json first. Each prompt's context is collapsed under 'Context'; the task is shown above it.\n"]
    for pid in sorted(key):
        order = [key[pid]["A"], key[pid]["B"]]
        user = by[pid][order[0]]["user"]
        task = user.rstrip().split("\n\n")[-1]
        s.append(f"\n## {pid}\n\n**Task**: {task}\n")
        s.append(f"\n<details><summary>Context (full prompt, {len(user.split())} words)</summary>\n\n{fence(user)}\n</details>\n")
        for label, arm in zip("AB", order):
            s.append(f"\n### Output {label}\n\n{fence(by[pid][arm]['text'])}\nC: __  I: __  F: __\n")
        s.append("\n**Overall (A/B/tie)**: __\n")
    text = "\n".join(s)
    # Self-check: the scorer's split must find exactly one block per prompt, each with its Overall line.
    blocks = dict(re.findall(r"## (\S+)\n(.*?)(?=\n## |\Z)", text, re.S))
    missing = [p for p in key if p not in blocks or "**Overall (A/B/tie)**" not in blocks[p]]
    if missing:
        print(f"ABORT: scorer split broken for {missing}", file=sys.stderr); return 1
    with open(os.path.join(a.out, "sheet.md"), "w") as f:
        f.write(text)
    print(f"rewrote sheet.md for {len(key)} long-context prompts (scorer split verified)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
