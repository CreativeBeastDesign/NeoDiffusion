#!/usr/bin/env python3
"""M8 E8 — blind scoring sheet for `.strict` vs `.referenceBias` (m8-logbook).

Reads the full paired outputs written by
  diffusion-bench llada --mask-diagnostic --blind-out scratch/m8_blind/pairs.json
and emits:
  - sheet.md   : per prompt, outputs labelled A/B in a per-prompt seeded shuffle,
                 with scoring boxes (coherence / instruction adherence / factual
                 consistency, 1–5 each + "which is better overall: A/B/tie")
  - key.json   : the A/B → mask assignment (DO NOT open before scoring)
  - checks.md  : scripted objective checks per output (length, early-EOS block,
                 degeneration heuristics: max 4-gram repetition, mask-token leaks)

Deterministic shuffle: seeded per prompt id, so re-generating the sheet cannot
accidentally unblind a half-scored session.

After André fills sheet.md, `--score sheet.md` merges scores with key.json and prints
the per-mask summary.
"""
import argparse
import hashlib
import json
import os
import re
from collections import Counter


def degeneration_checks(text):
    words = text.split()
    ngrams = Counter(tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3)))
    max_rep = max(ngrams.values()) if ngrams else 0
    return {
        "chars": len(text),
        "words": len(words),
        "max4gramRepeat": max_rep,
        "maskLeak": "<|mask|>" in text,
        "truncatedMidWord": bool(re.search(r"\w-?$", text.strip()[-1:])) if text.strip() else False,
    }


def build(args):
    with open(args.pairs) as f:
        pairs = json.load(f)["pairs"]
    by_prompt = {}
    for p in pairs:
        by_prompt.setdefault(p["prompt"], {})[p["mask"]] = p

    sheet, key, checks = [], {}, []
    sheet.append("# M8 E8 — blind mask-quality scoring\n")
    sheet.append("Score each output 1–5 (5 best) for: coherence (C), instruction "
                 "adherence (I), factual consistency where applicable (F, else '-'). "
                 "Then overall verdict A/B/tie. Do NOT open key.json first.\n")
    for pid, masks in sorted(by_prompt.items()):
        if len(masks) != 2:
            continue
        # Deterministic per-prompt shuffle.
        flip = int(hashlib.sha256(pid.encode()).hexdigest(), 16) % 2 == 1
        order = ["referenceBias", "strict"] if flip else ["strict", "referenceBias"]
        key[pid] = {"A": order[0], "B": order[1]}
        sheet.append(f"\n## {pid}\n\n**Prompt**: {masks[order[0]]['user']}\n")
        for label, mask in zip("AB", order):
            sheet.append(f"\n### Output {label}\n\n```\n{masks[mask]['text']}\n```\n")
            sheet.append(f"C: __  I: __  F: __\n")
            checks.append({
                "prompt": pid, "label": label,
                **degeneration_checks(masks[mask]["text"]),
                "tokens": masks[mask]["tokens"],
                "eosBlock": masks[mask]["eosBlock"],
            })
        sheet.append("\n**Overall (A/B/tie)**: __\n")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sheet.md"), "w") as f:
        f.write("\n".join(sheet))
    with open(os.path.join(args.out, "key.json"), "w") as f:
        json.dump(key, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "checks.md"), "w") as f:
        f.write("| prompt | label | words | max4gramRepeat | maskLeak | eosBlock | tokens |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for c in checks:
            f.write(f"| {c['prompt']} | {c['label']} | {c['words']} | "
                    f"{c['max4gramRepeat']} | {c['maskLeak']} | {c['eosBlock']} | "
                    f"{c['tokens']} |\n")
    print(f"wrote sheet.md ({len(key)} prompts), key.json, checks.md to {args.out}")


def score(args):
    with open(os.path.join(args.out, "key.json")) as f:
        key = json.load(f)
    text = open(args.score).read()
    verdicts = {}
    for pid, block in re.findall(r"## (\S+)\n(.*?)(?=\n## |\Z)", text, re.S):
        m = re.search(r"\*\*Overall \(A/B/tie\)\*\*: *(\w+)", block)
        if m and pid in key:
            v = m.group(1).strip().upper()
            verdicts[pid] = key[pid].get(v, "tie" if v == "TIE" else None)
    wins = Counter(verdicts.values())
    print(f"scored prompts: {len(verdicts)}")
    print(f"strict wins: {wins.get('strict', 0)} | referenceBias wins: "
          f"{wins.get('referenceBias', 0)} | ties: {wins.get('tie', 0)}")
    for pid, winner in sorted(verdicts.items()):
        print(f"  {pid}: {winner}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="scratch/m8_blind/pairs.json")
    ap.add_argument("--out", default="scratch/m8_blind")
    ap.add_argument("--score", help="filled sheet.md — merge with key and summarize")
    args = ap.parse_args()
    if args.score:
        score(args)
    else:
        build(args)
