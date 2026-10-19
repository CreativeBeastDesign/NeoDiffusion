#!/usr/bin/env python3
"""make_gsm8k_suite.py — write a GSM8K prompt suite from a slice of the official test set.

    python3 Tools/make_gsm8k_suite.py --start 100 --count 200 --name gsm8k_fresh200

Same source, id scheme (`gsm8k-<row index>`) and answer extraction as Tools/download_gsm8k.py
(which produced gsm8k_100 = rows 0–99), so ids never collide across slices. Before writing, checks
that rows 0–99 reproduce the checked-in gsm8k_100.json exactly (same upstream file). Prints the
sha256 of the downloaded test.jsonl.
"""
import argparse
import hashlib
import json
import os
import re
import urllib.request

URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
SUITES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusion-bench", "PromptSuites")


def to_prompt(i: int, line: str) -> dict:
    data = json.loads(line)
    m = re.search(r"####\s*(-?\d+)", data["answer"])
    return {"id": f"gsm8k-{i:03d}", "user": data["question"], "answer": m.group(1) if m else ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    raw = urllib.request.urlopen(URL).read()
    print(f"test.jsonl sha256 {hashlib.sha256(raw).hexdigest()} ({len(raw)} bytes)")
    lines = raw.decode("utf-8").splitlines()
    ref = json.load(open(os.path.join(SUITES, "gsm8k_100.json")))["prompts"]
    if [to_prompt(i, lines[i]) for i in range(100)] != ref:
        print("ABORT: rows 0-99 do not reproduce gsm8k_100.json — upstream file changed")
        return 1
    prompts = [to_prompt(i, lines[i]) for i in range(a.start, a.start + a.count)]
    out = os.path.join(SUITES, f"{a.name}.json")
    if os.path.exists(out):
        print(f"ABORT: {out} exists")
        return 1
    with open(out, "w") as f:
        json.dump({"name": a.name, "prompts": prompts}, f, indent=2)
    print(f"wrote {len(prompts)} prompts ({prompts[0]['id']}..{prompts[-1]['id']}) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
