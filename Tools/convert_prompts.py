#!/usr/bin/env python3
"""WP-2a: convert Plans/prompts.md (50 calibration prompts, 4 categories) into
Tools/diffusion-bench/PromptSuites/calibration.json ({name, prompts:[{id, user}]}).

Ids are category-prefixed (cal-chat-01 … cal-de-50) so the Spiffy graph builder can
ablate per category (German competence of the model is unverified — logbook risk 5).
"""
import json
import re
import argparse

CATEGORY_SLUGS = {
    1: "chat",
    2: "reason",
    3: "code",
    4: "de",
}


def parse(path):
    text = open(path, encoding="utf-8").read()
    # Category headers: "CATEGORY <n>: ..." between ==== rules.
    cat_positions = [(int(m.group(1)), m.start())
                     for m in re.finditer(r"CATEGORY (\d+):", text)]
    cat_positions.append((None, len(text)))
    prompts = []
    for (cat, start), (_, end) in zip(cat_positions, cat_positions[1:]):
        body = text[start:end]
        # Numbered prompts: "NN. text" possibly spanning lines until the next number
        # or a --- subgroup marker or section rule.
        for m in re.finditer(
                r"^(\d+)\.\s+(.*?)(?=^\d+\.\s|\Z)", body, re.M | re.S):
            num = int(m.group(1))
            item = m.group(2)
            # Strip subgroup markers and section rules that trail the last prompt of a group.
            item = re.sub(r"^---.*$", "", item, flags=re.M)
            item = re.sub(r"^=+$", "", item, flags=re.M)
            item = re.sub(r"^CATEGORY.*$", "", item, flags=re.M)
            item = "\n".join(line.rstrip() for line in item.strip().splitlines())
            item = item.strip()
            if item:
                prompts.append({
                    "id": f"cal-{CATEGORY_SLUGS[cat]}-{num:02d}",
                    "user": item,
                })
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="Plans/prompts.md")
    ap.add_argument("--out", default="Tools/diffusion-bench/PromptSuites/calibration.json")
    args = ap.parse_args()
    prompts = parse(args.src)
    assert len(prompts) == 50, f"expected 50 prompts, parsed {len(prompts)}"
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"name": "calibration", "prompts": prompts}, f,
                  ensure_ascii=False, indent=1)
    by_cat = {}
    for p in prompts:
        by_cat[p["id"].split("-")[1]] = by_cat.get(p["id"].split("-")[1], 0) + 1
    print(f"wrote {args.out}: {len(prompts)} prompts, categories {by_cat}")


if __name__ == "__main__":
    main()
