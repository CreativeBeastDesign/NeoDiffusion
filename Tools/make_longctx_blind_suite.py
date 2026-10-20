#!/usr/bin/env python3
"""make_longctx_blind_suite.py — long-context chat / reasoning / code prompts for the batched-prefill
blind sheet (blinds/bp_blind, 2026-10-20).

    ~/.venvs/neodiffusion-mlx/bin/python Tools/make_longctx_blind_suite.py

The standard chat/reasoning/code suites are 15–44 tokens — at most two 32-token prompt blocks — so
batched prefill barely touches them. These 12 prompts put 1.5–3k tokens of real context (this
repository's own documents, fact files and source code at the current commit, plus 17-shot GSM8K
with the last unused test rows 1300/1301) before a task, so the prefill path actually differs.
Writes Tools/diffusion-bench/PromptSuites/longctx_blind.json and prints token counts (model
tokenizer, user message only). Every prompt must stay ≤ 3300 tokens so prompt + template + gen 512
stays below 4096.
"""
import json
import os
import re
import urllib.request

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITES = os.path.join(ROOT, "Tools/diffusion-bench/PromptSuites")
GSM = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"


def read(rel: str) -> str:
    return open(os.path.join(ROOT, rel)).read()


def section(text: str, start: str, end: str | None) -> str:
    i = text.index(start)
    j = text.index(end, i + len(start)) if end else len(text)
    return text[i:j].strip()


def doc(title: str, body: str) -> str:
    return f"Document: {title}\n<<<\n{body}\n>>>\n\n"


def gsm_prompt(row: int) -> tuple[str, str]:
    test = [json.loads(l) for l in urllib.request.urlopen(GSM + "test.jsonl").read().decode().splitlines()]
    train = [json.loads(l) for l in urllib.request.urlopen(GSM + "train.jsonl").read().decode().splitlines()]
    shots = "".join(f"Question: {e['question'].strip()}\nAnswer: {re.sub(r'<<[^>]*>>', '', e['answer']).strip()}\n\n"
                    for e in train[:17])
    instr = "Solve the math word problem. Show your reasoning, then give the final answer after '####'.\n\n"
    gold = re.search(r"####\s*(-?\d+)", test[row]["answer"]).group(1)
    return instr + shots + f"Question: {test[row]['question'].strip()}\nAnswer:", gold


def main() -> int:
    claude = read("CLAUDE.md")
    readme = read("paper/README.md")
    plan = read("Plans/next-phase-plan-2026-10-18.md")
    phase1 = read("Plans/phase-1-conceptual-design.md")
    facts = lambda ids: "".join(doc(f"paper/facts/{i}.md", read(f"paper/facts/{i}.md")) for i in ids)
    prompts = [
        ("lc-chat-summary", doc("CLAUDE.md — Fixed decisions and Critical gotchas",
            section(claude, "## Fixed decisions", "## Working rules"))
         + "Using only the document above, list the five rules a new contributor is most likely to break, "
           "as a bulleted list with one sentence each on why it matters."),
        ("lc-chat-question", doc("paper/README.md", section(readme, "## 1. Why this structure", "## 5. Glossary"))
         + "According to the document above: what may an interpretation file (I-*) link to, what must it never "
           "contain, and how is that enforced? Answer in at most six sentences."),
        ("lc-chat-email", doc("Engine backlog (Plans/next-phase-plan-2026-10-18.md §3 and §7)",
            section(plan, "## 3. Goal B", "## 4. Paranoia standards") + "\n\n" + section(plan, "## 7. Decisions taken", None)[:2500])
         + "Write a short, friendly email (under 200 words) to a colleague who has been away, summarising which "
           "engine items are done and which are still open, based only on the document above."),
        ("lc-chat-explain", doc("Plans/phase-1-conceptual-design.md (excerpt)",
            phase1[max(0, phase1.index("**ExactPrefixCache**") - 3500):][:7000])
         + "Explain to someone who knows programming but not machine learning why this design keeps two separate "
           "caches, and what would go wrong if they were merged. Use plain language."),
        ("lc-reason-table", facts(["F-025", "F-026", "F-027"])
         + "Using only the three fact files above: (1) which KV mode's step delta grows the most between the "
           "shortest and the longest context, and by how much; (2) which mode stays closest to zero across all "
           "lengths; (3) by what factor does the uncompressed cache's time per generation grow from the shortest "
           "to the longest context? Show the arithmetic."),
        ("lc-reason-weights", facts(["F-019", "F-020", "F-022"])
         + "Using only the three fact files above, decide whether the step cost of 4-bit Keys (k4v8) depends on "
           "the weight precision. State what each file contributes, then give a one-sentence conclusion. Be "
           "careful to distinguish 'not detected' from 'absent'."),
    ]
    g1, a1 = gsm_prompt(1300)
    g2, a2 = gsm_prompt(1301)
    prompts += [("lc-reason-gsm1300", g1), ("lc-reason-gsm1301", g2)]
    prompts += [
        ("lc-code-explain", doc("Packages/DiffusionCore/Sources/LayerKVCache.swift", read("Packages/DiffusionCore/Sources/LayerKVCache.swift"))
         + "Explain what commitPending does for each storage form (quantized, pooled, raw), and why the pending "
           "K/V are cleared at the end. Refer to the code."),
        ("lc-code-test", doc("paper/scripts/e011_quality.py", read("paper/scripts/e011_quality.py"))
         + "Write pytest unit tests for mcnemar_exact covering: no discordant pairs, a symmetric split, a strongly "
           "lopsided split, and that the result never exceeds 1."),
        ("lc-code-review", doc("paper/scripts/rerun_identity.py", read("paper/scripts/rerun_identity.py"))
         + doc("paper/scripts/genlen_identity.py", read("paper/scripts/genlen_identity.py"))
         + "Review both scripts. For each, name one edge case or input that would make it report something "
           "misleading, and propose a one-line fix."),
        ("lc-code-refactor", doc("paper/scripts/weight_interaction.py", read("paper/scripts/weight_interaction.py"))
         + doc("paper/scripts/rerun_identity.py", read("paper/scripts/rerun_identity.py"))
         + "Both scripts load bench JSONL rows. Write one shared helper function load_rows(path, valid_only) that "
           "both could use, and show how each script's loading code would change."),
    ]
    tok = Tokenizer.from_file(os.path.join(ROOT, "models/llada2-1-mini/tokenizer.json"))
    out = []
    for pid, user in prompts:
        n = len(tok.encode(user, add_special_tokens=False).ids)
        print(f"{pid:20s} {n:5d} tokens")
        assert 64 <= n <= 3300, f"{pid}: {n} tokens outside [64, 3300]"
        out.append({"id": pid, "user": user})
    print(f"gold answers: lc-reason-gsm1300 = {a1}, lc-reason-gsm1301 = {a2}")
    path = os.path.join(SUITES, "longctx_blind.json")
    with open(path, "w") as f:
        json.dump({"name": "longctx_blind", "prompts": out}, f, indent=2)
    print(f"wrote {len(out)} prompts to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
