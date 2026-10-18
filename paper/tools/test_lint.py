#!/usr/bin/env python3
"""Self-test for lint_links.py: builds throw-away paper trees and checks that each rule fires
(and that a clean tree passes). Run: python3 paper/tools/test_lint.py"""
import hashlib, os, subprocess, sys, tempfile
from pathlib import Path

LINT = Path(__file__).with_name("lint_links.py")

def run(tree: dict[str, str]) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "paper"
        for rel, body in tree.items():
            p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(body)
        data = root / "facts" / "data"
        if data.exists():
            lines = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(data)}"
                     for f in data.rglob("*") if f.is_file() and f.name != "MANIFEST.sha256"
                     and not f.name.startswith("UNLISTED")]
            if "facts/data/MANIFEST.sha256" not in tree:
                (data / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
        r = subprocess.run([sys.executable, str(LINT)], env={**os.environ, "PAPER_ROOT": str(root)},
                           capture_output=True, text=True)
        return r.returncode, r.stdout

FACT = "id: F-001\nstatement: arm A ran 291 steps\nderived by: scripts/a.py\ninputs: [R-001](runs/R-001.md)\n"
RUN = "id: R-001\ncommand: x\ngit: abc\nhost: MacBookPro17,1\ndata: [d](../data/d.jsonl)\n"
SRC = "id: S-001\ntitle: KIVI\nurl: https://arxiv.org/abs/2402.02750\nverified: 2026-10-18\n"
INTERP = "id: I-001\nclaim: x\nstatus: conjecture\nrests on: [F-001](../facts/F-001.md), S-001\nfalsifier: y\nvalue 291 steps\n"
CLEAN = {"facts/F-001.md": FACT, "facts/runs/R-001.md": RUN, "facts/data/d.jsonl": "{}\n",
         "sources/S-001-kivi.md": SRC, "interpretations/I-001.md": INTERP}

cases = [
    ("clean tree passes", CLEAN, 0, None),
    ("fact cites interpretation", {**CLEAN, "facts/F-001.md": FACT + "see I-001\n"}, 1, "may not cite I-001"),
    ("fact links to source", {**CLEAN, "facts/F-001.md": FACT + "[k](../sources/S-001-kivi.md)\n"}, 1, "may not reference 'sources'"),
    ("source cites fact", {**CLEAN, "sources/S-001-kivi.md": SRC + "matches F-001\n"}, 1, "may not cite F-001"),
    ("observation cites source", {**CLEAN, "observations/O-001.md": "id: O-001\nwhat was looked at: x\nprocedure: blind\nobservation: per S-001\n"}, 1, "may not cite S-001"),
    ("causal word in fact", {**CLEAN, "facts/F-001.md": FACT + "slower because of keys\n"}, 1, "interpretive word"),
    ("missing field", {**CLEAN, "facts/F-002.md": "id: F-002\nstatement: x\n"}, 1, "missing required field"),
    ("unlisted data file", {**CLEAN, "facts/data/UNLISTED.jsonl": "{}\n"}, 1, "not in MANIFEST"),
    ("tampered data", {**CLEAN, "facts/data/MANIFEST.sha256": "0" * 64 + "  d.jsonl\n"}, 1, "sha256 mismatch"),
    ("id/filename mismatch", {**CLEAN, "facts/F-009.md": FACT}, 1, "does not match filename"),
    ("untraceable number (warn only)", {**CLEAN, "interpretations/I-001.md": INTERP + "about 12.34 tok/s\n"}, 0, "not found in any F-*"),
]
fails = 0
for name, tree, want_rc, want_text in cases:
    rc, out = run(tree)
    ok = rc == want_rc and (want_text is None or want_text in out)
    fails += not ok
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"\n  rc={rc}\n  " + out.replace("\n", "\n  ")))
# L5 needs a real git history: registration frozen at a commit; results may be appended, the
# pre-registered part may not change. (Regression for the 2026-10-18 strip bug, which left a blank line.)
def l5_case() -> tuple[int, int]:
    with tempfile.TemporaryDirectory() as d:
        r = Path(d); e = r / "paper" / "experiments" / "E-001.md"; e.parent.mkdir(parents=True)
        base = "id: E-001\ntests: x\nprediction: steps go up\n"
        e.write_text(base)
        g = lambda *a: subprocess.run(["git", "-C", d, "-c", "user.email=t@t", "-c", "user.name=t", *a], capture_output=True, text=True)
        g("init", "-q"); g("add", "."); g("commit", "-qm", "reg")
        sha = g("rev-parse", "--short", "HEAD").stdout.strip()
        lint = lambda: subprocess.run([sys.executable, str(LINT)], env={**os.environ, "PAPER_ROOT": str(r / "paper")}, capture_output=True, text=True).returncode
        e.write_text(base.replace("prediction:", f"registered_commit: {sha}\nprediction:") + "--- RESULTS ---\nok\n")
        a = lint()
        e.write_text(e.read_text().replace("steps go up", "steps go down"))
        return a, lint()
a, b = l5_case()
for name, got, want in [("L5 frozen prereg + appended results passes", a, 0), ("L5 edited prereg fails", b, 1)]:
    ok = got == want; fails += not ok; cases.append(name)
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f" (rc={got})"))
print(f"{len(cases) - fails}/{len(cases)} lint self-tests passed")
sys.exit(1 if fails else 0)
