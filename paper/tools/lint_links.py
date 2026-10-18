#!/usr/bin/env python3
"""lint_links.py — enforce the paper/ knowledge-base rules (paper/README.md §1 and §4).

Stdlib only. Exit 0 = clean, 1 = violations, 2 = usage error.

    python3 paper/tools/lint_links.py            # all checks
    python3 paper/tools/lint_links.py --strict   # also fail on numeric-traceability warnings

Checks
  L1  dependency direction: each layer may reference only the layers its row in README §1 allows,
      both by markdown link / path and by bare record ID (F-001, O-002, S-003, ...).
  L2  data integrity: every file under facts/data/ is listed in facts/data/MANIFEST.sha256 with a
      matching sha256, and every manifest entry exists.
  L3  record hygiene: F-* and O-* contain no causal / interpretive vocabulary; required header
      fields are present per template.
  L4  append-only: F/O/R/S files that exist in git HEAD may only grow (HEAD content must be a
      prefix of the working copy), and facts/data files may never change once committed.
  L5  pre-registration freeze: an E-* file with `registered_commit: <sha>` must keep everything
      above its `--- RESULTS ---` marker byte-identical to that commit.
  L6  numeric traceability (warning; error with --strict): numbers with ≥ 2 significant digits in
      I-* and drafts/ must appear verbatim in some F-* file.
  L7  ID uniqueness and filename/ID agreement (`id: F-007` must live in F-007*.md).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PAPER_ROOT") or Path(__file__).resolve().parents[1]).resolve()  # paper/
REPO = ROOT.parent

# layer -> set of layers it may reference. "data", "runs", "scripts" are sub-areas of facts/.
ALLOWED: dict[str, set[str]] = {
    "sources": set(),                                 # external URLs only
    "runs": {"data", "scripts"},
    "facts": {"data", "runs", "scripts"},
    "observations": {"data"},
    "experiments": {"interpretations", "runs", "facts", "scripts"},
    "interpretations": {"facts", "observations", "sources", "interpretations"},
    "drafts": {"facts", "observations", "sources", "interpretations", "experiments", "runs", "data", "scripts"},
    "root": {"facts", "observations", "sources", "interpretations", "experiments", "runs", "data", "scripts", "drafts", "tools"},
    "tools": set(),
    "scripts": {"data"},
}
ID_LAYER = {"S": "sources", "F": "facts", "O": "observations", "I": "interpretations",
            "E": "experiments", "R": "runs"}
ID_RE = re.compile(r"\b([SFOIER])-(\d{3,})\b")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
PATHISH_RE = re.compile(r"(?<![\w/])((?:\.\./)*(?:paper/)?(?:facts/data|facts/runs|facts|observations|sources|interpretations|experiments|drafts|scripts)/[\w./-]+)")
CAUSAL = re.compile(r"\b(because|therefore|suggests?|shows? that|proves?|demonstrates?|cures?|caused?|due to|explains?|implies|likely|probably)\b", re.I)
NUM_RE = re.compile(r"(?<![\w.-])[-+]?\d+(?:[.,]\d+)+%?|(?<![\w.-])\d{2,}(?:\.\d+)?%?")
REQUIRED = {
    "facts": ["id:", "statement:", "derived by:", "inputs:"],
    "observations": ["id:", "what was looked at:", "procedure:", "observation:"],
    "sources": ["id:", "title:", "url:", "verified:"],
    "runs": ["id:", "command:", "git:", "host:", "data:"],
    "experiments": ["id:", "tests:", "prediction:"],
    "interpretations": ["id:", "claim:", "status:", "rests on:", "falsifier:"],
}
RESULTS_MARKER = "--- RESULTS ---"


def layer_of(p: Path) -> str:
    rel = p.relative_to(ROOT)
    parts = rel.parts
    if len(parts) == 1:
        return "root"
    if parts[0] == "facts" and len(parts) > 2 and parts[1] in ("data", "runs"):
        return parts[1]
    return parts[0]


def target_layer(ref: str, src: Path) -> str | None:
    if re.match(r"^[a-z]+://", ref) or ref.startswith("#") or ref.startswith("mailto:"):
        return None
    ref = ref.split("#")[0]
    if not ref:
        return None
    cand = (src.parent / ref).resolve() if not ref.startswith("paper/") else (REPO / ref).resolve()
    try:
        rel = cand.relative_to(ROOT)
    except ValueError:
        return "outside-paper"
    parts = rel.parts
    if not parts:
        return "root"
    if parts[0] == "facts" and len(parts) > 1 and parts[1] in ("data", "runs"):
        return parts[1]
    return parts[0] if len(parts) > 1 else "root"


def git(*args: str) -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def git_bytes(*args: str) -> bytes | None:
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True)
    return r.stdout if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    md_files = sorted(p for p in ROOT.rglob("*.md") if "tools" not in p.relative_to(ROOT).parts)

    # L1 + L3 + L7
    seen_ids: dict[str, Path] = {}
    for f in md_files:
        layer = layer_of(f)
        text = f.read_text(encoding="utf-8")
        rel = f.relative_to(REPO)
        allowed = ALLOWED.get(layer, set())
        for ref in LINK_RE.findall(text) + PATHISH_RE.findall(text):
            tl = target_layer(ref, f)
            if tl is None:
                continue
            if tl == "outside-paper" and layer not in ("root", "drafts"):
                errors.append(f"L1 {rel}: links outside paper/ ({ref}) — record layers must be self-contained")
            elif tl not in ("outside-paper",) and tl != layer and tl not in allowed:
                errors.append(f"L1 {rel}: layer '{layer}' may not reference '{tl}' ({ref})")
        own = re.search(r"^id:\s*([SFOIER]-\d{3,})", text, re.M)
        for m in ID_RE.finditer(text):
            rid = m.group(0)
            if own and rid == own.group(1):
                continue
            tl = ID_LAYER[m.group(1)]
            if tl != layer and tl not in allowed and layer != "root":
                errors.append(f"L1 {rel}: layer '{layer}' may not cite {rid} ({tl})")
        if layer in REQUIRED:
            if f.name.lower() in ("readme.md", "index.md"):
                continue
            for field in REQUIRED[layer]:
                if not re.search(rf"^{re.escape(field)}", text, re.M | re.I):
                    errors.append(f"L3 {rel}: missing required field '{field}'")
            if own:
                rid = own.group(1)
                if not f.name.startswith(rid):
                    errors.append(f"L7 {rel}: id {rid} does not match filename")
                if rid in seen_ids:
                    errors.append(f"L7 {rel}: duplicate id {rid} (also {seen_ids[rid].relative_to(REPO)})")
                seen_ids[rid] = f
        if layer in ("facts", "observations"):
            for m in CAUSAL.finditer(text):
                line = text[: m.start()].count("\n") + 1
                errors.append(f"L3 {rel}:{line}: interpretive word '{m.group(0)}' in a record layer")

    # L2
    data_dir = ROOT / "facts" / "data"
    manifest = data_dir / "MANIFEST.sha256"
    listed: dict[str, str] = {}
    if manifest.exists():
        for ln in manifest.read_text().splitlines():
            if ln.strip() and not ln.startswith("#"):
                h, _, name = ln.partition("  ")
                listed[name.strip()] = h.strip()
    if data_dir.exists():
        for p in sorted(data_dir.rglob("*")):
            if p.is_dir() or p.name in ("MANIFEST.sha256", ".gitkeep"):
                continue
            name = str(p.relative_to(data_dir))
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if name not in listed:
                errors.append(f"L2 facts/data/{name}: not in MANIFEST.sha256 (ingest with paper/tools/ingest.py)")
            elif listed[name] != h:
                errors.append(f"L2 facts/data/{name}: sha256 mismatch (manifest {listed[name][:12]}…, file {h[:12]}…) — data files are immutable")
        for name in listed:
            if not (data_dir / name).exists():
                errors.append(f"L2 MANIFEST lists missing file facts/data/{name}")

    # L4 append-only vs HEAD
    if git("rev-parse", "HEAD") is not None:
        for f in md_files:
            if layer_of(f) not in ("facts", "observations", "runs", "sources"):
                continue
            relp = str(f.relative_to(REPO))
            old = git_bytes("show", f"HEAD:{relp}")
            if old is None:
                continue
            new = f.read_bytes()
            if not new.startswith(old):
                errors.append(f"L4 {relp}: committed record was modified, not appended to (supersede with a new ID instead)")
        if data_dir.exists():
            for p in data_dir.rglob("*"):
                if p.is_file() and p.name != "MANIFEST.sha256":
                    relp = str(p.relative_to(REPO))
                    old = git_bytes("show", f"HEAD:{relp}")
                    if old is not None and old != p.read_bytes():
                        errors.append(f"L4 {relp}: committed data file changed")

    # L5 pre-registration freeze
    for f in md_files:
        if layer_of(f) != "experiments":
            continue
        text = f.read_text(encoding="utf-8")
        m = re.search(r"^registered_commit:\s*([0-9a-f]{7,40})", text, re.M)
        if not m:
            if RESULTS_MARKER in text:
                errors.append(f"L5 {f.relative_to(REPO)}: has results but no registered_commit")
            continue
        old = git("show", f"{m.group(1)}:{f.relative_to(REPO)}")
        if old is None:
            errors.append(f"L5 {f.relative_to(REPO)}: registered_commit {m.group(1)} does not contain this file")
            continue
        strip = lambda t: re.sub(r"^registered_commit:.*$", "", t.split(RESULTS_MARKER)[0], flags=re.M)
        if strip(old) != strip(text):
            errors.append(f"L5 {f.relative_to(REPO)}: pre-registered section changed since {m.group(1)}")

    # L6 numeric traceability
    fact_text = "\n".join(p.read_text(encoding="utf-8") for p in md_files if layer_of(p) == "facts")
    for f in md_files:
        if layer_of(f) not in ("interpretations", "drafts"):
            continue
        for ln_no, ln in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if ln.lstrip().startswith(("id:", "rests on:", "status:")):
                continue
            for num in NUM_RE.findall(ln):
                if ID_RE.search(num) or num in fact_text:
                    continue
                warnings.append(f"L6 {f.relative_to(REPO)}:{ln_no}: number '{num}' not found in any F-*")

    for e in errors:
        print("ERROR  ", e)
    for w in warnings:
        print("WARN   ", w)
    print(f"lint_links: {len(md_files)} files, {len(errors)} errors, {len(warnings)} warnings")
    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
