#!/usr/bin/env python3
"""WP-2a: Spiffy draft-graph calibration + ceiling readout (arXiv:2509.18085 Alg. 2)
over `diffusion-bench llada --dump-traces` JSONL, plus the JOT pre-experiment numbers
(roadmap §3 entry condition) from the same traces.

The ceiling readout is the WP-2a Phase-3 go/no-go input: under temp-0 thresholded
Γ + Δ decoding, a draft formula {(i, j=1)} predicts the NEXT step's transition as
"the positions ranked i by confidence among currently-masked unmask to their current
argmax tokens". A step is draftable iff the actual next transition equals some
calibrated formula exactly AND no Δ edit fired (edits abort speculation — recorded
deviation). Projected forward savings = draftable-step fraction (each accepted level-1
draft skips one forward; deeper levels compound but level-1 dominates).

Usage:
  python3 Tools/build_draft_graph.py --traces scratch/wp2a_traces.jsonl \
      [--out scratch/draft_graph.json] [--budget 3 5 8] [--jot-k 4]
"""
import json
import argparse
import hashlib
from collections import Counter, defaultdict


def load_traces(path):
    """-> {promptId: {block: [step records sorted by step]}}"""
    runs = defaultdict(lambda: defaultdict(list))
    for line in open(path):
        r = json.loads(line)
        runs[r["promptId"]][r["block"]].append(r)
    for blocks in runs.values():
        for steps in blocks.values():
            steps.sort(key=lambda r: r["step"])
    return runs


def formula_for_transition(prev, cur):
    """The (i, j) multiset that WOULD have drafted `cur`'s Γ transition from `prev`'s state.

    i = rank of each transferred position among prev's masked positions by prev confidence
    (descending; rank 1 = most confident — Spiffy's position rank). j = 1 iff the token
    written at cur equals prev's argmax at that position, else None (undraftable at temp 0).
    Returns (frozenset of i-ranks, draftable: bool, edited: bool).
    """
    masked_prev = [p for p, m in enumerate(prev["masked"]) if m]
    order = sorted(masked_prev, key=lambda p: -prev["conf"][p])
    rank = {p: k + 1 for k, p in enumerate(order)}
    gamma_positions = [p for p, g in enumerate(cur["gamma"]) if g]
    edited = any(cur["delta"])
    if not gamma_positions:
        return frozenset(), False, edited
    ranks = []
    draftable = True
    for p in gamma_positions:
        if p not in rank:
            return frozenset(), False, edited  # inconsistent trace (shouldn't happen)
        ranks.append(rank[p])
        if cur["x0"][p] != prev["x0"][p]:
            draftable = False  # the eventual token was not prev's argmax (j > 1)
    return frozenset(ranks), draftable, edited


def category(prompt_id):
    parts = prompt_id.split("-")
    return parts[1] if prompt_id.startswith("cal-") and len(parts) >= 2 else "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--out", default="scratch/draft_graph.json")
    ap.add_argument("--budget", type=int, nargs="+", default=[3, 5, 8])
    ap.add_argument("--jot-k", type=int, default=4,
                    help="stability window for the JOT pre-experiment")
    args = ap.parse_args()

    runs = load_traces(args.traces)

    # ---- Part 1: transition statistics (the rewound-trace analogue at temp 0) ----
    formula_counts = Counter()          # frozenset(i-ranks) -> count (draftable only)
    per_cat = defaultdict(lambda: [0, 0, 0, 0])  # cat -> [steps, draftable, token_miss, edit_abort]
    for pid, blocks in runs.items():
        cat = category(pid)
        for steps in blocks.values():
            for prev, cur in zip(steps, steps[1:]):
                if not any(prev["masked"]):
                    continue  # refinement tail — nothing to draft
                ranks, draftable, edited = formula_for_transition(prev, cur)
                stats = per_cat[cat]
                stats[0] += 1
                if edited:
                    stats[3] += 1
                    continue  # Δ aborts speculation (recorded deviation)
                if not ranks:
                    continue
                if draftable:
                    stats[1] += 1
                    formula_counts[ranks] += 1
                else:
                    stats[2] += 1

    total_steps = sum(v[0] for v in per_cat.values())
    total_draftable = sum(v[1] for v in per_cat.values())

    # ---- Part 2/3: formula selection (degree-1-accumulation degenerates to frequency
    # at level 1; deeper levels need runtime chaining — only built on GO) ----
    top_formulas = formula_counts.most_common(max(args.budget) if args.budget else 8)
    graphs = {}
    for D in args.budget:
        chosen = [sorted(f) for f, _ in top_formulas[:D]]
        covered = sum(c for f, c in top_formulas[:D])
        graphs[f"D{D}"] = {
            "formulas": chosen,
            "coveredSteps": covered,
            "coverageOfDraftable": covered / max(total_draftable, 1),
            "projectedForwardSavings": covered / max(total_steps, 1),
        }

    # ---- JOT pre-experiment (roadmap §3 entry condition) ----
    k = args.jot_k
    jot_stable = jot_positions = jot_collisions = 0
    for blocks in runs.values():
        for steps in blocks.values():
            B = len(steps[0]["conf"]) if steps else 0
            for p in range(B):
                unmask_step = next(
                    (t for t, s in enumerate(steps) if s["gamma"][p]), None)
                if unmask_step is None or unmask_step < k:
                    continue
                jot_positions += 1
                history = [steps[t]["x0"][p] for t in range(unmask_step - k, unmask_step)]
                if len(set(history)) == 1:
                    jot_stable += 1
                    if any(s["delta"][p] for s in steps[unmask_step:]):
                        jot_collisions += 1

    # ---- Report ----
    print("== WP-2a Spiffy ceiling readout (temp-0, thresholded Γ + Δ decoding) ==")
    print(f"traces: {len(runs)} prompts, {total_steps} draftable-context steps")
    print(f"{'cat':>7} {'steps':>6} {'draftable%':>10} {'token-miss%':>11} {'edit-abort%':>11}")
    for cat, (s, d, m, e) in sorted(per_cat.items()):
        if s:
            print(f"{cat:>7} {s:>6} {100*d/s:>9.1f}% {100*m/s:>10.1f}% {100*e/s:>10.1f}%")
    print(f"\noverall draftable-step fraction: {100*total_draftable/max(total_steps,1):.1f}%")
    for name, g in graphs.items():
        print(f"  {name}: covers {100*g['coverageOfDraftable']:.1f}% of draftable "
              f"-> projected forward savings {100*g['projectedForwardSavings']:.1f}%")
    print("\nGO/NO-GO (plan §Phase 3): GO iff best projected savings >= 10%")

    print(f"\n== JOT pre-experiment (k={k}) ==")
    if jot_positions:
        print(f"positions unmasked after >= {k} steps: {jot_positions}")
        print(f"  prediction-stable for {k} steps before unmask: "
              f"{100*jot_stable/jot_positions:.1f}%")
        print(f"  of those, later edited by Δ (freeze collision): "
              f"{100*jot_collisions/max(jot_stable,1):.1f}%")
    else:
        print("no qualifying positions (blocks settle too fast at this k)")

    payload = {
        "sourceTraces": args.traces,
        "graphs": graphs,
        "perCategory": {c: {"steps": v[0], "draftable": v[1],
                            "tokenMiss": v[2], "editAbort": v[3]}
                        for c, v in per_cat.items()},
        "jot": {"k": k, "positions": jot_positions,
                "stable": jot_stable, "collisions": jot_collisions},
    }
    payload["hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out} (hash {payload['hash']})")


if __name__ == "__main__":
    main()
