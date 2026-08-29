"""Final out-of-sample validation of the trained policy vs the heuristic.

Loads advisers/policy_weights.json and compares the learned policy against the
heuristic on a LARGE held-out scenario set (seeds the trainer never used), across
realistic + mixed modes. Prints means/deal/Q1 and a ship recommendation. The
trained policy should ship ONLY if it beats the heuristic out-of-sample by a real
margin (> noise).

    python eval/validate_policy.py [n_gen=40]
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import anchor
from labcore import GROUPS, WEIGHTS, scenario_specs, build_tasks, init_worker, task

A = anchor.AnchorNegotiator


def run(specs):
    scores = []
    agg = {g: [0.0, 0, 0] for g in GROUPS}
    for group, opp, spec, first in build_tasks(specs):
        r = task((group, opp, spec, first))[2]
        if not r or "error" in r:
            continue
        scores.append(r["score"])
        a = agg[group]
        a[0] += r["score"]; a[1] += 1 if r["deal"] else 0; a[2] += 1
    overall = sum(WEIGHTS[g] * (agg[g][0] / max(1, agg[g][2])) for g in GROUPS)
    deal = sum(WEIGHTS[g] * (agg[g][1] / max(1, agg[g][2])) for g in GROUPS)
    scores.sort()
    q1 = scores[len(scores) // 4] if scores else 0.0
    return overall, deal, q1


def main():
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    init_worker({}, 100)
    pw = json.loads((ROOT / "advisers" / "policy_weights.json").read_text())
    weights = pw["weights"]
    print(f"loaded policy: train val_mean={pw.get('val_mean')} (gen {pw.get('gen')})")
    # Two held-out sets the trainer never saw (train used seed_base 10000+; val 90000).
    for mode, sb in [("anl", 500000), ("anl", 700000), ("realistic", 500000)]:
        specs = scenario_specs(n_gen=n_gen, mode=mode, seed_base=sb)
        A.USE_LEARNED_POLICY = False
        h = run(specs)
        A.USE_LEARNED_POLICY = True
        A.POLICY_WEIGHTS = [float(x) for x in weights]
        l = run(specs)
        print(f"[{mode}] heuristic mean={h[0]:.4f} deal={h[1]:.2f} q1={h[2]:.3f} | "
              f"learned mean={l[0]:.4f} deal={l[1]:.2f} q1={l[2]:.3f} | "
              f"delta={l[0]-h[0]:+.4f}")
    print("\nSHIP the learned policy only if delta is clearly positive (> ~0.01) "
          "on BOTH modes; otherwise keep the heuristic.")


if __name__ == "__main__":
    main()
