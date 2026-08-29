"""Offline trainer for the state-conditioned concession policy (heavy compute).

Design for a genuine shot at beating the (proven-optimal) heuristic:
  1. SEED the policy weights by least-squares fitting the heuristic Boulware
     target curve, so the seed policy ~= the heuristic (we start AT the bar).
     The opponent-feature weights start at 0 (pure time-based, like Boulware);
     the evolution strategy then explores adding STATE-DEPENDENCE on top.
  2. Evolution strategy (mu/lambda with sigma decay) over the 9 weights.
  3. Fitness = lab mean + bottom-quartile bonus, deal-rate guarded; end-game
     safety net always on. Many scenarios per eval to reduce noise.

The result is FROZEN into anchor.py -- no runtime learning/memory.

    uv run python eval/train_policy.py [gens=20] [lambda=10] [n_gen=10]
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

import anchor
from labcore import GROUPS, WEIGHTS, scenario_specs, build_tasks, init_worker, task

A = anchor.AnchorNegotiator
DIM = 9


def evaluate(specs):
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
    obj = overall + 0.3 * q1 - (0.5 if deal < 0.80 else 0.0)
    return obj, overall, q1, deal


def obj_of(W, specs):
    A.USE_LEARNED_POLICY = True
    A.POLICY_WEIGHTS = [float(x) for x in W]
    return evaluate(specs)


def fit_seed_to_boulware():
    """Least-squares fit the time-only weights so sigmoid(W.f(t)) ~= the
    heuristic Boulware target fraction. Opponent-feature weights start at 0."""
    ff = A.FAIR_FLOOR_FRACTION
    exp = A.CONCESSION_EXPONENT
    rt = A.RESCUE_TIME
    ts = np.linspace(0.0, rt * 0.999, 40)
    fracs = []
    for t in ts:
        conc = 1.0 - (t / rt) ** (1.0 / exp)
        fr = ff + (1.0 - ff) * conc           # target fraction in [ff, 1]
        fracs.append(min(0.985, max(0.5, fr)))
    logit = np.log(np.array(fracs) / (1.0 - np.array(fracs)))
    X = np.stack([np.ones_like(ts), ts, ts ** 2, ts ** 3], axis=1)
    w4, *_ = np.linalg.lstsq(X, logit, rcond=None)
    return np.array([w4[0], w4[1], w4[2], w4[3], 0, 0, 0, 0, 0], dtype=float)


def main():
    gens = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    lam = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    n_gen = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    init_worker({}, 100)
    specs = scenario_specs(n_gen=n_gen, mode="realistic")
    np.random.seed(7)

    A.USE_LEARNED_POLICY = False
    base = evaluate(specs)
    print(f"HEURISTIC baseline: obj={base[0]:.3f} mean={base[1]:.3f} q1={base[2]:.3f} deal={base[3]:.2f}", flush=True)

    best = fit_seed_to_boulware()
    best_obj, bm, bq, bd = obj_of(best, specs)
    print(f"seed (fit to Boulware): obj={best_obj:.3f} mean={bm:.3f} q1={bq:.3f} deal={bd:.2f}", flush=True)
    print(f"seed weights={[round(x,3) for x in best]}", flush=True)

    # Evolution strategy. Time weights (0-3) get smaller steps (seed is good);
    # opponent-feature weights (4-8) get larger steps to explore state-dependence.
    sigma = np.array([0.3, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0])
    for g in range(gens):
        cands = [best + sigma * np.random.randn(DIM) for _ in range(lam)]
        evals = [(obj_of(c, specs), c) for c in cands]
        evals.sort(key=lambda e: e[0][0], reverse=True)
        (o, m, q, d), w = evals[0]
        if o > best_obj:
            best_obj, best, bm, bq, bd = o, w, m, q, d
        sigma *= 0.88
        print(f"gen {g}: best_obj={best_obj:.3f} mean={bm:.3f} q1={bq:.3f} deal={bd:.2f}", flush=True)

    out = {"weights": [float(x) for x in best], "obj": best_obj, "mean": bm,
           "q1": bq, "deal": bd, "baseline_obj": base[0], "baseline_mean": base[1],
           "n_gen": n_gen}
    (ROOT / "advisers" / "policy_weights.json").write_text(json.dumps(out, indent=2))
    print("\n=== RESULT ===")
    print(f"heuristic mean={base[1]:.3f}  |  learned mean={bm:.3f}  (delta {bm-base[1]:+.3f})")
    print(f"weights={[round(x,3) for x in best]}")
    print("saved -> advisers/policy_weights.json")


if __name__ == "__main__":
    main()
