"""Joint parameter search (heavy compute, local eval only).

Manual sweeps were mostly 1D; the optimum may be a JOINT setting with interactions.
This random-searches the continuous parameter vector, evaluating each candidate on
the lab opponents x scenarios, and keeps the best by an objective that favors both
mean score AND consistency (a high bottom quartile -- the leaders' hallmark).

    uv run python eval/optimize.py [n_samples=80] [n_gen=12]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

import anchor
from labcore import GROUPS, WEIGHTS, scenario_specs, build_tasks, init_worker, task

# Search ranges per parameter (name -> (lo, hi)).
SPACE = {
    "FAIR_FLOOR_FRACTION": (0.60, 0.98),
    "RESCUE_TIME": (0.80, 0.97),
    "RESCUE_FLOOR_FRACTION": (0.05, 0.35),
    "SECURE_BEST_TIME": (0.93, 0.99),
    "CONCESSION_EXPONENT": (0.05, 0.40),
    "SECURE_ACCEPT_FLOOR": (0.0, 0.15),
    "CAPTURE_FRACTION": (0.75, 0.95),
}


def evaluate(specs):
    """Run the lab, return (objective, mean, q1, deal)."""
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
    # Objective: mean, plus a consistency bonus (bottom quartile), with a deal-rate
    # guard so we never trade back into the no-deal failure mode.
    obj = overall + 0.3 * q1 - (0.5 if deal < 0.80 else 0.0)
    return obj, overall, q1, deal


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    init_worker({}, 100)
    specs = scenario_specs(n_gen=n_gen, mode="realistic")
    rng = random.Random(12345)
    names = list(SPACE)
    # Always include the current shipped config as candidate 0 (the incumbent).
    incumbent = {n: getattr(anchor.AnchorNegotiator, n) for n in names}
    results = []
    for s in range(n_samples + 1):
        cfg = incumbent if s == 0 else {
            n: round(rng.uniform(*SPACE[n]), 3) for n in names}
        for n, v in cfg.items():
            setattr(anchor.AnchorNegotiator, n, v)
        obj, mean, q1, deal = evaluate(specs)
        results.append((obj, mean, q1, deal, dict(cfg)))
        tag = "INCUMBENT" if s == 0 else f"sample {s}"
        print(f"{tag:>10}: obj={obj:.3f} mean={mean:.3f} q1={q1:.3f} deal={deal:.2f}",
              flush=True)
    results.sort(reverse=True)
    print("\n=== TOP 5 ===")
    for obj, mean, q1, deal, cfg in results[:5]:
        print(f"obj={obj:.3f} mean={mean:.3f} q1={q1:.3f} deal={deal:.2f}  {cfg}")
    print(f"\nINCUMBENT obj was: {[r for r in results if r[4]==incumbent][0][0]:.3f}")


if __name__ == "__main__":
    main()
