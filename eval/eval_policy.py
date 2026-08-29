"""Evaluate ONE policy (or the heuristic) on the lab; print [obj,mean,q1,deal] JSON.

Run as a separate OS process by the parallel ES driver (train_es.py) so rollouts
parallelise across cores without the in-process-Pool global-state corruption.

    python eval/eval_policy.py <weights.json | "heuristic"> [n_gen=10]
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


def main():
    arg = sys.argv[1]
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    init_worker({}, 100)
    specs = scenario_specs(n_gen=n_gen, mode="realistic")
    if arg == "heuristic":
        anchor.AnchorNegotiator.USE_LEARNED_POLICY = False
    else:
        W = json.loads(Path(arg).read_text())
        anchor.AnchorNegotiator.USE_LEARNED_POLICY = True
        anchor.AnchorNegotiator.POLICY_WEIGHTS = [float(x) for x in W]

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
    print(json.dumps([obj, overall, q1, deal]))


if __name__ == "__main__":
    main()
