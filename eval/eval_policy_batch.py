"""Evaluate a BATCH of policies in one process (amortises the negmas import).

Input: a JSON file {"n_gen":N, "seed_base":S, "cands":[w-list | null, ...]}
(null = heuristic). Output (stdout): JSON list of [obj,mean,q1,deal] per candidate.
Used by the parallel ES driver (train_es.py).

    python eval/eval_policy_batch.py <spec.json>
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


def score_one(tasks):
    scores = []
    agg = {g: [0.0, 0, 0] for g in GROUPS}
    for group, opp, spec, first in tasks:
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
    return [obj, overall, q1, deal]


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    init_worker({}, cfg.get("nsteps", 100))
    specs = scenario_specs(n_gen=cfg["n_gen"], mode=cfg.get("mode", "realistic"),
                           seed_base=cfg.get("seed_base", 7000))
    tasks = build_tasks(specs)
    out = []
    for W in cfg["cands"]:
        if W is None:
            A.USE_LEARNED_POLICY = False
        else:
            A.USE_LEARNED_POLICY = True
            A.POLICY_WEIGHTS = [float(x) for x in W]
        out.append(score_one(tasks))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
