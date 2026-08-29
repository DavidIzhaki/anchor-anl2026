"""Parallel experiment CLI for Anchor (local eval only).

    uv run python eval/lab.py [n_steps=100] [n_gen=18] [workers=20] [K=V ...]

K=V overrides set AnchorNegotiator class attributes for OUR agent (e.g.
SECURE_BEST_FAIR=False, FAIR_FLOOR_FRACTION=0.72, ADAPTIVE_CONCESSION=True).
Values parse as float / bool (True/False) / str. Reports per-group mean Score,
deal-rate, Advantage, Concealing, opponent score, and the weighted OVERALL.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

import labcore
from labcore import GROUPS, WEIGHTS, scenario_specs, build_tasks, init_worker, task


def parse_val(v: str):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def main() -> None:
    args = sys.argv[1:]
    pos, overrides = [], {}
    seed_base = 7000
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            if k == "SEED_BASE":  # eval-only: vary the scenario draw (not an attr)
                seed_base = int(float(v))
            else:
                overrides[k] = parse_val(v)
        else:
            pos.append(a)
    n_steps = int(pos[0]) if len(pos) > 0 else 100
    n_gen = int(pos[1]) if len(pos) > 1 else 18
    mode = pos[2] if len(pos) > 2 else "realistic"

    specs = scenario_specs(n_gen=n_gen, mode=mode, seed_base=seed_base)
    tasks = build_tasks(specs)
    # Serial execution: the in-process worker pool corrupted negmas global state
    # across the mixed task stream (Tough deal-rate 0.61 serial vs 0.03 pooled).
    # Serial is deterministic and matches bench.py exactly. For throughput we run
    # multiple lab.py invocations as separate OS processes (each independent).
    init_worker(overrides, n_steps)
    t0 = time.time()
    results = [task(t) for t in tasks]
    dt = time.time() - t0

    # Aggregate per group.
    agg = {g: dict(score=0.0, deal=0.0, adv=0.0, con=0.0, opp=0.0, n=0, err=0)
           for g in GROUPS}
    per_opp = {}
    for group, opp, r in results:
        a = agg[group]
        if r is None or "error" in (r or {}):
            a["err"] += 1
            continue
        a["score"] += r["score"]; a["deal"] += 1 if r["deal"] else 0
        a["adv"] += r["adv"]; a["con"] += r["con"]; a["opp"] += r["opp_score"]
        a["n"] += 1
        po = per_opp.setdefault(opp, dict(score=0.0, deal=0.0, n=0))
        po["score"] += r["score"]; po["deal"] += 1 if r["deal"] else 0; po["n"] += 1

    print(f"n_steps={n_steps} n_gen={n_gen} mode={mode} "
          f"tasks={len(tasks)} time={dt:.1f}s overrides={overrides}")
    print(f"{'group':>10}{'score':>9}{'deal':>7}{'adv':>8}{'con':>8}{'oppScore':>10}{'n':>6}{'err':>5}")
    overall = overall_deal = 0.0
    for g in GROUPS:
        a = agg[g]
        n = max(1, a["n"])
        sc, dl = a["score"] / n, a["deal"] / n
        print(f"{g:>10}{sc:>9.3f}{dl:>7.2f}{a['adv']/n:>8.3f}{a['con']/n:>8.3f}"
              f"{a['opp']/n:>10.3f}{a['n']:>6}{a['err']:>5}")
        overall += WEIGHTS[g] * sc
        overall_deal += WEIGHTS[g] * dl
    print(f"{'OVERALL':>10}{overall:>9.3f}{overall_deal:>7.2f}   (weights {WEIGHTS})")

    # Per-opponent (sorted worst-first to spotlight weak matchups).
    print(f"\n{'opponent':>22}{'score':>9}{'deal':>7}{'n':>5}")
    for opp, po in sorted(per_opp.items(), key=lambda kv: kv[1]["score"] / max(1, kv[1]["n"])):
        n = max(1, po["n"])
        print(f"{opp:>22}{po['score']/n:>9.3f}{po['deal']/n:>7.2f}{po['n']:>5}")


if __name__ == "__main__":
    main()
