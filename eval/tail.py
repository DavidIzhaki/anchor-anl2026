"""Tail diagnostic (local eval only).

The tournament rewards reducing our BAD tail (low Q1 / catastrophic Min =
concealing forfeits), not just the mean. This runs the lab opponents x scenarios,
collects per-negotiation (score, advantage, concealing, deal), and reports the
score distribution (min/Q1/median/Q3), the count of near-zero / concealing-forfeit
negotiations, and the worst cases (which opponent / scenario / why).

    uv run python eval/tail.py [n_steps=100] [n_gen=24] [mode=realistic] [K=V ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

import labcore
from labcore import GROUPS, scenario_specs, build_tasks, init_worker, run_mech, make_scenario


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = max(0, min(len(xs) - 1, int(p * (len(xs) - 1))))
    return xs[i]


def main() -> None:
    args = sys.argv[1:]
    pos, ov = [], {}
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            ov[k] = (v == "True") if v in ("True", "False") else (
                float(v) if v.replace(".", "").replace("-", "").isdigit() else v)
        else:
            pos.append(a)
    n_steps = int(pos[0]) if len(pos) > 0 else 100
    n_gen = int(pos[1]) if len(pos) > 1 else 24
    mode = pos[2] if len(pos) > 2 else "realistic"

    init_worker(ov, n_steps)
    specs = scenario_specs(n_gen=n_gen, mode=mode)
    rows = []  # (score, adv, con, deal, opp, spec)
    for group, opp, spec, first in build_tasks(specs):
        try:
            scen = make_scenario(spec)
            r = run_mech(scen, opp, first, n_steps)
        except Exception:
            continue
        rows.append((r["score"], r["adv"], r["con"], r["deal"],
                     opp.split(".")[-1], spec[0] if spec[0] == "local" else f"gen{spec[1]}"))

    scores = [r[0] for r in rows]
    print(f"overrides={ov} n_steps={n_steps} n={len(rows)}")
    print(f"min={min(scores):.3f} Q1={pct(scores,.25):.3f} median={pct(scores,.5):.3f} "
          f"Q3={pct(scores,.75):.3f} mean={sum(scores)/len(scores):.3f}")
    forfeit = [r for r in rows if r[2] < 0.10]      # concealing nearly forfeited
    nearzero = [r for r in rows if r[0] < 0.40]      # below a normal no-deal floor
    lowcon = sum(1 for r in rows if r[2] < 0.40)
    print(f"concealing-forfeit (con<0.10): {len(forfeit)}  | score<0.40: {len(nearzero)} "
          f"| con<0.40: {lowcon}  ({100*lowcon/len(rows):.1f}% of negs)")
    print("\nWorst 15 negotiations (score | adv | con | deal | opponent | scenario):")
    for r in sorted(rows)[:15]:
        print(f"  {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | {'Y' if r[3] else '-'} | {r[4]:<20} | {r[5]}")
    # concealing distribution
    cons = sorted(r[2] for r in rows)
    print(f"\nconcealing: min={cons[0]:.3f} Q1={pct(cons,.25):.3f} median={pct(cons,.5):.3f} mean={sum(cons)/len(cons):.3f}")


if __name__ == "__main__":
    main()
