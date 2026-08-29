"""2D grid sweep of the two highest-leverage deal-closing knobs (local eval only).

Reuses score_sweep.group_scores / GROUPS / WEIGHTS verbatim. Sweeps
FAIR_FLOOR_FRACTION x RESCUE_TIME and reports, per cell, the per-group mean score
and deal-rate plus the tournament-mix-weighted OVERALL. Optional 3rd/4th args fix
other attributes (e.g. SECURE_BEST_FAIR) for the whole grid.

    uv run python eval/grid.py
    uv run python eval/grid.py SECURE_BEST_FAIR 1     # force the flag on
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from anchor import AnchorNegotiator
from score_sweep import group_scores, scenarios, GROUPS, WEIGHTS

import os


def _envlist(name, default):
    raw = os.environ.get(name)
    return [float(x) for x in raw.split(",")] if raw else default


FLOORS = _envlist("GRID_FLOORS", [0.92, 0.85, 0.78, 0.72, 0.65, 0.58])
RESCUES = _envlist("GRID_RESCUES", [0.95, 0.88, 0.82, 0.75])


def main() -> None:
    # Optional fixed overrides: pairs of (attr, value) from argv.
    args = sys.argv[1:]
    for i in range(0, len(args) - 1, 2):
        attr, raw = args[i], args[i + 1]
        try:
            val = float(raw)
            # treat bool-ish attrs explicitly
            if attr.endswith("FAIR") or attr.startswith("SECURE") or attr == "DECEPTION":
                val = bool(val)
        except ValueError:
            val = raw
        setattr(AnchorNegotiator, attr, val)
        print(f"[fixed] {attr} = {getattr(AnchorNegotiator, attr)!r}")

    scens = scenarios()
    cols = list(GROUPS)
    print(f"Grid FAIR_FLOOR x RESCUE_TIME; OVERALL weighted by {WEIGHTS}\n")
    header = f"{'floor':>6}{'rescue':>7}"
    header += "".join(f"{g[:9]:>16}" for g in cols)
    header += f"{'OVERALL':>10}{'deal':>6}"
    print(header)
    print(f"{'':>13}" + "".join(f"{'sc    dl':>16}" for _ in cols))
    best = None
    for floor in FLOORS:
        setattr(AnchorNegotiator, "FAIR_FLOOR_FRACTION", floor)
        for rescue in RESCUES:
            setattr(AnchorNegotiator, "RESCUE_TIME", rescue)
            r = group_scores(scens)
            overall = sum(WEIGHTS[g] * r[g][0] for g in cols)
            overall_deal = sum(WEIGHTS[g] * r[g][1] for g in cols)
            cells = "".join(f"{r[g][0]:>10.3f}{r[g][1]:>6.2f}" for g in cols)
            print(f"{floor:>6.2f}{rescue:>7.2f}{cells}{overall:>10.3f}{overall_deal:>6.2f}")
            if best is None or overall > best[0]:
                best = (overall, floor, rescue, r)
    print()
    o, f, rt, r = best
    print(f"BEST OVERALL={o:.3f} at FAIR_FLOOR_FRACTION={f}, RESCUE_TIME={rt}")
    print(f"  rivals={r['rivals'][0]:.3f}(deal {r['rivals'][1]:.2f})  "
          f"conceders={r['conceders'][0]:.3f}  firm={r['firm'][0]:.3f}  "
          f"model={r['model'][0]:.3f}")


if __name__ == "__main__":
    main()
