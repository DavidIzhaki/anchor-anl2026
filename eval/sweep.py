"""Parameter sweep for AnchorNegotiator vs a benchmark opponent (local eval only).

Sweeps the anti-exploitation knobs (fair-floor fraction and concession exponent)
and reports the aggregate result per combination, so we can pick the operating
point that beats the benchmark by the largest margin without causing no-deals.

    uv run python eval/sweep.py                      # vs benchmarkNegotiator
    uv run python eval/sweep.py examples.boa.BOANeg
"""

from __future__ import annotations

import sys

from bench import evaluate  # noqa: E402  (sys.path set inside bench)
import anchor
from anchor import AnchorNegotiator

FLOORS = [0.75, 0.85, 0.92, 0.97]
MODES = ["greedy", "nash", "opponent"]


def main() -> None:
    opponent = sys.argv[1] if len(sys.argv) > 1 else "examples.map.MAPNeg"
    AnchorNegotiator.CONCESSION_EXPONENT = 0.10
    print(f"Sweep vs {opponent}  (exp=0.10)\n")
    print(f"{'floor':>7}{'mode':>10}{'deals':>8}{'wins':>7}{'myAdv':>8}"
          f"{'myScore':>9}{'opScore':>9}{'margin':>9}")
    best = None
    for mode in MODES:
        for floor in FLOORS:
            AnchorNegotiator.FAIR_FLOOR_FRACTION = floor
            AnchorNegotiator.SELECTION_MODE = mode
            _, agg = evaluate(opponent)
            n = agg["n"]
            margin = agg["margin"] / n
            print(f"{floor:>7.2f}{mode:>10}{agg['deals']:>5}/{n:<2}"
                  f"{agg['wins']:>5}/{n:<2}{agg['myAdv']/n:>8.3f}"
                  f"{agg['myScore']/n:>9.3f}{agg['opScore']/n:>9.3f}{margin:>+9.3f}")
            if best is None or margin > best[0]:
                best = (margin, floor, mode, agg["wins"], agg["deals"], n)
        print()
    print(f"Best: margin={best[0]:+.3f} at floor={best[1]} mode={best[2]} "
          f"(wins {best[3]}/{best[5]}, deals {best[4]}/{best[5]})")


if __name__ == "__main__":
    main()
