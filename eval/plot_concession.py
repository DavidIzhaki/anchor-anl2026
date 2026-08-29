"""Regenerate report/concession.png from the LIVE AnchorNegotiator constants.

Plots the utility target over relative time (normalised: reservation=0, max=1) so
the figure always matches the shipped parameters.

    uv run python eval/plot_concession.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anchor import AnchorNegotiator as A


def target(t: float) -> float:
    """Mirror of _target_utility with reservation=0, u_max=1 (so y is the
    fraction of our utility range)."""
    floor = A.FAIR_FLOOR_FRACTION
    if t < A.RESCUE_TIME:
        tt = t / A.RESCUE_TIME
        concession = 1.0 - tt ** (1.0 / A.CONCESSION_EXPONENT)
        return floor + (1.0 - floor) * concession
    frac = (t - A.RESCUE_TIME) / (1.0 - A.RESCUE_TIME)
    rf = A.RESCUE_FLOOR_FRACTION
    return rf + (floor - rf) * (1.0 - frac)


def main() -> None:
    ts = np.linspace(0, 1, 1000)
    ys = [target(t) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(ts, ys, lw=2.2, color="#1f4e79")
    ax.axhline(A.FAIR_FLOOR_FRACTION, ls="--", lw=1, color="#888",
               label=f"fair floor ({A.FAIR_FLOOR_FRACTION:.2f})")
    ax.axvline(A.RESCUE_TIME, ls=":", lw=1, color="#c0504d",
               label=f"rescue start (t={A.RESCUE_TIME:.2f})")
    ax.axhline(A.RESCUE_FLOOR_FRACTION, ls="--", lw=1, color="#cbb",
               label=f"rescue floor ({A.RESCUE_FLOOR_FRACTION:.2f})")
    ax.set_xlabel("relative time")
    ax.set_ylabel("utility target (fraction of range)")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = ROOT / "report" / "concession.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
