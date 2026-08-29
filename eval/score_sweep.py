"""Tournament-score sweep (local eval only).

The ANL tournament ranks by our ABSOLUTE mean score (Advantage + Concealing),
not by pairwise margin. A no-deal scores ~0.5 (Concealing only); a balanced deal
scores ~1.0. So closing deals matters even when the opponent also gains.

This sweeps a chosen AnchorNegotiator constant and reports, over a representative
opponent mix x scenarios, our mean SCORE, split by opponent style:
  * conceders  (we should extract high Advantage)
  * firm       (closing a balanced deal beats a no-deal)
  * model-capable (the contested ones)

    uv run python eval/score_sweep.py RESCUE_FLOOR_FRACTION 0.9 0.6 0.4 0.2 0.0
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from negmas.inout import Scenario
from negmas.preferences.generators import generate_multi_issue_ufuns

import anchor
from anchor import AnchorNegotiator
from bench import run_one, SCENARIOS_DIR

GROUPS = {
    # The live ANL field is ~38 firm, sophisticated student agents. `rivals` is
    # our proxy for them (self-play + firm-modular agents) and is the group that
    # must improve; the OVERALL objective weights it most heavily (see WEIGHTS).
    "rivals": ["rivals.AnchorRival", "rivals.FirmTopNeg", "rivals.FirmBestNeg"],
    "conceders": ["negmas.sao.BoulwareTBNegotiator", "negmas.sao.ConcederTBNegotiator",
                  "negmas.sao.LinearTBNegotiator", "negmas.sao.NiceNegotiator"],
    "firm": ["negmas.sao.ToughNegotiator", "negmas.sao.MiCRONegotiator",
             "negmas.sao.HybridNegotiator", "negmas.sao.NaiveTitForTatNegotiator"],
    "model": ["whale.WhaleNegotiator", "examples.boa.BOANeg",
              "examples.simple.SimpleNegotiator"],
}

# Tournament-mix weighting of the groups for the OVERALL objective. The real
# field is overwhelmingly firm rivals; pure conceders barely exist in it, so we
# must NOT let their high extraction scores dominate the objective (that was the
# original tuning trap that produced an over-firm, low-deal-rate agent).
WEIGHTS = {"rivals": 0.55, "firm": 0.25, "conceders": 0.10, "model": 0.10}


def scenarios(n_gen: int = 10) -> list[Scenario]:
    out = []
    for path in sorted(SCENARIOS_DIR.iterdir()):
        if path.is_dir():
            s = Scenario.load(path, ignore_discount=True)
            if s is not None:
                out.append(s)
    for k in range(n_gen):
        rng = random.Random(3000 + k)
        u = generate_multi_issue_ufuns(n_issues=rng.randint(1, 4), n_values=(3, 8),
                                       ufun_names=("A", "B"), rational_fractions=[1.0, 1.0])
        out.append(Scenario(outcome_space=u[0].outcome_space, ufuns=u))
    return out


def group_scores(scens) -> dict:
    res = {}
    for g, opps in GROUPS.items():
        sc = dl = n = 0.0
        for opp in opps:
            for s in scens:
                for first in (True, False):
                    try:
                        r = run_one(s, opp, first)
                    except Exception:
                        continue
                    sc += r["mine"]["Score"]; dl += 1 if r["agreement"] else 0; n += 1
        res[g] = (sc / n, dl / n) if n else (0.0, 0.0)
    return res


def _coerce(v: str):
    """Parse a sweep value as float when possible, else keep the raw string
    (so we can sweep string attributes like SELECTION_MODE / MODEL_WEIGHTING)."""
    try:
        return float(v)
    except ValueError:
        return v


def main() -> None:
    attr = sys.argv[1]
    values = [_coerce(v) for v in sys.argv[2:]] or [getattr(AnchorNegotiator, attr)]
    scens = scenarios()
    cols = list(GROUPS)
    print(f"Sweeping {attr}; metric = OUR mean score (Advantage+Concealing)")
    print(f"OVERALL = weighted by tournament mix {WEIGHTS}; (unwt) = old flat mean\n")
    print(f"{attr:>10}" + "".join(f"{g:>22}" for g in cols)
          + f"{'OVERALL':>12}{'(unwt)':>9}")
    print(f"{'':>10}" + "".join(f"{'score  deal':>22}" for _ in cols)
          + f"{'score deal':>12}")
    for v in values:
        setattr(AnchorNegotiator, attr, v)
        r = group_scores(scens)
        overall = sum(WEIGHTS[g] * r[g][0] for g in cols)
        overall_deal = sum(WEIGHTS[g] * r[g][1] for g in cols)
        unwt = sum(r[g][0] for g in cols) / len(cols)
        cells = "".join(f"{r[g][0]:>13.3f}{r[g][1]:>9.2f}" for g in cols)
        vfmt = f"{v:>10.2f}" if isinstance(v, float) else f"{str(v):>10}"
        print(f"{vfmt}{cells}{overall:>8.3f}{overall_deal:>6.2f}{unwt:>9.3f}")


if __name__ == "__main__":
    main()
