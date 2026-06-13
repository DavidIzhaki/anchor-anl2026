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
    "conceders": ["negmas.sao.BoulwareTBNegotiator", "negmas.sao.ConcederTBNegotiator",
                  "negmas.sao.LinearTBNegotiator", "negmas.sao.NiceNegotiator"],
    "firm": ["negmas.sao.ToughNegotiator", "negmas.sao.MiCRONegotiator",
             "negmas.sao.HybridNegotiator", "negmas.sao.NaiveTitForTatNegotiator"],
    "model": ["examples.map.MAPNeg", "examples.boa.BOANeg",
              "examples.simple.SimpleNegotiator"],
}


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


def main() -> None:
    attr = sys.argv[1]
    values = [float(v) for v in sys.argv[2:]] or [getattr(AnchorNegotiator, attr)]
    scens = scenarios()
    print(f"Sweeping {attr}; metric = OUR mean score (Advantage+Concealing)\n")
    print(f"{attr:>10}{'conceders':>22}{'firm':>22}{'model':>22}{'OVERALL':>10}")
    print(f"{'':>10}{'score  deal':>22}{'score  deal':>22}{'score  deal':>22}")
    for v in values:
        setattr(AnchorNegotiator, attr, v)
        r = group_scores(scens)
        overall = sum(r[g][0] for g in GROUPS) / len(GROUPS)
        cells = "".join(f"{r[g][0]:>13.3f}{r[g][1]:>9.2f}" for g in GROUPS)
        print(f"{v:>10.2f}{cells}{overall:>10.3f}")


if __name__ == "__main__":
    main()
