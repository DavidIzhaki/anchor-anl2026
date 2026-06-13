"""Broad robustness arena (local eval only; NOT in the submission zip).

Tests our agent against a large, diverse roster of real NegMAS opponents (plus
the local benchmark examples.map.MAPNeg) across the 7 local scenarios AND many
randomly generated scenarios spanning cooperative -> competitive structure.

Reports, sorted by our mean margin:
  * a per-opponent summary, and
  * a cooperative vs competitive split (by the utility correlation of each
    scenario), overall and for the model-capable opponents we most contest.

    uv run python eval/arena.py             # default: 24 generated scenarios
    uv run python eval/arena.py 48          # 48 generated scenarios
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from negmas.helpers import get_class
from negmas.inout import Scenario
from negmas.preferences.generators import generate_multi_issue_ufuns

from bench import run_one, SCENARIOS_DIR

# ~30 real NegMAS opponents spanning conceders, hardliners, tit-for-tat,
# time-based, oriented, following and strong hybrids -- plus the benchmark and
# the three skeleton examples.
ROSTER = [
    "examples.map.MAPNeg",
    "examples.boa.BOANeg", "examples.map.MAPNeg", "examples.simple.SimpleNegotiator",
    "negmas.sao.BoulwareTBNegotiator", "negmas.sao.ConcederTBNegotiator",
    "negmas.sao.LinearTBNegotiator", "negmas.sao.TimeBasedConcedingNegotiator",
    "negmas.sao.ToughNegotiator", "negmas.sao.NiceNegotiator",
    "negmas.sao.AspirationNegotiator", "negmas.sao.NaiveTitForTatNegotiator",
    "negmas.sao.SimpleTitForTatNegotiator", "negmas.sao.HybridNegotiator",
    "negmas.sao.MiCRONegotiator", "negmas.sao.FastMiCRONegotiator",
    "negmas.sao.BestOfferOrientedTBNegotiator",
    "negmas.sao.FirstOfferOrientedTBNegotiator",
    "negmas.sao.LastOfferOrientedTBNegotiator", "negmas.sao.TopFractionNegotiator",
    "negmas.sao.CABNegotiator", "negmas.sao.CANNegotiator", "negmas.sao.CARNegotiator",
    "negmas.sao.WABNegotiator", "negmas.sao.WANNegotiator", "negmas.sao.WARNegotiator",
    "negmas.sao.AdditiveFirstFollowingTBNegotiator",
    "negmas.sao.AdditiveLastOfferFollowingTBNegotiator",
    "negmas.sao.AdditiveParetoFollowingTBNegotiator",
    "negmas.sao.MultiplicativeParetoFollowingTBNegotiator",
    "negmas.sao.RandomNegotiator",
]


def competitiveness(scenario: Scenario) -> float:
    """Pearson correlation of the two ufuns over (sampled) outcomes.

    ~+1 fully cooperative (same preferences), ~-1 fully competitive (zero-sum).
    """
    u0, u1 = scenario.ufuns
    outcomes = list(scenario.outcome_space.enumerate_or_sample(max_cardinality=300))
    a = np.array([float(u0(o)) for o in outcomes])
    b = np.array([float(u1(o)) for o in outcomes])
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def load_local() -> list[tuple[str, Scenario, float]]:
    out = []
    for path in sorted(SCENARIOS_DIR.iterdir()):
        if path.is_dir():
            s = Scenario.load(path, ignore_discount=True)
            if s is not None:
                out.append((path.name, s, competitiveness(s)))
    return out


# How the generated scenarios choose their rational fraction:
#   "mixed"     -> a spread incl. degenerate low-rf (harsh stress test)
#   "realistic" -> rational_fraction = 1.0 (matches real ANL domains)
#   "competitive" -> low rational fractions only
RF_POLICIES = {
    "mixed": [1.0, 1.0, 0.6, 0.4, 0.3],
    "realistic": [1.0],
    "competitive": [0.4, 0.3, 0.2],
}


def gen_scenarios(n: int, mode: str = "mixed") -> list[tuple[str, Scenario, float]]:
    choices = RF_POLICIES[mode]
    out = []
    for k in range(n):
        rng = random.Random(1000 + k)  # per-scenario deterministic seed
        n_issues = rng.randint(1, 4)
        rf = rng.choice(choices)
        ufuns = generate_multi_issue_ufuns(
            n_issues=n_issues, n_values=(3, 8),
            ufun_names=("First", "Second"), rational_fractions=[rf, rf],
        )
        s = Scenario(outcome_space=ufuns[0].outcome_space, ufuns=ufuns)
        out.append((f"gen{k:02d}", s, competitiveness(s)))
    return out


COOP_CUT = -0.2  # corr >= cut => cooperative bucket, else competitive


def main() -> None:
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    mode = sys.argv[2] if len(sys.argv) > 2 else "mixed"
    print(f"[generated-scenario mode: {mode}]")
    scenarios = load_local() + gen_scenarios(n_gen, mode)
    n_coop = sum(1 for _, _, c in scenarios if c >= COOP_CUT)
    print(f"Arena: {len(ROSTER)} opponents x {len(scenarios)} scenarios x 2 orders "
          f"= {len(ROSTER) * len(scenarios) * 2} negotiations")
    print(f"Scenarios: {n_coop} cooperative (corr>={COOP_CUT}), "
          f"{len(scenarios) - n_coop} competitive\n")

    rows = []
    coop_all = [0.0, 0]  # sum, n
    comp_all = [0.0, 0]
    for opp in ROSTER:
        try:
            get_class(opp)
        except Exception:
            continue
        margin = myAdv = myScore = 0.0
        wins = deals = n = 0
        for _, scenario, comp in scenarios:
            coop = comp >= COOP_CUT
            for first in (True, False):
                try:
                    r = run_one(scenario, opp, first)
                except Exception:
                    continue
                m = r["mine"]["Score"] - r["theirs"]["Score"]
                margin += m; myAdv += r["mine"]["Advantage"]
                myScore += r["mine"]["Score"]
                wins += 1 if m > 0 else 0
                deals += 1 if r["agreement"] else 0; n += 1
                (coop_all if coop else comp_all)[0] += r["mine"]["Score"]
                (coop_all if coop else comp_all)[1] += 1
        if n:
            rows.append((opp.split(".")[-1], myScore / n, margin / n, wins, deals,
                         n, myAdv / n))

    # Rank by OUR score (the tournament metric), not margin.
    rows.sort(key=lambda r: r[1], reverse=True)
    print(f"{'opponent':>26}{'ourScore':>10}{'margin':>9}{'wins':>9}{'deals':>9}{'myAdv':>8}")
    print("-" * 71)
    tot_score = tot_margin = 0.0
    for name, score, mg, wins, deals, n, adv in rows:
        tot_score += score; tot_margin += mg
        flag = "" if mg > 0 else "  <-- losing margin"
        print(f"{name:>26}{score:>10.3f}{mg:>+9.3f}{wins:>6}/{n:<2}{deals:>6}/{n:<2}"
              f"{adv:>8.3f}{flag}")
    print("-" * 71)
    print(f"{'MEAN over opponents':>26}{tot_score / len(rows):>10.3f}{tot_margin / len(rows):>+9.3f}")
    print(f"\nOur mean score by scenario type:")
    print(f"  cooperative : {coop_all[0] / max(1, coop_all[1]):.3f}  (n={coop_all[1]})")
    print(f"  competitive : {comp_all[0] / max(1, comp_all[1]):.3f}  (n={comp_all[1]})")


if __name__ == "__main__":
    main()
