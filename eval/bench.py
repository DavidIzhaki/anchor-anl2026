"""Local evaluation harness (NOT part of the submission zip).

Runs our agent against a chosen opponent across every local scenario and both
move orders, and reports per-scenario and mean Advantage / Concealing / Score
for both sides, plus the win margin. Used to iterate on beating a benchmark
opponent (e.g. examples.map.MAPNeg).

Usage:
    uv run python eval/bench.py                      # default: vs benchmarkNegotiator
    uv run python eval/bench.py examples.boa.BOANeg  # vs another opponent
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import the project modules (main.py lives one level up).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from negmas.helpers import get_class, instantiate
from negmas.inout import Scenario
from negmas.sao import SAOMechanism

from main import calc_scores, MY_NEGOTIATOR

SCENARIOS_DIR = ROOT / "scenarios"
N_STEPS = 100


def run_one(scenario: Scenario, opponent: str, negotiator_first: bool) -> dict:
    neg, opp = MY_NEGOTIATOR, opponent
    neg_name, opp_name = "Mine", opp.split(".")[-1]
    m = SAOMechanism(n_steps=N_STEPS, outcome_space=scenario.outcome_space)
    if negotiator_first:
        m.add(instantiate(neg, ufun=scenario.ufuns[0], id=neg_name, name=neg_name))
        m.add(instantiate(opp, ufun=scenario.ufuns[1], id=opp_name, name=opp_name))
    else:
        m.add(instantiate(opp, ufun=scenario.ufuns[0], id=opp_name, name=opp_name))
        m.add(instantiate(neg, ufun=scenario.ufuns[1], id=neg_name, name=neg_name))
    m.run()
    scores = calc_scores(m)
    mine = scores[[k for k in scores if k.endswith("AnchorNegotiator") or k == "AnchorNegotiator"][0]]
    theirs = scores[[k for k in scores if k != "AnchorNegotiator" and not k.endswith("AnchorNegotiator")][0]]
    return {
        "agreement": m.agreement is not None,
        "mine": mine,
        "theirs": theirs,
    }


def evaluate(opponent: str) -> tuple[list, dict]:
    """Run all scenarios x both orders; return (rows, aggregate stats)."""
    rows = []
    for path in sorted(SCENARIOS_DIR.iterdir()):
        if not path.is_dir():
            continue
        scenario = Scenario.load(path, ignore_discount=True)
        if scenario is None:
            continue
        for first in (True, False):
            r = run_one(scenario, opponent, first)
            rows.append((path.name, "1st" if first else "2nd", r))

    agg = dict(myAdv=0.0, myCon=0.0, myScore=0.0, opAdv=0.0, opScore=0.0,
               margin=0.0, wins=0, deals=0, n=0)
    for _, _, r in rows:
        mine, theirs = r["mine"], r["theirs"]
        margin = mine["Score"] - theirs["Score"]
        agg["myAdv"] += mine["Advantage"]; agg["myCon"] += mine["Concealing"]
        agg["myScore"] += mine["Score"]; agg["opAdv"] += theirs["Advantage"]
        agg["opScore"] += theirs["Score"]; agg["margin"] += margin
        agg["wins"] += 1 if margin > 0 else 0
        agg["deals"] += 1 if r["agreement"] else 0
        agg["n"] += 1
    return rows, agg


def main() -> None:
    opponent = sys.argv[1] if len(sys.argv) > 1 else "examples.map.MAPNeg"
    get_class(opponent)  # fail fast if not importable

    rows, sums = evaluate(opponent)

    print(f"\n=== Mine vs {opponent}  (n_steps={N_STEPS}) ===")
    print(f"{'scenario':<16}{'ord':<5}{'deal':<6}{'myAdv':>8}{'myCon':>8}{'myScore':>9}"
          f"{'opAdv':>8}{'opScore':>9}{'margin':>8}")
    for name, order, r in rows:
        mine, theirs = r["mine"], r["theirs"]
        margin = mine["Score"] - theirs["Score"]
        print(f"{name:<16}{order:<5}{('Y' if r['agreement'] else '-'):<6}"
              f"{mine['Advantage']:>8.3f}{mine['Concealing']:>8.3f}{mine['Score']:>9.3f}"
              f"{theirs['Advantage']:>8.3f}{theirs['Score']:>9.3f}{margin:>8.3f}")

    n = sums["n"]
    print("-" * 79)
    print(f"{'MEAN':<16}{'':<5}{sums['deals']}/{n:<4}"
          f"{sums['myAdv']/n:>8.3f}{sums['myCon']/n:>8.3f}{sums['myScore']/n:>9.3f}"
          f"{sums['opAdv']/n:>8.3f}{sums['opScore']/n:>9.3f}{sums['margin']/n:>8.3f}")
    print(f"Wins (my score > theirs): {sums['wins']}/{n}   "
          f"Deals: {sums['deals']}/{n}   Mean margin: {sums['margin']/n:+.3f}")


if __name__ == "__main__":
    main()
