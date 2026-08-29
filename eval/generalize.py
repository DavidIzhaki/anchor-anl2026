"""Generalization test: does v6 (pairwise+decouple) still beat v4 against DIVERSE
negmas opponents OUTSIDE our tuning rivals set? (local eval only)

For each opponent we run BOTH configs on the same ANL scenarios/orders and report
per-opponent score, deal-rate, and concealing (con<1.0 => the opponent models us, so
the concealing term is genuinely contested -- where v6's gain should show).

    python eval/generalize.py [n_gen=24] [n_steps=300]
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "eval"))

import anchor
from labcore import scenario_specs, make_scenario, run_mech

A = anchor.AnchorNegotiator

# Diverse opponents NOT in our tuning rivals set (rivals = Anchor self-play, FirmTop/
# FirmBest, BOA/MAP/Whale, HardHeaded/AgentX, Simple). Mix of modelers + behaviourals.
WIDE = [
    # Diverse MODELLING opponents (GSmith) outside the tuning set -- con<1.0, so the
    # concealing term is contested and v6's gain should show.
    "rivals.MidModelerNeg",
    "rivals.ConcederModelerNeg",
    # Behaviourals / non-modellers (sanity: v6 must not regress; con=1.0 expected).
    "negmas.sao.NaiveTitForTatNegotiator",
    "negmas.sao.AspirationNegotiator",
    "negmas.sao.CABNegotiator",
    "negmas.sao.MiCRONegotiator",
]


def run_cfg(specs, n_steps):
    """Return per-opponent {opp: (score, deal, con, n)} for the CURRENT class config."""
    res = {}
    for opp in WIDE:
        s = d = c = 0.0
        n = 0
        for spec in specs:
            for first in (True, False):
                try:
                    r = run_mech(make_scenario(spec), opp, first, n_steps)
                except Exception:
                    continue
                s += r["score"]; d += 1 if r["deal"] else 0; c += r["con"]; n += 1
        if n:
            res[opp.split(".")[-1]] = (s / n, d / n, c / n, n)
    return res


def main():
    n_gen = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    specs = scenario_specs(n_gen=n_gen, mode="anl")

    A.MODEL_KIND = "pairwise"; A.DECOUPLE_BID = True
    v6 = run_cfg(specs, n_steps)
    A.MODEL_KIND = "stability"; A.DECOUPLE_BID = False
    v4 = run_cfg(specs, n_steps)

    print(f"Generalization vs DIVERSE opponents (ANL, n_gen={n_gen}, n_steps={n_steps})")
    print(f"{'opponent':>34}{'v4_score':>10}{'v6_score':>10}{'dScore':>8}{'v4_con':>8}{'v6_con':>8}{'deal':>6}")
    tv4 = tv6 = 0.0
    for opp in sorted(v6):
        a, b = v4.get(opp), v6[opp]
        if not a:
            continue
        tv4 += a[0]; tv6 += b[0]
        print(f"{opp:>34}{a[0]:>10.3f}{b[0]:>10.3f}{b[0]-a[0]:>+8.3f}{a[2]:>8.3f}{b[2]:>8.3f}{b[1]:>6.2f}")
    k = max(1, len(v6))
    print(f"{'MEAN':>34}{tv4/k:>10.3f}{tv6/k:>10.3f}{(tv6-tv4)/k:>+8.3f}")
    print("(con<1.0 => opponent models us => concealing contested => where v6 should gain)")


if __name__ == "__main__":
    main()
