"""Map our performance across the full domain space (local eval only).

Sweeps (n_issues x n_values x rational_fraction) and reports our mean score /
deal / advantage / concealing per cell, against a diverse opponent subset, to
find which domain REGIMES we generalise poorly on. Single serial process (no
orphan workers). Override AnchorNegotiator attrs via K=V to compare configs.

    python eval/landscape.py [reps=3] [K=V ...]
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import anchor
from labcore import run_mech
from negmas.preferences.generators import generate_multi_issue_ufuns
from negmas.inout import Scenario

# Diverse opponent subset (fast but covers types: conceder, firm, modeller,
# self-play, inverse-modeller, strong baseline).
OPPS = ["negmas.sao.BoulwareTBNegotiator", "negmas.sao.ToughNegotiator",
        "negmas.sao.MiCRONegotiator", "examples.boa.BOANeg", "whale.WhaleNegotiator",
        "rivals.AnchorRival", "examples.simple.SimpleNegotiator",
        "negmas.sao.ConcederTBNegotiator"]

ISSUES = [1, 2, 3, 5]
VALUES = [3, 6, 9]
RFS = [1.0, 0.7, 0.4]


def make(ni, nv, rf, seed):
    random.seed(seed); np.random.seed(seed % (2**31))
    uf = generate_multi_issue_ufuns(n_issues=ni, n_values=(nv, nv),
                                    ufun_names=("A", "B"), rational_fractions=[rf, rf])
    return Scenario(outcome_space=uf[0].outcome_space, ufuns=uf)


def main():
    args = sys.argv[1:]
    reps = 3
    for a in args:
        if "=" in a:
            k, v = a.split("=", 1)
            val = (v == "True") if v in ("True", "False") else (
                float(v) if v.replace(".", "").replace("-", "").isdigit() else v)
            setattr(anchor.AnchorNegotiator, k, val)
        elif a.isdigit():
            reps = int(a)
    print(f"overrides applied: {[a for a in args if '=' in a]}")
    print(f"{'issues':>6}{'values':>7}{'rf':>5}{'score':>8}{'deal':>6}{'adv':>7}{'con':>7}")
    worst = []
    for rf in RFS:
        for ni in ISSUES:
            for nv in VALUES:
                sc = dl = adv = con = n = 0.0
                for r in range(reps):
                    s = make(ni, nv, rf, 4000 + r * 13 + ni * 7 + nv)
                    for opp in OPPS:
                        for first in (True, False):
                            try:
                                res = run_mech(s, opp, first, 100)
                            except Exception:
                                continue
                            sc += res["score"]; dl += 1 if res["deal"] else 0
                            adv += res["adv"]; con += res["con"]; n += 1
                if n:
                    m = sc / n
                    print(f"{ni:>6}{nv:>7}{rf:>5.1f}{m:>8.3f}{dl/n:>6.2f}{adv/n:>7.3f}{con/n:>7.3f}")
                    worst.append((m, ni, nv, rf, dl / n, adv / n, con / n))
    worst.sort()
    print("\nWEAKEST regimes (score | issues,values,rf | deal adv con):")
    for m, ni, nv, rf, d, a, c in worst[:6]:
        print(f"  {m:.3f}  ({ni}i,{nv}v,rf{rf})  deal={d:.2f} adv={a:.3f} con={c:.3f}")
    print(f"\nOVERALL mean across regimes: {np.mean([w[0] for w in worst]):.3f}")


if __name__ == "__main__":
    main()
