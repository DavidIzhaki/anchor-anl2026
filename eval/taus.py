"""Direct opponent-model accuracy harness (local eval only).

Measures, against the model-capable opponents (the only ones where the Concealing
term is actually contested), the mean:
  * tau_me  : how well WE model the opponent  (Kendall tau of our estimate vs their
              true ufun) -- the FREE lever we want to raise.
  * tau_opp : how well the opponent models US.
  * share   : our resulting Concealing share = acc_me / (acc_me + acc_opp), where
              acc = (1 + tau) / 2. 0.5 == even; >0.5 == we win the split.

This is the metric for the opponent-model work (concession order, recency, etc.):
change the model, re-run this, keep what raises tau_me / share.

    uv run python eval/taus.py            # default model-capable roster
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from negmas.helpers import get_class, instantiate
from negmas.inout import Scenario
from negmas.preferences import compare_ufuns
from negmas.preferences.generators import generate_multi_issue_ufuns
from negmas.sao import SAOMechanism

from bench import SCENARIOS_DIR, MY_NEGOTIATOR

# Opponents that emit an opponent model (so tau_opp is meaningful and the
# Concealing split is genuinely contested).
ROSTER = [
    "examples.map.MAPNeg",
    "examples.boa.BOANeg",
    "examples.map.MAPNeg",
    "examples.simple.SimpleNegotiator",
]


def scenarios(n_gen: int = 12) -> list[Scenario]:
    out = []
    for path in sorted(SCENARIOS_DIR.iterdir()):
        if path.is_dir():
            s = Scenario.load(path, ignore_discount=True)
            if s is not None:
                out.append(s)
    for k in range(n_gen):
        rng = random.Random(2000 + k)
        ufuns = generate_multi_issue_ufuns(
            n_issues=rng.randint(1, 4), n_values=(3, 8),
            ufun_names=("A", "B"), rational_fractions=[1.0, 1.0],
        )
        out.append(Scenario(outcome_space=ufuns[0].outcome_space, ufuns=ufuns))
    return out


def run(scenario: Scenario, opp: str, first: bool):
    m = SAOMechanism(n_steps=100, outcome_space=scenario.outcome_space)
    me = instantiate(MY_NEGOTIATOR, ufun=scenario.ufuns[0 if first else 1])
    op = instantiate(opp, ufun=scenario.ufuns[1 if first else 0])
    (m.add(me), m.add(op)) if first else (m.add(op), m.add(me))
    m.run()
    tau_me = compare_ufuns(op.ufun, me.opponent_ufun, method="kendall")
    op_model = getattr(op, "opponent_ufun", None)
    tau_opp = compare_ufuns(me.ufun, op_model, method="kendall") if op_model else None
    return tau_me, tau_opp


def main() -> None:
    scens = scenarios()
    print(f"Model accuracy over {len(scens)} scenarios x 2 orders\n")
    print(f"{'opponent':>20}{'tau_me':>9}{'tau_opp':>9}{'share':>8}")
    print("-" * 46)
    gtm = gto = gshare = gn = 0.0
    for opp in ROSTER:
        try:
            get_class(opp)
        except Exception:
            continue
        tm = to = sh = n = 0.0
        nshare = 0
        for s in scens:
            for first in (True, False):
                try:
                    a, b = run(s, opp, first)
                except Exception:
                    continue
                tm += a; n += 1
                if b is not None:
                    to += b
                    acc_me, acc_opp = (1 + a) / 2, (1 + b) / 2
                    if acc_me + acc_opp > 0:
                        sh += acc_me / (acc_me + acc_opp); nshare += 1
        if n:
            mt, mo = tm / n, (to / nshare if nshare else float("nan"))
            ms = sh / nshare if nshare else float("nan")
            print(f"{opp.split('.')[-1]:>20}{mt:>9.3f}{mo:>9.3f}{ms:>8.3f}")
            gtm += tm; gto += to; gshare += sh; gn += n
    print("-" * 46)
    print(f"{'MEAN':>20}{gtm / gn:>9.3f}{gto / gn:>9.3f}{gshare / gn:>8.3f}")


if __name__ == "__main__":
    main()
