"""Parallel evaluation engine for Anchor (local eval only; importable).

Why a new engine: the original score_sweep is single-process and groups
opponents in a way that lets non-modelling negmas agents (Tough, conceders)
inflate our score on no-deals (they don't model us, so concealing ~1.0 even with
no deal). That masking is what mistuned the original agent. Here the headline
group is `rivals` = opponents that DO emit a model of us, so a no-deal scores
~0.5 -- the honest proxy for the 38-agent live field.

Design (senior-eng notes):
  * Workers rebuild scenarios from a seed/spec (no pickling of negmas objects).
  * Experiment flags are applied per worker via the Pool initializer, because
    Windows 'spawn' workers do not inherit runtime setattr from the parent.
  * AnchorRival (self-play) is PINNED to a baseline config (see rivals.py), so an
    experiment measures help against a fixed rival, not a moving target.
  * Reports per-group mean Score / deal-rate / Advantage / Concealing and a
    tournament-mix-weighted OVERALL.
"""

from __future__ import annotations

import random
import zlib
from pathlib import Path

import numpy as np


def _seed_for(spec, opp, first) -> int:
    """Stable (process-independent) per-task seed. Python's hash() is salted per
    process, so we use crc32 to make serial and parallel runs identical."""
    return zlib.crc32(repr((spec, opp, first)).encode()) & 0x7FFFFFFF

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

from negmas.helpers import get_class, instantiate
from negmas.inout import Scenario
from negmas.sao import SAOMechanism
from negmas.preferences.generators import generate_multi_issue_ufuns

from main import calc_scores, MY_NEGOTIATOR

SCENARIOS_DIR = ROOT / "scenarios"

# ---- Opponent groups -------------------------------------------------------
# `rivals` = the honest live-field proxy: every member emits an opponent model
# of us (self-play + GSmith / weight-learning / inverse modellers), so on a
# no-deal we only get the ~0.5 Concealing split (as in the real tournament).
GROUPS = {
    "rivals": [
        "rivals.AnchorRival", "rivals.ShippedV4Rival",
        "rivals.FirmTopNeg", "rivals.FirmBestNeg",
        "examples.boa.BOANeg", "examples.map.MAPNeg", "whale.WhaleNegotiator",
        "weightlearner.HardHeadedNeg", "weightlearner.AgentXNeg",
        "examples.simple.SimpleNegotiator",
    ],
    "firm": [
        "negmas.sao.ToughNegotiator", "negmas.sao.MiCRONegotiator",
        "negmas.sao.FastMiCRONegotiator", "negmas.sao.TopFractionNegotiator",
        "negmas.sao.HybridNegotiator", "negmas.sao.WARNegotiator",
    ],
    "conceders": [
        "negmas.sao.BoulwareTBNegotiator", "negmas.sao.ConcederTBNegotiator",
        "negmas.sao.LinearTBNegotiator", "negmas.sao.NiceNegotiator",
        "negmas.sao.TimeBasedConcedingNegotiator",
    ],
}
# The live field is overwhelmingly modelling rivals; conceders/non-modellers are
# rare in it, so they get low weight (their high no-deal scores must not dominate).
WEIGHTS = {"rivals": 0.6, "firm": 0.25, "conceders": 0.15}


def make_scenario(spec):
    """Rebuild a scenario from a picklable spec.

    spec is ("local", name) or ("gen", seed, n_issues, n_values, rf).
    """
    kind = spec[0]
    if kind == "local":
        return Scenario.load(SCENARIOS_DIR / spec[1], ignore_discount=True)
    if kind == "anl_gen":
        # Live-matching: ~1000 outcomes, Pareto mix (linear/curve/zero-sum), and
        # per-agent reservations SAMPLED from [0,1] capped at Nash (guarantee_
        # rational) -- the actual ANL distribution (mean res ~0.32), not our old
        # full-rational low-reservation domains (mean ~0.11).
        _, seed, n_issues, sizes = spec
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        ufuns = generate_multi_issue_ufuns(
            n_issues=n_issues, sizes=tuple(sizes),
            pareto_generators=("piecewise_linear", "curve", "zero_sum"),
            reserved_values=(0.0, 1.0), guarantee_rational=True,
            ufun_names=("First", "Second"),
        )
        return Scenario(outcome_space=ufuns[0].outcome_space, ufuns=ufuns)
    _, seed, n_issues, n_values, rf = spec
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    ufuns = generate_multi_issue_ufuns(
        n_issues=n_issues, n_values=n_values,
        ufun_names=("First", "Second"), rational_fractions=[rf, rf],
    )
    return Scenario(outcome_space=ufuns[0].outcome_space, ufuns=ufuns)


def _anl_sizes(rng, n_issues):
    """Issue sizes whose product is ~900-1100 (ANL domains are ~1000 outcomes)."""
    targets = {2: [(32, 31), (40, 25), (50, 20)],
               3: [(10, 10, 10), (16, 8, 8), (20, 10, 5)],
               4: [(6, 6, 6, 5), (5, 5, 5, 8), (10, 10, 5, 2)],
               5: [(4, 4, 4, 4, 4), (5, 5, 4, 2, 5)]}
    return rng.choice(targets[n_issues])


# Generated-scenario rational-fraction policy by mode. Real ANL domains are
# (near-)fully rational, so `realistic` matches the live field; `mixed` adds some
# competitive structure for robustness; `hard` is a harsh stress test.
RF_MODES = {
    "realistic": [1.0],
    "mixed": [1.0, 1.0, 1.0, 0.8, 0.6, 0.4],
    "hard": [0.4, 0.3, 0.2],
}


def scenario_specs(n_gen=18, seed_base=7000, mode="realistic", include_local=True):
    """Local 7 + a generated batch. `mode` sets the rational-fraction policy."""
    specs = []
    if include_local:
        for path in sorted(SCENARIOS_DIR.iterdir()):
            if path.is_dir():
                specs.append(("local", path.name))
    if mode == "anl":
        # The live-matching distribution (see make_scenario "anl_gen"). Pure
        # generated set -- exclude the hand-authored local domains, which the live
        # tournament does not use.
        specs = []
        for k in range(n_gen):
            rng = random.Random(seed_base + k)
            n_issues = rng.choice([2, 3, 3, 4, 4, 5])
            sizes = _anl_sizes(rng, n_issues)
            specs.append(("anl_gen", seed_base + k, n_issues, list(sizes)))
        return specs
    rfs = RF_MODES[mode]
    for k in range(n_gen):
        rng = random.Random(seed_base + k)
        n_issues = rng.randint(1, 5)
        n_values = (3, rng.randint(5, 10))
        rf = rng.choice(rfs)
        specs.append(("gen", seed_base + k, n_issues, n_values, rf))
    return specs


# ---- worker state (set by Pool initializer) --------------------------------
_NSTEPS = 100
_OVERRIDES: dict = {}


def init_worker(overrides: dict, n_steps: int):
    global _NSTEPS, _OVERRIDES
    _NSTEPS = n_steps
    _OVERRIDES = dict(overrides)
    # Apply experiment flags to OUR agent only (AnchorRival pins its own).
    from anchor import AnchorNegotiator
    for k, v in overrides.items():
        setattr(AnchorNegotiator, k, v)


def run_mech(scen, opp: str, first: bool, n_steps: int) -> dict:
    m = SAOMechanism(n_steps=n_steps, outcome_space=scen.outcome_space)
    mine = MY_NEGOTIATOR
    opp_short = opp.split(".")[-1]
    if first:
        m.add(instantiate(mine, ufun=scen.ufuns[0], id="Mine", name="Mine"))
        m.add(instantiate(opp, ufun=scen.ufuns[1], id=opp_short, name=opp_short))
    else:
        m.add(instantiate(opp, ufun=scen.ufuns[0], id=opp_short, name=opp_short))
        m.add(instantiate(mine, ufun=scen.ufuns[1], id="Mine", name="Mine"))
    m.run()
    scores = calc_scores(m)
    mine_key = "AnchorNegotiator"
    them_key = [k for k in scores if k != mine_key][0]
    me, them = scores[mine_key], scores[them_key]
    return {
        "deal": m.agreement is not None,
        "adv": me["Advantage"], "con": me["Concealing"], "score": me["Score"],
        "opp_score": them["Score"],
    }


def task(args) -> tuple:
    group, opp, spec, first = args
    try:
        scen = make_scenario(spec)
        if scen is None:
            return (group, opp.split(".")[-1], None)
        # Reseed AFTER scenario construction so the negotiation itself
        # (outcome sampling in our agent and the opponent) is deterministic and
        # identical in serial and parallel runs -- the global RNG state must not
        # leak between tasks in a reused worker.
        s = _seed_for(spec, opp, first)
        random.seed(s)
        np.random.seed(s)
        r = run_mech(scen, opp, first, _NSTEPS)
        return (group, opp.split(".")[-1], r)
    except Exception as e:  # keep the sweep alive; record the failure
        return (group, opp.split(".")[-1], {"error": repr(e)})


def build_tasks(specs):
    tasks = []
    for group, opps in GROUPS.items():
        for opp in opps:
            try:
                get_class(opp)
            except Exception:
                continue
            for spec in specs:
                for first in (True, False):
                    tasks.append((group, opp, spec, first))
    return tasks
