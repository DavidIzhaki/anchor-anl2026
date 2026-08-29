"""Firm rival sparring opponents (local eval only; NOT in the submission zip).

The live ANL field is ~38 firm, sophisticated student agents, not the conceder-
heavy negmas built-in roster our parameters were originally tuned on. These
classes are a proxy for that field, used to tune deal-closing behaviour:

  * AnchorRival -- our own agent on the other side of the table (self-play). The
    single best proxy for "firm, models well, holds high". It is a trivial
    subclass so that calc_scores (which keys results by class name) does not
    collapse both sides into one entry. Because the tunable knobs are class
    attributes on AnchorNegotiator, a sweep that sets them on the parent is
    inherited here too -- so both sides negotiate at the candidate parameters.
  * FirmTopNeg -- holds near its ideal (top 5% offers) and only accepts the
    top 10%, with a GSmith frequency model. A firm-but-not-degenerate rival.
  * FirmBestNeg -- the hardest case: offers and accepts ONLY its single best
    outcome. A pure hardliner with a model.
"""

import importlib.util
from pathlib import Path

from negmas.sao.negotiators.modular import BOANegotiator
from negmas.sao.components.offering import OfferTop, OfferBest
from negmas.sao.components.acceptance import AcceptTop, AcceptBest
from negmas.gb.components.genius.models import GSmithFrequencyModel

from anchor import AnchorNegotiator


def _load_frozen(version: str):
    """Load a shipped adviser snapshot as its OWN module, so self-play against a
    prior version exercises that version's actual code (model + bidding), not the
    live class with a few attributes pinned. This makes every iteration spar
    against our true lineage -- the honest "did we beat what we shipped?" test."""
    path = Path(__file__).resolve().parent.parent / "advisers" / f"{version}.py"
    spec = importlib.util.spec_from_file_location(f"frozen_{version}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.AnchorNegotiator


class AnchorRival(AnchorNegotiator):
    """Self-play opponent: our agent at a PINNED reference configuration.

    Pinning the knobs (rather than inheriting whatever the experiment currently
    sets on AnchorNegotiator) makes this a STABLE firm-modelling rival, so an
    experiment measures whether our change helps against opponents that are NOT
    also running the change -- the honest question for the live field.
    Reference = the Round-A operating point.
    """

    FAIR_FLOOR_FRACTION = 0.72
    RESCUE_TIME = 0.95
    RESCUE_FLOOR_FRACTION = 0.10
    SECURE_BEST_FAIR = False


class ShippedV4Rival(_load_frozen("anchor_v4_generalized")):
    """Self-play against the FROZEN shipped v4 (its real code, not the live class).
    The bar to beat: any change must not regress against what we already shipped."""


class FirmTopNeg(BOANegotiator):
    """Offers top 5%, accepts top 10%; GSmith model. Firm rival."""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(
            acceptance=AcceptTop(fraction=0.1),
            offering=OfferTop(fraction=0.05),
            model=GSmithFrequencyModel(),
        )
        super().__init__(*args, **kwargs)


class AcceptHighFirmNeg(BOANegotiator):
    """Holds near its ideal (top 2%) and accepts only its top 5%; GSmith model.
    A harsh "accept-only-high" rival used to check our end-game does not simply
    cave to a lopsided best when holding our offer would have closed higher."""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(
            acceptance=AcceptTop(fraction=0.05),
            offering=OfferTop(fraction=0.02),
            model=GSmithFrequencyModel(),
        )
        super().__init__(*args, **kwargs)


class FirmBestNeg(BOANegotiator):
    """Offers and accepts only its single best outcome; GSmith model. Hardliner."""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(
            acceptance=AcceptBest(),
            offering=OfferBest(),
            model=GSmithFrequencyModel(),
        )
        super().__init__(*args, **kwargs)


# --- N8 generalization: diverse MODELING opponents (GSmith) with configs NOT in the
# tuning set, to confirm v6's concealing gain holds across modeler types. ---
class MidModelerNeg(BOANegotiator):
    """Mid-firm modeller: offers top 15%, accepts top 25%; GSmith model."""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(acceptance=AcceptTop(fraction=0.25),
                       offering=OfferTop(fraction=0.15),
                       model=GSmithFrequencyModel())
        super().__init__(*args, **kwargs)


class ConcederModelerNeg(BOANegotiator):
    """Conceding modeller: offers top 35%, accepts top 50%; GSmith model."""

    def __init__(self, *args, **kwargs):
        kwargs |= dict(acceptance=AcceptTop(fraction=0.50),
                       offering=OfferTop(fraction=0.35),
                       model=GSmithFrequencyModel())
        super().__init__(*args, **kwargs)
