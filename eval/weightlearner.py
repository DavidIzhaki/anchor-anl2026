"""Weight-learning sparring opponents (local eval only).

These use NegMAS opponent models that LEARN issue weights from which issues we
keep unchanged between consecutive offers (HardHeaded / AgentX). They are the
opponents the decoy-freeze concealment is designed to fool -- the stock GSmith
model that BOANeg/MAPNeg use ignores issue weights, so it cannot be fooled this
way. Used only to measure whether decoy-freeze lowers their tau of us.
"""

from negmas.sao.negotiators.modular import BOANegotiator
from negmas.sao.components.offering import TimeBasedOfferingPolicy
from negmas.sao.components.acceptance import ACNext
from negmas.gb.components.genius.models import (
    GHardHeadedFrequencyModel,
    GAgentXFrequencyModel,
)


class HardHeadedNeg(BOANegotiator):
    def __init__(self, *args, **kwargs):
        off = TimeBasedOfferingPolicy()
        kwargs |= dict(acceptance=ACNext(off), offering=off,
                       model=GHardHeadedFrequencyModel())
        super().__init__(*args, **kwargs)


class AgentXNeg(BOANegotiator):
    def __init__(self, *args, **kwargs):
        off = TimeBasedOfferingPolicy()
        kwargs |= dict(acceptance=ACNext(off), offering=off,
                       model=GAgentXFrequencyModel())
        super().__init__(*args, **kwargs)
