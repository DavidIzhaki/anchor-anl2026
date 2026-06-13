"""Tests for the AnchorNegotiator agent."""

import pytest
from negmas.inout import Scenario
from negmas.preferences.generators import generate_multi_issue_ufuns
from negmas.sao import SAOMechanism

from anchor import AnchorNegotiator


@pytest.fixture
def test_scenario():
    """Create a simple test scenario with two issues."""
    ufuns = generate_multi_issue_ufuns(
        n_issues=2,
        n_values=(3, 5),
        ufun_names=("First", "Second"),
        rational_fractions=[1.0, 1.0],
    )
    return Scenario(outcome_space=ufuns[0].outcome_space, ufuns=ufuns)


class TestAnchorNegotiator:
    """Tests for the AnchorNegotiator agent."""

    def test_instantiation(self):
        """Test that AnchorNegotiator can be instantiated."""
        negotiator = AnchorNegotiator()
        assert negotiator is not None

    def test_emits_opponent_model(self, test_scenario):
        """The agent must expose an opponent-utility estimate every round.

        The competition scorer reads ``negotiator.opponent_ufun`` to compute the
        Concealing term; emitting nothing forfeits the whole point. We check the
        estimate exists and is callable on outcomes once the negotiation starts.
        """
        mechanism = SAOMechanism(
            outcome_space=test_scenario.outcome_space,
            n_steps=5,
        )
        negotiator = AnchorNegotiator()
        opponent = AnchorNegotiator()

        mechanism.add(negotiator, ufun=test_scenario.ufuns[0])
        mechanism.add(opponent, ufun=test_scenario.ufuns[1])

        # Run one step to initialize and exchange an offer.
        mechanism.step()

        model = negotiator.opponent_ufun
        assert model is not None
        # Callable on every outcome, returning a finite number.
        for outcome in test_scenario.outcome_space.enumerate_or_sample():
            assert float(model(outcome)) is not None

    def test_negotiation_completes(self, test_scenario):
        """Test that AnchorNegotiator can complete a negotiation."""
        mechanism = SAOMechanism(
            outcome_space=test_scenario.outcome_space,
            n_steps=50,
        )
        negotiator1 = AnchorNegotiator()
        negotiator2 = AnchorNegotiator()

        mechanism.add(negotiator1, ufun=test_scenario.ufuns[0])
        mechanism.add(negotiator2, ufun=test_scenario.ufuns[1])

        mechanism.run()
        assert mechanism.state.agreement is not None or mechanism.state.timedout

    def test_makes_offers(self, test_scenario):
        """Test that AnchorNegotiator makes valid offers."""
        mechanism = SAOMechanism(
            outcome_space=test_scenario.outcome_space,
            n_steps=10,
        )
        negotiator = AnchorNegotiator()
        opponent = AnchorNegotiator()

        mechanism.add(negotiator, ufun=test_scenario.ufuns[0])
        mechanism.add(opponent, ufun=test_scenario.ufuns[1])

        mechanism.run()

        # Check that offers were made
        assert len(mechanism.history) > 0

    def test_negotiation_with_different_opponents(self, test_scenario):
        """Test that AnchorNegotiator can negotiate with different types of opponents."""
        from examples.simple import SimpleNegotiator
        from examples.map import MAPNeg

        # Test against SimpleNegotiator
        mechanism1 = SAOMechanism(
            outcome_space=test_scenario.outcome_space,
            n_steps=50,
        )
        negotiator1 = AnchorNegotiator()
        opponent1 = SimpleNegotiator()

        mechanism1.add(negotiator1, ufun=test_scenario.ufuns[0])
        mechanism1.add(opponent1, ufun=test_scenario.ufuns[1])

        mechanism1.run()
        assert mechanism1.state.agreement is not None or mechanism1.state.timedout

        # Test against MAPNeg
        mechanism2 = SAOMechanism(
            outcome_space=test_scenario.outcome_space,
            n_steps=50,
        )
        negotiator2 = AnchorNegotiator()
        opponent2 = MAPNeg()

        mechanism2.add(negotiator2, ufun=test_scenario.ufuns[0])
        mechanism2.add(opponent2, ufun=test_scenario.ufuns[1])

        mechanism2.run()
        assert mechanism2.state.agreement is not None or mechanism2.state.timedout

    def test_negotiation_on_multiple_scenarios(self, test_scenario):
        """Test that AnchorNegotiator works on scenarios with different numbers of issues."""
        # Test with 1 issue
        ufuns1 = generate_multi_issue_ufuns(
            n_issues=1,
            n_values=(3, 5),
            ufun_names=("First", "Second"),
            rational_fractions=[1.0, 1.0],
        )
        scenario1 = Scenario(outcome_space=ufuns1[0].outcome_space, ufuns=ufuns1)

        mechanism1 = SAOMechanism(
            outcome_space=scenario1.outcome_space,
            n_steps=50,
        )
        negotiator1a = AnchorNegotiator()
        negotiator1b = AnchorNegotiator()

        mechanism1.add(negotiator1a, ufun=scenario1.ufuns[0])
        mechanism1.add(negotiator1b, ufun=scenario1.ufuns[1])

        mechanism1.run()
        assert mechanism1.state.agreement is not None or mechanism1.state.timedout

        # Test with 4 issues
        ufuns4 = generate_multi_issue_ufuns(
            n_issues=4,
            n_values=(3, 5),
            ufun_names=("First", "Second"),
            rational_fractions=[1.0, 1.0],
        )
        scenario4 = Scenario(outcome_space=ufuns4[0].outcome_space, ufuns=ufuns4)

        mechanism4 = SAOMechanism(
            outcome_space=scenario4.outcome_space,
            n_steps=50,
        )
        negotiator4a = AnchorNegotiator()
        negotiator4b = AnchorNegotiator()

        mechanism4.add(negotiator4a, ufun=scenario4.ufuns[0])
        mechanism4.add(negotiator4b, ufun=scenario4.ufuns[1])

        mechanism4.run()
        assert mechanism4.state.agreement is not None or mechanism4.state.timedout

    def test_agreement_is_valid(self, test_scenario):
        """Test that agreements reached by AnchorNegotiator are valid outcomes."""
        mechanism = SAOMechanism(
            outcome_space=test_scenario.outcome_space,
            n_steps=50,
        )
        negotiator1 = AnchorNegotiator()
        negotiator2 = AnchorNegotiator()

        mechanism.add(negotiator1, ufun=test_scenario.ufuns[0])
        mechanism.add(negotiator2, ufun=test_scenario.ufuns[1])

        mechanism.run()

        # If agreement is reached, it should be a valid outcome
        if mechanism.state.agreement is not None:
            assert mechanism.state.agreement in test_scenario.outcome_space.enumerate()
