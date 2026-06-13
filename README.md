# Anchor — A Score-First Negotiation Agent for ANL 2026

**Anchor** is our entry to the [ANAC 2026 Automated Negotiation League (ANL)](https://anac.cs.brown.edu/anl),
a bilateral negotiation competition on the [NegMAS](https://github.com/yasserfarouk/negmas) platform.
This year's twist is **preference concealment**: an agent is scored both on the deal it
reaches *and* on how well it hides its own preferences while modelling its opponent.

> Author: **David Izhaki**, Bar-Ilan University.

---

## Strategy in one paragraph

The tournament ranks agents by their **absolute mean score** = *Advantage* (how far the
deal beats your reservation value) + *Concealing* (a shared point split by how well each
side models the other). Because a no-deal scores only the ~0.5 Concealing term while a
balanced deal scores ~1.0, **closing a deal is worth far more than "winning" any single
negotiation**. Anchor therefore:

- **Bids firmly** — it opens near its ideal and holds a Boulware floor (the *anchor*), so a
  firm opponent cannot drag it down to that opponent's ideal. It only concedes in a short
  end-game, to *close a deal rather than time out*.
- **Models hard** — a frequency opponent model that weights the opponent's **early** offers
  most, since early offers reveal true preferences while late ones are concessions. The
  estimate is emitted every round (required, or the Concealing point is forfeited).
- **Deceives lightly** — we implemented three deliberate concealment mechanisms and
  *measured every one to be counter-productive*: under a full-domain Kendall-tau metric,
  distorting your own offers costs more Advantage than the shared point returns. Anchor
  conceals by bidding firmly and modelling well, **not** by adding noise.

Across ~30 NegMAS opponents on the provided and randomly generated domains, Anchor scores
positively against every opponent type (mean ≈ 1.5). The full design and evaluation are in
[`report/report_anac.pdf`](report/report_anac.pdf).

## The agent

The whole agent is a single, interpretable class in [`anchor.py`](anchor.py)
(`AnchorNegotiator`, a `negmas` `SAOCallNegotiator`), organised into the three classic
**BOA** components:

| Component | Method | What it does |
|---|---|---|
| Opponent model | `update_opponent_model` | frequency + stability estimate, early-time weighted, emitted every round |
| Bidding | `concealing_bidding_strategy` | floored Boulware target + opponent-aware bid selection + end-game rescue |
| Acceptance | `acceptance_strategy` | ACNext + reservation floor + capture-best + secure-fair-best |

Several design alternatives (a Bayesian model, three deception variants, a runtime
opponent-type switch) are implemented as flag-gated constants and left **off by default** —
each was measured and found not to pay. They remain in the code for the report's ablation.

## Repository layout

```
anchor.py          the agent (class AnchorNegotiator)
requirements.txt   runtime dependency (negmas)
main.py            CLI: run single negotiations and tournaments
examples/          reference opponents (boa, map, simple)
scenarios/         local negotiation domains
tests/             unit tests
eval/              evaluation harness (per-opponent benches, arena, tau/score sweeps)
report/            the academic report (PDF + LaTeX) and the concession-curve figure
```

## Setup

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

or with pip:

```bash
pip install -e .
```

Requires Python 3.14 and `negmas>=0.15.4`.

## Run it

```bash
# a single negotiation against a reference opponent
uv run anl2026 run --opponent examples.boa.BOANeg

# a tournament across the local scenarios
uv run anl2026 tournament --parallel

# unit tests
uv run pytest

# evaluation harness
uv run python eval/bench.py examples.boa.BOANeg   # per-scenario table vs one opponent
uv run python eval/arena.py                        # broad arena across many opponents
uv run python eval/taus.py                          # opponent-model accuracy (tau) report
```

## Report

The 2–6 page academic report describing the design and evaluation:
[`report/report_anac.pdf`](report/report_anac.pdf).
