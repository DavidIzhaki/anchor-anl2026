# Anchor — A Score-First Negotiation Agent for ANL 2026

**Anchor** is our entry to the [ANAC 2026 Automated Negotiation League (ANL)](https://anac.cs.brown.edu/anl),
a bilateral negotiation competition on the [NegMAS](https://github.com/yasserfarouk/negmas) platform.
This year's twist is **preference concealment**: an agent is scored both on the deal it
reaches *and* on how well it hides its own preferences while modelling its opponent.

> Authors: **David Izhaki** and **Itai Kahana**, Bar-Ilan University.
>
> **Anchor was selected as one of the ANL 2026 finalists.** In the qualifying live
> tournaments it ranked as high as **4th of 38** (and 9th of 41 on a harder scenario set),
> while being the fastest crash-free agent in the field (~0.2 s per negotiation vs. 4–20 s
> for the other leaders).

---

## Strategy in one paragraph

The tournament ranks agents by their **absolute mean score** = *Advantage* (how far the
deal beats your reservation value) + *Concealing* (a shared point split by how well each
side models the other). Because a no-deal scores only the ~0.5 Concealing term while a
balanced deal scores ~1.0, **closing a deal is worth far more than "winning" any single
negotiation**. Anchor therefore:

- **Bids firmly, then closes** — it opens near its ideal and holds a Boulware floor (the
  *anchor*), so a firm opponent cannot drag it down to that opponent's ideal. In a
  deadline-adaptive end-game it secures any positive-advantage deal the opponent has shown
  rather than time out. This deal-closing end-game was the single biggest improvement
  (rank 18 → 4 in the live tournaments).
- **Models hard** — a frequency + stability opponent model, weighting the opponent's
  **early** offers most (early offers reveal true preferences; late ones are concessions).
  Version 6 adds a **pairwise** signal the plain frequency model discards: values *we* keep
  offering that *they* keep rejecting are probably bad for them. The estimate is emitted
  every round (required, or the Concealing point is forfeited).
- **Decouples what it emits from what it reveals** — the scorer reads our *emitted* model,
  while the opponent's model of us is driven by our *revealed* bids. Anchor emits the
  sharper pairwise model (raising our τ) but bids on the plain frequency model (revealing
  nothing extra), lifting the Concealing share at zero Advantage cost.
- **Deceives lightly** — three deliberate concealment mechanisms were implemented and
  *measured to be counter-productive*: under a full-domain Kendall-τ metric, distorting your
  own offers costs more Advantage than the shared point returns. Anchor conceals by bidding
  firmly and modelling well, **not** by adding noise.

Across ~30 NegMAS opponents (including self-play and firm preference-modelling rivals) on
the provided and randomly generated domains, Anchor closes ~86% of negotiations and scores
positively against every opponent type. The full design and evaluation are in
[`report/report_anac.pdf`](report/report_anac.pdf); the complete experiment log
(every hypothesis, measurement, and keep/reject decision) is in
[`report/EXPERIMENTS.md`](report/EXPERIMENTS.md).

## The agent

The whole agent is a single, interpretable class in [`anchor.py`](anchor.py)
(`AnchorNegotiator`, a `negmas` `SAOCallNegotiator`), organised into the three classic
**BOA** components:

| Component | Method | What it does |
|---|---|---|
| Opponent model | `update_opponent_model` | pairwise frequency + stability estimate, early-weighted, emitted every round; separate plain model for bidding (decouple) |
| Bidding | `concealing_bidding_strategy` | floored Boulware target + Nash bid selection + deadline-adaptive end-game rescue + last-offer grab |
| Acceptance | `acceptance_strategy` | ACNext + reservation floor + capture-best + end-game secure-best |

The shipped configuration is **v6** (`MODEL_KIND="pairwise"`, `DECOUPLE_BID=True`). Many
alternatives — a Bayesian model, max-entropy IRL, lookahead bid search, MiCRO bidding, an
offline-trained (evolution-strategy) concession policy, scenario conditioning, and three
deception variants — are implemented as flag-gated constants and left **off by default**:
each was measured and found not to pay. They remain in the code for the report's ablation.
Every previously shipped version is archived in [`advisers/`](advisers/MANIFEST.md).

## Repository layout

```
anchor.py              the agent (class AnchorNegotiator) — this is the submission
requirements.txt       runtime dependency (negmas==0.15.4)
main.py                CLI: run single negotiations and tournaments with ANL 2026 scoring
make_submission.*      builds submission.zip (anchor.py + requirements.txt)
examples/              reference opponents from the official skeleton (boa, map, simple)
whale.py               a strong local benchmark opponent used by the evaluation harness
scenarios/             the 7 local negotiation domains
tests/                 unit tests (agent contract, CLI, examples)
eval/                  evaluation harness (see below)
advisers/              frozen snapshots of every shipped version (v1 … v6) + MANIFEST
report/                report_anac.{tex,pdf}  — the ANAC competition report
                       report.{tex,pdf}       — the course report
                       EXPERIMENTS.md         — full R&D log
                       concession.png         — the concession-curve figure
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

Requires Python 3.14 and `negmas>=0.15.4`. The eval harness additionally uses `numpy`
(pulled in by negmas) and, for the concession plot, `matplotlib`.

## Run the agent

```bash
# a single negotiation against a reference opponent (add --verbose --show-trace to inspect)
uv run anl2026 run --opponent examples.boa.BOANeg
uv run anl2026 run --scenario Camera --opponent examples.map.MAPNeg

# a tournament across the local scenarios with the official ANL 2026 scoring
uv run anl2026 tournament --parallel
uv run anl2026 tournament --scenario Camera --scenario Car

# unit tests
uv run pytest
```

To use Anchor as an opponent from your own NegMAS code:

```python
from anchor import AnchorNegotiator
from negmas.sao import SAOMechanism

m = SAOMechanism(outcome_space=scenario.outcome_space, n_steps=100)
m.add(AnchorNegotiator(name="anchor"), ufun=scenario.ufuns[0])
m.add(YourNegotiator(name="you"), ufun=scenario.ufuns[1])
m.run()
print(m.agreement, m.negotiators[0].private_info["opponent_ufun"])   # our emitted model
```

## Evaluation harness (`eval/`)

All scripts are local-only (not part of the submission) and are run from the repo root.
Each script's docstring documents its arguments.

| Script | Purpose |
|---|---|
| `bench.py <opponent>` | per-scenario Advantage / Concealing / Score vs one opponent, both move orders |
| `arena.py` | broad arena: ~30 NegMAS opponents × local + generated domains (cooperative → competitive) |
| `lab.py [n_steps] [n_gen] [workers] [K=V …]` | parallel experiment CLI; `K=V` overrides agent constants (e.g. `DECEPTION=True`) |
| `labcore.py` | the evaluation engine behind `lab.py` (opponent groups, `mode="anl"` live-matching domain generator) |
| `rivals.py`, `weightlearner.py` | firm / modelling sparring opponents (a proxy for the live field), incl. the frozen v4 as a self-play rival |
| `taus.py` | opponent-model accuracy: τ_me, τ_opp and the resulting Concealing share |
| `score_sweep.py`, `sweep.py`, `grid.py`, `optimize.py` | 1-D, 2-D and joint parameter sweeps |
| `generalize.py`, `landscape.py`, `tail.py` | robustness across opponent types, domain regimes, and the bad-tail diagnostic |
| `train_es.py`, `train_policy.py`, `eval_policy*.py`, `validate_policy.py` | the offline RL / evolution-strategy concession-policy experiment (archived, not shipped) |
| `plot_concession.py` | regenerates `report/concession.png` from the live agent constants |

```bash
uv run python eval/bench.py whale.WhaleNegotiator
uv run python eval/lab.py 100 18 8 MODEL_KIND=stability DECOUPLE_BID=False   # ablate v6 back to v4
uv run python eval/taus.py
```

## Submission

```bash
./make_submission.sh        # or make_submission.bat on Windows
```

produces `submission.zip` (`anchor.py` + `requirements.txt`), which is what is uploaded to
the ANL site.

## Reports

- [`report/report_anac.pdf`](report/report_anac.pdf) — the ANAC competition report (design,
  evaluation, ablations).
- [`report/report.pdf`](report/report.pdf) — the course report.
- [`report/EXPERIMENTS.md`](report/EXPERIMENTS.md) — the complete experiment log.
