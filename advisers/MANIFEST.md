# Anchor agent — version archive ("advisers")

Snapshots of every shipped/active version of `anchor.py`, so any version can be
recovered. NOT part of the submission (excluded by make_submission). The **active**
agent is always the top-level `anchor.py`; the files here are frozen backups.

**Convention:** before each change to `anchor.py`, copy the outgoing version here as
`anchor_vN_<label>.py` and add a row below. To recover a version, copy it back over
`anchor.py` (e.g. `cp advisers/anchor_v1_rank4.py anchor.py`).

| Version | File | Provenance / result | Key params (delta from active) |
|---|---|---|---|
| v1 | `anchor_v1_rank4.py` | **Live tournament #19115 → RANK 4 / 38** (Anchor 1.86.54, mean 1.285). The deal-closing fix that took us 18→4. | `RESCUE_FLOOR_FRACTION=0.10`, `SECURE_ACCEPT_FLOOR=0.0` (the later refinements not yet present) |
| v2 | `anchor_v2_bottomtail.py` | Post-rank-4 bottom-tail refinement (this session). Lab: realistic 1.10, mixed 1.02, deal 0.86; still #1 in self-contained tournaments. | `RESCUE_FLOOR_FRACTION=0.20`, `SECURE_ACCEPT_FLOOR=0.05` (refuse worthless deals + hold offers higher) |
| **active** | `../anchor.py` | = v2 (current). | — |

## Notes
- v1 is a faithful *behavioral* reconstruction of the rank-4 agent: identical to v2
  except the two params above (the only behavior-changing edits made after rank-4).
  The extra ablation flags in v2 (`SECURE_ANY`, `NEVER_UNDERSELL`, `ADAPTIVE_SECURE`,
  `RESCUE_SELECT` modes, `MODEL_KIND="entropy"`) are all default-OFF and do not
  change behavior, so v1 and v2 differ only by the two params.
- Shipped config (v2 active): `SECURE_BEST_FAIR=False, SECURE_BEST_TIME=0.97,
  FAIR_FLOOR_FRACTION=0.85, RESCUE_TIME=0.95, RESCUE_FLOOR_FRACTION=0.20,
  SECURE_ACCEPT_FLOOR=0.05, OPENING_BOOST=3.0, LAST_OFFER_GRAB=True,
  MODEL_KIND="stability", MODEL_WEIGHTING="early", SELECTION_MODE="nash",
  CONCESSION_EXPONENT=0.10, CAPTURE_FRACTION=0.85`.
- Full experiment history: `../report/EXPERIMENTS.md`.

## v3 — RL residual-MLP policy (offline-trained)
| v3 | `anchor_v3_rlpolicy.py` | Offline ES-trained (150 gens, ~4.3h, domain-randomized) residual-MLP concession policy (USE_LEARNED_POLICY=True, 67 weights baked in). Out-of-sample: **+0.0085 realistic / +0.0050 mixed** vs heuristic, improves Q1 (0.939→0.953), deal rate unchanged. Real but BELOW our pre-registered +0.01 ship bar; lab→live transfer unproven. **NOT shipped** — archived. To try it: `cp advisers/anchor_v3_rlpolicy.py anchor.py`. |

## v4 — generalized (short-deadline robustness)
| v4 | `anchor_v4_generalized.py` | = v2 heuristic + MIN_RESCUE_ROUNDS 5->4, MIN_SECURE_ROUNDS 3->2 (holds slightly longer before end-game; lifts short-deadline scores ns10 0.86->0.87, neutral elsewhere). Verified robust across n_steps 10..10000 (no timeout: 6.7s max at ns=10000) and domain types. The competitive/small-domain regimes are structurally capped (advantage~0, no agreement zone) and unfixable. This is the most robust heuristic version. |

## v5 — RL retrained on weak regime: REJECTED (does not generalize)
| v5 | `anchor_v5_rlgen_REJECTED.py` | RL residual-MLP retrained on the WEAK regime (mixed mode, n_steps=60). Training-val +0.0071, but OUT-OF-SAMPLE neutral: realistic −0.0001, mixed +0.0018, short ns20/30 −0.0004/−0.0009. Below the +0.01 bar; did not transfer. NOT shipped. **v4 remains the shipped agent.** |

## v6 — pairwise opponent model + emit/bid DECOUPLE (SHIPPED)
| v6 | `anchor_v6_pairwise_decouple.py` | **First validated improvement over v4.** `MODEL_KIND="pairwise"` (value score = their_freq − 0.3·our_freq: values we keep offering and they keep rejecting are bad for them — signal the frequency model discards) raises tau_me +0.07 (0.571→0.642, robust across GAMMA). `DECOUPLE_BID=True` emits that accurate model (scored → tau_me up) but BIDS on the plain frequency model (reveals no more than v4 → tau_opp flat), lifting the Concealing SHARE at ZERO advantage cost. Validated +0.006–0.009 OVERALL across 4 independent regimes (ANL seeds 7000/9000, old realistic, short ns=40); rivals concealing +0.013–0.014; advantage preserved; firm/conceders flat; beats frozen-v4 (ShippedV4Rival) head-to-head. |

## SHIPPED: v6_pairwise_decouple (active anchor.py, submission.zip)
v4's robust heuristic bidding + the pairwise/decouple opponent-model upgrade. Delta vs
v4: `MODEL_KIND="pairwise"`, `DECOUPLE_BID=True` (PAIRWISE_GAMMA=0.3). All v4 robustness
(n_steps 10–10000, crash-free, structural competitive ceiling) preserved — bidding is
unchanged; only the EMITTED model and a separate bid-model differ. RL (v3, v5) gave no
out-of-sample gain. v4_generalized remains the fallback (and is wired as ShippedV4Rival
for self-play regression testing).
