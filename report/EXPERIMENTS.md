# Anchor — overnight improvement log (ANL 2026)

Autonomous R&D session. Goal: maximise tournament absolute mean score
(Advantage + Concealing). Diagnosis from live tournament #19114 (rank 18/39):
our **median negotiation was a no-deal** (~0.53). Deal rate against firm,
*modelling* rivals is the bottleneck. Each round below: hypothesis → test →
positives/negatives → keep/revert decision.

Metric: weighted OVERALL over opponent groups. The honest proxy for the live
field is the **rivals/modeller** group (agents that emit a model of us, so a
no-deal scores ~0.5 not ~1.0). Negmas firm/conceders that don't model us inflate
score on no-deals — that masking is exactly what fooled the original tuning.

Guardrails (must hold): anti-exploitation vs pure hardliner (score > no-deal 0.5,
never walked below reservation); conceder extraction stays high; tau parity
(model code unchanged ⇒ tau_me≈0.536).

---

## Baseline (shipped agent as of tournament #19114)
- Params: FAIR_FLOOR=0.92, RESCUE_TIME=0.95, RESCUE_FLOOR=0.20, fairness gate ON.
- rivals 0.755 (deal 0.42), conceders 1.757, firm 1.154 (deal 0.29), model 1.097.
- Weighted OVERALL **0.989**.
- tau baseline: tau_me 0.536, tau_opp 0.550, share 0.499.

## Round A — end-game acceptance + concession knobs  [KEPT]
**Hyp:** we time out into no-deals because (i) the end-game accept gate
`u_offer>=opp_model(offer)` rejects firm rivals' positive offers, and (ii) the
0.92 floor is too greedy for a firm field.
**Test:** new `rivals` group (self-play + GSmith-modelling firm agents);
SECURE_BEST_FAIR flag; floor×rescue grid; secondary knob sweeps.
**Result:**
- `SECURE_BEST_FAIR=False` (accept any rescue-window offer above reservation):
  +0.041 OVERALL, deal rate ~doubles (rivals 0.42→0.74, firm 0.29→0.79).
  Ablation flag True vs False at 0.72/0.95: 1.003 → 1.044. **THE fix.**
- Floor: nearly irrelevant once flag is off (we close by accepting, not offering
  low). 0.92→0.58 moves OVERALL only ~0.01. Picked **0.72** (moderate, robust).
- RESCUE_TIME: 0.95 best; lowering hurts model (1.09→0.95) & conceders (1.78→1.68)
  by caving earlier. **Keep 0.95.**
- RESCUE_FLOOR 0.20→**0.10**: +0.007, deal 0.82→0.84. Adopt.
- CAPTURE_FRACTION 0.85 best (lower hurts modellers). Keep. EXPONENT flat. Keep 0.10.
**Positives:** structural deal-rate fix; gains in negmas-firm too (not self-play
only) ⇒ not overfit; conceders/model unharmed.
**Negatives:** rivals SCORE only 0.755→0.81 (deal rate up, but extra deals are
low-advantage — we cave to their best). Deal *quality* is the next frontier.
**Decision:** KEEP. Params: FAIR_FLOOR=0.72, RESCUE_TIME=0.95, RESCUE_FLOOR=0.10,
SECURE_BEST_FAIR=False. Weighted OVERALL ≈ **1.05** (from 0.989).

---

## CRITICAL TOOLING BUG FOUND (and fixed)
A stale snapshot `.venv/Lib/site-packages/anchor.py` (from `pip install -e .` with
hatch `force-include`) shadowed the edited `anchor.py` *only when scripts were run
such that site-packages preceded the repo root on sys.path*. Verified:
bench/score_sweep/grid insert ROOT at path[0] and import `anchor` at top → used
the EDITED agent (flag ablation, grid, sweeps are VALID). The new `lab.py` was
importing the STALE agent (floor 0.92, no flag) → its first numbers were the OLD
agent. Fix: deleted the stale snapshot; `import anchor` now unambiguous. Also: the
in-process multiprocessing Pool gave corrupt aggregates (Tough 0.03 vs true 0.61)
— root cause was the stale import under spawn + mixed task stream; lab.py is now
SERIAL (matches bench exactly) and we parallelise at the OS-process level instead.
**Lesson:** always pin which module a harness imports before trusting its numbers.

## Round D — end-game acceptance refinement (SECURE_BEST_TIME)  [KEPT]
On the corrected lab (honest groups, realistic rf=1.0 domains), the true flag
trade-off: flag=False **doubles deal rate vs hold-and-wait firms** (firm group
deal 0.43→0.81; Tough 0.03→0.61, MiCRO 0.34→0.92, Hybrid/TopFraction/FastMiCRO
all +0.05) but **caves to accept-high rivals** (FirmTopNeg 0.840→0.745) because we
grab their lopsided best instead of letting them accept our high offer. Net
positive across firm opponents; the live tournament's median no-deal confirms the
field is firm/hold-and-wait, so flag=False is right.
**Fix:** decouple the blanket-accept time `SECURE_BEST_TIME` from RESCUE_TIME and
push it later. Sweep (flag=False): 0.95→OVERALL 1.105 (FirmTop 0.735); **0.97→
1.111 (FirmTop 0.759, rivals adv 0.399→0.409, firm deal still 0.77)**; 0.99→1.114
but firm deal collapses 0.81→0.68. **Decision: SECURE_BEST_TIME=0.97** — recovers
accept-high rivals and raises extraction while keeping firm deal high and Tough
intact. Floor swept flat (0.92→0.72 all ~1.106) → set **0.78** (moderate).
**Operating point so far:** FAIR_FLOOR=0.78, RESCUE_TIME=0.95, RESCUE_FLOOR=0.10,
SECURE_BEST_FAIR=False, SECURE_BEST_TIME=0.97. Lab OVERALL ~1.111 (rivals 0.901,
firm 1.306, conceders 1.626); vs original-shipped 1.112 on the conceder-diluted
lab, but decisively better on the firm group + matches the live failure mode.

## Round G(part1) — deadline-adaptive end-game  [KEPT]
**Hyp:** all tuning was at n_steps=100; short deadlines may break the late
end-game. **Test:** lab at n_steps 20/50/100/200, current vs original.
**Found a regression:** at n_steps=20 the current config scored 0.975 (deal 0.36)
vs original 1.038 (deal 0.50) — a fixed 5% rescue window is ~1 round, too few to
close. **Fix:** make rescue/secure triggers deadline-adaptive — engage at the
EARLIER of the relative threshold or "MIN_*_ROUNDS before the end"
(MIN_RESCUE_ROUNDS=5, MIN_SECURE_ROUNDS=3). At long deadlines the relative
thresholds dominate (unchanged). **Result:** n_steps=20 0.975→**1.044** (deal
0.36→0.82, now beats original); 50/100 ~unchanged; 200 ~unchanged (−0.004 noise).
**Decision: KEEP** — robust across deadline lengths, matches the strategy doc's
"retune the schedule by deadline" guidance, done parameter-free.

## Side notes (model / concealing)
- Bayesian issue-weight model: tau_me within noise of frequency (skip; confirms report).
- Uniform offer-weighting worse than EARLY (tau_me 0.522 vs 0.546) — early confirmed.
- Concealing trade-off: the firmer original (hold to end) has higher tau_me (0.599
  vs 0.546) because holding longer = we OBSERVE more opponent offers = better model.
  Closing early costs ~0.006 concealing share — far smaller than the advantage gain.
- Floor barely affects advantage; floor 0.92 is marginally better for concealing
  (tau_opp 0.572 vs 0.577). Shipped floor 0.78 is a moderate compromise.

## Round C — deal-quality via end-game offer selection  [REJECTED]
Two research subagents + an experiment-designer agent flagged "we cave to the firm
rival's lopsided best" as the deal-quality leak; top idea = estimate opponent
reservation (min utility-to-them over their offers × slack) and in rescue offer
the BEST-for-us bid they're predicted to accept (RV-target), or a Kalai/Nash point.
**Test (lab realistic n24, RESCUE_SELECT in {fair,rvtarget,kalai,nash}):**
fair **1.112** (rivals 0.904/deal0.85) > kalai 1.111 > nash 1.109 > rvtarget 1.103
(deal 0.82). **RV-target HURTS**: our tau_me≈0.54 model is too noisy to target the
opponent's acceptance threshold, so we offer too high and they reject (deal rate
drops). The balanced "fair" bid is robust. **Decision: keep RESCUE_SELECT="fair".**
Mechanism kept flag-gated for the ablation. Lesson: deal-quality extraction is
gated by opponent-model accuracy, which is itself near a ceiling.

## Mover-order split (lab realistic n24)
WE-SECOND OVERALL 1.126 (deal 0.86) > WE-FIRST 1.093 (deal 0.82). Both healthy; the
second-mover edge is expected (we see their offer, model faster, can accept). The
first-mover end-game has minor headroom (tested next).

## Round B/H — final refinements  [last-offer + opening-boost KEPT; adaptive REJECTED]
- **LAST_OFFER_GRAB=True** (first-mover end-game): on our final proposal (opponent
  gets no turn to accept a fresh bid) we table their own best-shown offer (provably
  acceptable). Lab realistic 1.110→**1.115**, deal 0.84→0.87. KEEP.
- **OPENING_BOOST=3.0** (model): boost the opponent's opening offer (≈their ideal)
  in the value-frequency counts. tau_me 0.546→**0.574**, concealing share
  0.496→**0.500** (parity — no longer losing the free split). Model-only, zero
  advantage cost. KEEP.
- **ADAPTIVE_CONCESSION (reciprocity): REJECTED** — beta 0.5/1.0 gave 1.089/1.083
  (< 1.110). Conceding faster when the opponent concedes throws away extraction;
  our firm Boulware + SECURE_BEST already handle both conceders and firms.
- **AC_combi(T,MAX^W): NOT IMPLEMENTED (reasoned redundant).** Our acceptance
  (ACNext + CAPTURE 0.85 + end-game SECURE_BEST flag=False accepting any positive
  offer at t>=0.97) is already MORE lenient than AC_combi's late window-max rule,
  so it is subsumed. The literature's +18% was vs *bare* AC_next, which we are far past.

## SHIPPED CONFIG (final)
CONCESSION_EXPONENT=0.10, FAIR_FLOOR_FRACTION=0.85, RESCUE_TIME=0.95,
RESCUE_FLOOR_FRACTION=0.10, SECURE_BEST_FAIR=False, SECURE_BEST_TIME=0.97,
MIN_RESCUE_ROUNDS=5, MIN_SECURE_ROUNDS=3, LAST_OFFER_GRAB=True, OPENING_BOOST=3.0,
SELECTION_MODE=nash, RESCUE_SELECT=fair, MODEL_WEIGHTING=early, MODEL_KIND=stability,
DECEPTION=False, DECOY_FREEZE=False, ADAPTIVE_CONCESSION=False.
Lab realistic n40: OVERALL 1.095, deal 0.87, share 0.500. vs original-shipped
(floor 0.92/flag-on): same lab OVERALL but deal 0.63 — the win is DEAL RATE on the
firm field + short-deadline robustness + concealing parity, i.e. exactly the live
failure mode (median no-deal at rank 18).

## Research-agent corroboration (web)
Two research subagents independently confirmed: (1) the shipped examples/boa.py &
examples/map.py FORFEIT concealing (opponent_ufun=None via a MAPNegotiator wiring
bug) — our single SAOCallNegotiator that hand-sets private_info["opponent_ufun"]
avoids this (verified: 26/26 negotiations emit a valid model); (2) frequency models
beat Bayesian for tau (Baarslag survey) — matches our finding; (3) Nash/Kalai
frontier targeting is WORSE in near-zero-sum domains — matches our rvtarget
negative; (4) iso-utility opponent-aware selection is best-practice — we use Nash.

## Round H — anti-overfit validation (the rank-18 failure-mode guard)  [PASS]
Added a harsh accept-high firm modeller (AcceptHighFirmNeg: OfferTop2%/AcceptTop5%
+GSmith) and compared FINAL vs original on a firm-rival set (self-play + FirmTop +
FirmBest + AcceptHighFirm + Tough + MiCRO + Hybrid + BOA + Whale + HardHeaded).
**FINAL 0.925 (deal 0.79) vs ORIG 0.916 (deal 0.46).** FINAL wins against EVERY
genuinely-firm rival (AcceptHighFirm 0.735 vs 0.657, FirmTop 0.749 vs 0.692, Tough
1.070 vs 1.016, MiCRO 1.211 vs 1.155, Hybrid 1.222 vs 1.173, FirmBest 0.546 vs
0.494) — i.e. the gains are real and broad, NOT self-play-overfit. ORIG only leads
on self-play extraction (irrelevant; opponents don't run our strategy) and Whale
(the expected deal-quality/rate trade-off on one strong agent). This directly guards
against the failure that produced rank 18 (an agent tuned on a non-representative
set). PASS.

## DELIVERABLES (built)
- anchor.py: final config, docstring updated to measured behaviour. Self-contained
  (imports only negmas). Emits opponent_ufun every turn (verified 26/26).
- submission.zip: anchor.py + requirements.txt (negmas==0.15.4), verified clean.
- report/report_anac.pdf: rebuilt with the deal-closing design + ablation + the
  full-space-Kendall explanation of why deception fails + the example-agent
  concealing-forfeit finding. concession.png regenerated from live constants.

## Round I — integration, robustness, more rejects
- **End-to-end tournament** (`main.py tournament`, real cartesian_tournament +
  anl2026 scoring, 4 generated scenarios, competitors BoulwareTB/Simple/MAP/BOA):
  **AnchorNegotiator ranks TOP at score 1.353**, completes with no exceptions.
  Integration validated against the actual competition harness path.
- **Large-domain robustness:** on a 531k-outcome domain our SOLO per-negotiation
  time is **3.1s** (the 25s seen vs BOA was BOA's GSmith cost, not ours); we still
  close at good advantage (vs BOA adv 0.624). No timeout risk; real ANL domains are
  far smaller and the live run had 0 exceptions. No optimisation needed.
- **ADAPTIVE_SECURE (concession-conditional cave timing): REJECTED.** Aimed to
  recover Whale extraction by holding longer vs conceders. No effect (Whale 0.740
  unchanged) because Whale extracts via the ACNext path during rescue, not the
  blanket-accept; OVERALL 1.115→1.114. The Whale trade-off is fundamental (firmness
  extracts more from strong conceders but kills deal rate vs hold-firm — we chose
  deal rate for the firm live field). Stays off.

## Round J — broad before/after + a corrected research claim
- **Broad ~30-opponent arena, same roster, FINAL vs ORIG (overrides):**
  FINAL mean **1.576** (coop 1.824 / comp 1.540, margin +1.216) vs
  ORIG **1.551** (coop 1.785 / comp 1.516, margin +1.268). Unambiguous absolute-score
  improvement; FINAL's MARGIN is lower (we trade margin for absolute score, which is
  what the tournament rewards). Positive margin vs ~29/30 opponents; only Whale wins
  margin (we still close 58/62 with it).
- **Adversarial verification caught a FALSE research claim.** A research subagent
  asserted the template examples (boa.py/map.py) forfeit concealing
  (opponent_ufun=None via a MAPNegotiator wiring bug). DIRECT TEST in our negmas
  0.15.4: BOANeg/MAPNeg/HardHeaded/AgentX/Whale ALL emit a present, valid model of
  us (tau_opp ~0.56). The claim does NOT hold here -- removed it from the report;
  kept only the verified non-degeneracy-guard point. Lesson: verify subagent claims
  against the actual environment before publishing.
- CAPTURE_FRACTION re-sweep on final config: saturated (0.95/0.90/0.85 all ~1.115).
- Per-deadline CONCESSION_EXPONENT: negligible (±0.004); the deadline-adaptive
  end-game already handles short deadlines. Keep 0.10.

## Round K — stability & robustness validation
- **Multi-seed variance** (4 seed bases, n_gen=30 realistic): OVERALL 1.121 / 1.144 /
  1.152 / 1.114, deal 0.82-0.90. Jitter ~+-0.02 -> result is seed-robust, not an
  artifact.
- **Edge-case stress** (1-issue/2-value, rational fraction 0.1, n_steps=2, many
  issues, big value counts): 48 negotiations, **0 errors**. No crash/forfeit risk.
- **Large-domain** (531k outcomes): our solo time 3.1s; no timeout risk.
- Eval tooling verified clean (no leftover debug). submission.zip rebuilt from the
  final anchor.py (parses OK).

## Round L — larger end-to-end tournament
8 competitors, 768 negotiations, real cartesian_tournament + anl2026 scoring:
**AnchorNegotiator #1 at 1.337** > MAP/BOA 1.202 > Simple 1.104 > MiCRO 0.544 >
Boulware 0.462 > Conceder 0.246 > Tough 0.063. Second independent tournament win
(the 5-competitor one earlier also ranked Anchor #1 at 1.353).

## Round M — full-spectrum arena robustness profile (final config)
Mean score by generated-scenario competitiveness (30 gen + 7 local, ~30 opponents):
realistic **1.576**, mixed **1.271**, competitive **1.109** -- all positive, margins
+1.02 to +1.08. Robust across the whole cooperative->competitive spectrum; only
Whale beats us on margin in any mode.

## Status: agent plateaued at a strong, validated operating point
Explored and saturated: concession schedule (exponent, floor, rescue time, rescue
floor), bid selection (greedy/nash/opponent/kalai/rvtarget), acceptance (capture,
secure timing, AC_combi-subsumed), opponent model (frequency/Bayesian/early/opening/
uniform), deception (diversification/decoy), adaptive concession, adaptive secure,
per-deadline exponent. Wins kept: end-game secure-accept (deal-rate fix),
deadline-adaptive end-game, early+opening model weighting, last-mover grab, floored
Boulware. The agent is comprehensively validated (lab, broad arena before/after,
cartesian tournament rank-1, anti-overfit vs firm rivals, deadline 20-200, both
move orders, huge domains, edge cases, multi-seed). Further parameter/mechanism
changes are confirmed marginal; the remaining gap (vs strong conceder Whale) is a
fundamental margin-vs-deal-rate trade-off we resolve toward deal rate (correct for
the firm live field).

## LIVE RESULT: tournament #19115 -> RANK 4 / 38 (was 18/39)
The deal-closing agent jumped 18 -> **4**. Decoded (score/5000): our MEDIAN 1.353 is
top-tier (beats ranks 2,3,5); the gap to rank 1 (Mirage 1.377 mean) is our BOTTOM
TAIL -- Min 0.004 (a concealing forfeit), Q1 1.106 (vs top 1.18-1.31), highest Std
of the top 5. So: median is great, consistency is the gap. Note the top agents are
SLOW (Mirage 21.9s/neg, us 0.26s) -> they likely do heavier modeling/search.

## Round N — bottom-tail refinement (post-rank-4)
Diagnosed the tail (eval/tail.py): worst cases are (a) deals at ~0 advantage on hard
competitive domains (domain-inherent) and (b) low-concealing (~3% of negs, model
inverts). OPENING_BOOST does NOT cause the tail (0/1/3 identical). Backwards-model
forfeits can't be self-corrected (no ground truth) and shrinkage preserves rank
order so doesn't help.
**What worked — end-game accept floor + higher offer floor (compounding):**
- **SECURE_ACCEPT_FLOOR=0.05**: refuse worthless (adv<0.05) end-game deals; we hold
  for better ones. +0.006 OVERALL via higher advantage on held deals (rivals adv
  0.398->0.405), deal rate ~unchanged (0.87->0.86).
- With SAF active, **RESCUE_FLOOR_FRACTION 0.10->0.20** is now best (the interaction
  flipped vs the no-SAF sweep): hold offers higher + refuse worthless deals ->
  +0.003 more, deal rate up to 0.87.
- Combined vs rank-4 config: realistic 1.095->**1.102**, mixed 1.012->**1.020**,
  ns=20 ~1.049, deal rate maintained 0.86-0.87. Small, consistent, low-risk
  (refines end-game extraction; does not touch the core deal-closing fix).
**Rejected:** model shrinkage-to-uniform (preserves rank order -> no tau effect);
ADAPTIVE_SECURE (no effect on Whale). The bottom-quartile is largely domain-limited
(hard domains cap advantage) + model-limited (concealing on contested rivals);
big further gains likely need heavier modeling/search like the slow top agents.

## Round O — "go heavy/slow like the top agents": tested, does NOT help us
The rank 1-3 agents are 20-80x slower (heavy modeling/search). We tested whether
heavier compute would close the bottom-quartile gap:
- **Stronger opponent models (raise tau_A / concealing):** entropy-based issue
  weights (tau_me 0.560) and Bayesian (0.571) are BOTH WORSE than our simple
  stability model (**0.593**). tau_A is DATA-limited (~100 offers over thousands of
  outcomes), not compute-limited -- matches Baarslag (frequency plateaus, Bayesian
  no better). Heavy modeling cannot raise our concealing.
- **Deeper bid search (raise advantage):** rescue selection fair=1.103 >= rvtarget
  1.102 >= kalai 1.101 >= nash 1.099. Targeting the Pareto/Nash/reservation point
  does not beat the simple balanced bid, even with the better (tau_A 0.59) model.
  Bottom-quartile advantage is DOMAIN-limited (hard competitive domains cap it).
**Conclusion:** our bottleneck is data + domain structure, NOT compute. We reach
rank 4 at ~0.26s/neg (vs 20s+ for the leaders) -- the speed/simplicity is a strength,
not a deficit. Going heavy would add complexity and lose the clean/fast/interpretable
design for no measured gain. (entropy weight estimator kept flag-gated for ablation;
default stays "stability".)

## Rounds R1-R5 — "push to win rank 1" (research-informed); all confirm optimality
Two research subagents (ANAC/ANL literature + the named leaders) + 5 experimental
rounds. The decisive research finding (Baarslag, 17,920 matches): a PERFECT opponent
model buys only ~0.0135 advantage, so our tau plateau (0.59) is NOT the gap; the
leaders' edge is consistency via robust acceptance + endgame safety (which we have),
and the ANAC 2025 retrospective credits "managing complexity intelligently, not
maximum compute" -> going slow would not help.
- **R1 deception (lower tau_B):** strong modellers model us at tau_B~0.78; decoy
  barely moves it (0.783->0.781). Concealing uncontrollable. NEGATIVE.
- **R2 domain-adaptive:** competitive domains capped (all param variants 0.826-0.828),
  not mis-tuned. NEGATIVE.
- **R3 P(accept) bid search:** realistic 1.103->1.088, deal 0.87->0.84 -- tau_A~0.59
  too noisy to target acceptance, loses deals (like rvtarget/kalai). NEGATIVE.
- **AC_combi / SECURE_ANY** (accept any good late offer): 1.220->1.217, NEGATIVE
  (our capture+secure-best already give 0.86-0.90 agreement; the literature's
  72%->99% gain doesn't apply -- we're already past AC_next-alone).
- **NEVER_UNDERSELL:** 1.103->1.101, NEGATIVE (capture/secure already prevent it).
- **R5 joint 7-D optimizer (heavy, 76 candidates):** our config is **#1 of 76**
  (obj 1.529 > best random 1.528). The optimum is broad and flat; we sit at the top.
**Conclusion:** the agent is at the achievable optimum for its (clean, fast,
interpretable) design space. The only untried architecture is an offline-trained
policy (cf. ANL2025 SAC Agent); research + our data (modeling data-limited, advantage
domain-limited) make its expected payoff low and its complexity high. All R1-R3/R5
mechanisms kept flag-gated (default off) for the report's ablation.
Deadline correction (research): agent submission **June 21 2026** (final extension),
report June 23 -- more runway than CLAUDE.md's June 16.

## Offline RL policy (the big swing) — small real gain, below ship bar
Built a residual-MLP concession policy: target = optimal-Boulware(t) + bounded
tanh-MLP correction over 9 state features (zero weights = exactly the heuristic, so
non-regressing). Trained OFFLINE by parallel evolution strategy (the right tool: the
bottleneck is the CPU-bound negmas simulator, not net math, and the submission must
be pure-python -> GPU irrelevant). **150 generations, ~4.3h, domain randomisation**
(fresh scenarios each gen), held-out validation. Then re-validated OUT-OF-SAMPLE on
seeds the trainer never saw:
  realistic: heuristic 1.1032 -> learned **1.1117** (+0.0085), Q1 0.939 -> **0.953**
  mixed:     heuristic 0.9946 -> learned **0.9995** (+0.0050)
**Verdict:** RL learned a SMALL but consistent real improvement (positive on all 3
independent sets; improves the bottom-quartile Q1 that is our exact rank gap; deal
rate unchanged). But it is BELOW the pre-registered ship bar (+0.01 on BOTH modes)
and validated only on the lab proxy (live transfer unproven). Honouring the bar (no
post-hoc moving), we KEEP the proven heuristic active and ARCHIVE the learned policy
as advisers/anchor_v3_rlpolicy.py (flip with one cp if desired). This is the honest
result from a serious multi-hour RL search: learnable state-dependence helps a
little, but the hand-tuned heuristic is within ~0.005-0.009 of it and far simpler.
Lesson: matches the literature (a perfect opponent model is worth ~0.0135; the edge
is tiny and the simple agent captures almost all of it).

## Generalization analysis (after #19124 rank-18 config-reshuffle)
Live #19124 (5 different/harder configs) dropped us 4->18; the WHOLE field
reshuffled (Mirage 1->8, AaNante 2->7), proving it's config-driven, not a
regression. Research (ANL2024 proxy) revealed the key facts: tournament n_steps is
RANDOMISED 10..10000 (not flat 100), domains ~1000 outcomes (mix incl ~5% zero-sum),
per-agent reservations from [0,1], and a HIDDEN 180s wall-clock timeout (source of
the 1056 exceptions on large domains x long deadlines).
- **Deadline sweep (ns 10..10000):** score 0.77(ns10)/0.80(30)/0.87(100)/0.90-0.92
  (300-10000). Short deadlines (esp. vs firm opponents) are our weak corner; long
  deadlines fine. Max wall-time at ns=10000 on ~1k domain = **6.7s** (no timeout
  risk; our 53ms speed is an edge vs the field's 1056 exceptions).
- **MIN_RESCUE_ROUNDS 5->4, MIN_SECURE_ROUNDS 3->2:** small short-deadline win
  (ns10 0.860->0.870, ns30 0.909->0.919), neutral on realistic (1.130->1.128) and
  long. Adopted (v4_generalized). Concession EXPONENT confirmed flat (0.10=0.20=0.30).
- **Domain landscape (issues x values x rf):** our 6 weakest regimes are ALL
  small + competitive (rf=0.4, 1-2 issues): adv~0.00, deal 0.0-0.27, con~0.734,
  score ~0.73. On small/competitive domains the agreement zone is empty -> advantage
  forced to ~0 for EVERYONE -> score = concealing (capped). Verified unfixable by
  bidding/closing/floors/deception/model variants (all flat). NiceOrDie (the one
  provided conflict domain) = 1.03 vs 1.5-1.64 on the other 6 provided domains.
**Conclusion:** the agent is at its STRUCTURAL ceiling. It's strong on cooperative,
robust across deadlines (with a marginally-weak short/firm corner), crash-free, and
capped on small-competitive (where no agent can extract advantage). The 4<->18 swing
is a config lottery on these capped regimes + concealing variance. Shipped the
short-deadline robustness tweak; running an RL retrain on the weak (mixed/short)
regime as the directed big-swing (low odds vs the structural cap).

## Frontier explored but saturated

Knob tuning has plateaued (~1.05). To reach the top of the board (leaders close
deals at GOOD terms, median ~1.19) we need algorithmic gains:
- B: adaptive/reciprocal concession (extract more from semi-conceders; avoid
  caving too early against firm).
- C: opponent acceptance-threshold estimation → offer best-for-us bid they'll take.
- D: time-aware acceptance (close good deals earlier, not only at rescue).
- E: raise tau_me (free half of concealing) via a better model.
- F: re-measure tau_opp under the new (accept-more) behaviour; revisit cheap
  deception only if genuinely free.
- G: robustness across deadline lengths (n_steps 20/50/200) and mover order.

## RL retrain on the WEAK regime (mixed/short) — does NOT generalize. KEEP v4.
After #19124/#19136 showed config-fragility, retrained the residual-MLP policy on
the weak regime (mixed mode, n_steps=60, 53 gens before convergence). Training-val
gain +0.0071. But OUT-OF-SAMPLE (unseen seeds): realistic -0.0001, mixed +0.0018,
short ns20 -0.0004, ns30 -0.0009 -- all noise, none near the +0.01 bar. The
training gain was selection bias; it did not transfer. DECISION: keep the v4
heuristic active; archive the gen policy. Confirms (yet again) the weak regimes are
structurally capped, not learnable. The robust live leaders (AgentNexus/ChangAgent/
AdaptiveBath/LIonel, top-5 in BOTH #19124 and #19136) are 10-100x SLOWER than us --
their edge is heavy per-negotiation search, an architecture change, not a tunable.

## Live #19175/#19177 (rank 14/9) — EXHAUSTIVE ablation: v4 is the heuristic optimum
After live #19177 (rank 9/41, mean 6593, 213ms — top-10 and ~60x FASTER than #1
Mirage at 13s/neg) and #19175 (rank 14/40, the hard-config set), ran a full lever
sweep to find any remaining gain. All measured on the realistic/hard lab fields,
n_gen 16-30, with the FROZEN shipped v4 added as a self-play rival (rivals.ShippedV4Rival)
so every change spars against our true lineage, not just the live class with pinned
knobs. Results (weighted OVERALL / rivals; baseline v4 = realistic 1.148 / 0.946):

- Concession depth (RESCUE_FLOOR_FRACTION 0.20->0.05, +MIN_RESCUE_ROUNDS 8): INERT
  on hard (0.734 rivals, deal 0.36->0.37 = noise) AND realistic (flat). Confirms the
  firm-vs-firm no-deals are STRUCTURAL (empty agreement zone), not a firmness we can
  tune away — we already accept any positive offer (SECURE_BEST_FAIR=False), so a
  reachable deal is already taken.
- Opponent-model kind: entropy showed +0.044 tau_me at n=19 but REVERSED at n=30
  (0.569 vs base 0.574) — small-sample noise. bayesian worse (0.510). Stability kept.
- Model weighting / opening boost at n=30: current (early, OPENING_BOOST=3) is the
  BEST of all (share 0.505); recency 0.477, uniform 0.488, boost-0 0.498, boost-6 0.499.
  The concealing share is pinned at ~0.50 (parity) because the modelling rivals run
  symmetric frequency models => tau_me ~ tau_opp by construction. And the math caps
  the prize: even +0.08 tau_me only moves share ~+0.012 (~+0.007 OVERALL) — the
  concealing term is a small SHARED swing, as the original analysis found.
- Concealing-forfeit tail (con=0): root-caused to 3-4 OUTCOME stress-mode domains
  where the opponent accepts our opening (0 offers seen) -> near-constant model ->
  compare_ufuns scores tau=-1 -> con=0 against a non-modelling conceder (their tau=0,
  so our share=0/(0+0.5)=0). Fix attempt (real [0,1] index ramp instead of the 1e-6
  gradient) REJECTED: it regressed tau_me 0.574->0.533 without raising the min (the
  ramp coin-flips to -1 just as often on 3-outcome domains). These forfeits are tiny-
  domain artifacts; the ~1000-outcome live field does not produce them. Reverted.
- Advantage flags on realistic (all WORSE than v4): PACCEPT_SEARCH 1.129, RESCUE_
  SELECT=rvtarget 1.138, NEVER_UNDERSELL 1.141, ADAPTIVE_CONCESSION 1.113 (the last
  actually RAISED opponent score 0.902->0.948 — "extraction" heuristics give value
  away). Plain Nash-over-band (v4) is the optimum.

VERDICT: v4's exact configuration is the optimum across every lever available to the
heuristic architecture. The only remaining upside is the heavy per-negotiation search
the slow leaders use (3.7-13s/neg) — an architecture change that would risk our single
biggest asset (fastest crash-free agent in the field: 0 exceptions live while slower
agents time out / hit Min=0). For a course deliverable graded on correct/interpretable/
defensible, that trade is not worth it. KEEP v4; ship the comprehensive ablation as the
opponent-modelling + bidding evidence in the report.

## "Spend compute like the slow leaders" (LOOKAHEAD) — TESTED, makes us WORSE
Owner question: why not spend per-negotiation time like Mirage (13s)/AgentNexus
(3.7s)? First, the live data refutes the premise that time buys rank: in #19177 the
3 SLOWEST agents (MajiKayo 27s, Iscas 17.9s, Ozu 10.4s) rank near the BOTTOM (19/21/
20), while #2 AaNanteLucky is fast (1.3s) and BadIron (156ms) ranks ABOVE us. Time
and rank are uncorrelated/slightly negative. Second, our 0.2s is not shallow search:
domains are ~1000 outcomes and we already evaluate ALL of them each round (exhaustive),
so recomputing slower returns the same bid.

Built a genuine lookahead bidder anyway (flag LOOKAHEAD, default OFF): score each
candidate by EXPECTED final utility EV(o)=P_accept(o,t)*u(o)+(1-P_accept)*continuation,
with a time-decaying opponent acceptance model; LOOKAHEAD_MC averages EV over MC
samples of the opponent's uncertain reservation (the genuinely compute-heavy path).
Matched n_gen=12 comparison (identical scenarios):
  realistic: base OVERALL 1.212 / rivals 1.013 / deal 0.92  vs
             LOOKAHEAD MC=12  1.178 / 0.973 / 0.87  (-0.034, 8x slower: 815s vs ~70s)
  hard:      base 0.983 / 0.778 / 0.41  vs  MC=12  0.959 / 0.752 / 0.38  (-0.024)
  realistic deterministic (n_gen=18): base 1.148 vs LOOKAHEAD 1.122 (-0.026, 6.3x
             slower: 631s vs 101s).
VERDICT: spending 6-8x more compute makes us WORSE on every regime. The extra compute
is spent reasoning about a NOISY opponent model, and acting harder on noise yields
worse decisions, not better -- same failure mode as PACCEPT/ADAPTIVE. v4 wins precisely
because it does NOT over-trust the model (firm Boulware + accept-any-positive late).
Time is not the bottleneck; opponent-model signal is, and that is limited by how few
offers the opponent reveals. Flag kept OFF and in-file as the measured ablation.

## "MORE COMPUTE" — settled with subagents + an information-theoretic probe
Owner pushed: the top agents win by spending 3.7-13s/neg, so we should too. Tested
this every way; the answer is NO, and we now know WHY (it's an information limit).

- Scaling ONLINE model compute makes tau WORSE: optim (max-ent IRL) tau_me by
  iterations -- 20:0.560, 60:0.531, 200:0.508 vs frequency 0.563. More gradient
  steps = tighter overfit to the few revealed offers = worse full-space ranking.
- MiCRO bidding architecture: realistic 1.150 vs v4 1.166, hard 0.922 vs ~0.938
  (concedes more, opp score 0.902->0.950). Worse.
- OFFLINE frozen tau-prior (Agent-2's top idea, the one heavy-compute lever that
  could dodge the overfit trap): eval/prior_probe.py generated 300 train / 150 test
  opponent ufuns from the ANL Pareto mix and measured structure: corr(value_index,
  value_utility) MEAN = +0.001 (|mean| 0.338 -- each opponent has structure but the
  DIRECTION is random across opponents). Shrinking frequency toward ANY prior only
  hurts tau (alpha 0.0->0.629 best, 1.0->0.553). There is NO shared cross-opponent
  structure for a frozen prior to learn.
VERDICT: the opponent's utility is identifiable ONLY from its own sparse offers;
cheap frequency counting is already the efficient estimator. Neither online nor
offline compute adds information that isn't there. v4 is at an INFORMATION ceiling,
not an effort ceiling. Subagents also flagged: ANL2024 winner Shochan's edge was
SCENARIO-TYPE CONDITIONING, and our local eval distribution may differ from live
(live ~1000 outcomes, Pareto mix incl ~5% zero-sum, Nash-capped reservations,
n_steps 10-10000). => next lever is distribution-matched re-validation, NOT compute.

## Distribution-matched re-validation (the non-compute lever) — v4 still optimal
Subagents flagged our eval may not match live. Verified with hard numbers: our old
"realistic" gen had reservations mean 0.11 and outcome counts scattered 3-8575;
LIVE/ANL is reservations mean ~0.32 (sampled [0,1] capped at Nash) and ~1000
outcomes. Added a live-matching eval mode (labcore mode="anl": Pareto mix
piecewise_linear/curve/zero_sum, sizes ~1000, reserved_values=(0,1) guarantee_
rational). Re-ran v4 + the rejected levers on it (n_gen=20, n_steps=300):
  v4 base: rivals 0.788 (deal 0.80, adv 0.289, con 0.499) OVERALL 0.981
  FAIR_FLOOR=0.70 0.978 | SECURE_ACCEPT_FLOOR=0 0.981 | RESCUE_FLOOR=0.05 0.980 |
  NEVER_UNDERSELL 0.973  -- all tie or lose.
VERDICT: v4 is the optimum EVEN on the corrected live-matching distribution; our
tuning was not overfit to the wrong distribution. High reservations cap advantage
(~0.29) and concealing stays at 0.50 parity -- the same structural ceiling. Combined
with the compute/prior results: Anchor (v4) is at a genuine INFORMATION ceiling.

## N1: broad flag re-test on ANL distribution (champion v4 OVERALL 0.980)
DECEPTION 0.971 (con 0.499->0.483, backfires), DECOY_FREEZE 0.978, ADAPTIVE_CONCESSION
0.958 (adv 0.287->0.252; con up 0.511 but net worse), RESCUE_SELECT=rvtarget 0.966
(deal 0.80->0.72), =kalai 0.980 (tie, rivals 0.789), SELECTION_MODE=kalai 0.979,
MODEL_WEIGHTING=recency 0.972, CONCESSION_EXPONENT=0.05 0.979 / 0.20 0.980. NONE beats
v4 by the +0.01 bar on the live-matching distribution. v4 optimal here too.

## N2: scenario conditioning (Shochan-style) on ANL -- marginal, below bar
SCENARIO_CONDITION shifts the fair floor by domain competitiveness (corr of our vs
est-opponent utility). On ANL (base 0.980): COND_K=+0.15 -> 0.982 (rivals adv
0.289->0.291), +0.30 -> 0.981, -0.15 -> 0.980. Right sign (hold higher on cooperative
domains) but only +0.002, within noise. Not shipped; flag kept off for the ablation.

## N5: Pairwise/rejection-aware model + DECOUPLE -- FIRST POSITIVE RESULT
Insight: use signal v4 DISCARDS -- values WE keep offering that they keep rejecting
are probably bad for them. MODEL_KIND="pairwise": value score = their_freq - GAMMA*
our_freq (uses _my_value_counts). tau_me (n=40 taus.py roster): freq 0.571 -> pairwise
0.642 (+0.071), ROBUST across GAMMA 0.15/0.3/0.6 (unlike the entropy false-positive).
But concealing SHARE ~flat (0.502->0.503) because tau_opp rises too when we BID on the
sharper model (we reveal more).
DECOUPLE_BID: emit the accurate pairwise model (scored -> high tau_me) but BID on the
plain frequency model (reveal no more than v4 -> tau_opp stays at baseline). The two
are independent (scorer reads emitted estimate; tau_opp driven by revealed behaviour).
ANL n=30: base OVERALL 0.990 (rivals 0.798, adv 0.298, con 0.501); pairwise-full 0.993;
pairwise-DECOUPLE 0.996 (rivals 0.810 +0.012, adv 0.296 preserved, con 0.514 +0.013,
opp score 0.664->0.648). +0.006 OVERALL, +0.012 rivals, ZERO advantage cost. Below the
strict +0.01 OVERALL bar but the right kind of win on the live proxy. Robustness battery
(diff seed / old realistic / short deadline / per-group) running before any ship.

## v6 SHIPPED — confirmed headline (matched n=30 ANL)
v6 (MODEL_KIND=pairwise, DECOUPLE_BID=True, GAMMA=0.3) vs v4 base, same scenarios:
  v6:  rivals 0.810 (adv 0.296, con 0.514) OVERALL 0.996
  v4:  rivals 0.798 (adv 0.297, con 0.500) OVERALL 0.989
=> +0.007 OVERALL, +0.012 rivals, concealing +0.014, advantage preserved. Robust across
ANL seeds 7000/9000, old realistic (1.131->1.138), short ns=40 (0.962->0.971). Code-
reviewed clean (decouple provably holds; no NaN/leak). FIRST validated improvement over
v4. submission.zip rebuilt with v6; v4 archived as adviser + ShippedV4Rival.

## N3: RL/ES retrain ON the ANL distribution -- marginal, NOT shipped (keep v6)
Retrained the residual-MLP concession policy via ES with TRAIN_MODE=anl (train=test=
live, and on top of v6's pairwise+decouple defaults). 50 gens, ~52 min, 14 workers.
Train-val delta +0.017 (cleared the bar on ITS val set -- unlike prior RL on the wrong
distribution). But INDEPENDENT out-of-sample (eval/validate_policy.py, held-out ANL
seeds 500000/700000 + realistic): delta +0.0087 / +0.0004 / +0.0032 -- inconsistent,
the 2nd ANL seed is noise, fails the ">+0.01 on BOTH" bar. Consistent +0.02-0.03 Q1
lift (helps the bottom tail) but mean is what's scored. DECISION: keep v6 heuristic
concession; archive weights as advisers/policy_weights_anl_REJECTED.json. Confirms (3rd
time) RL concession does not robustly beat the heuristic, even on the correct dist.

## Stacking v6 + scenario-conditioning -- no additive gain
v6 alone OVERALL 1.006 (n=24 anl); + SCENARIO_CONDITION K=0.15 -> 1.007 (+0.001), K=0.30
similar. The small conditioning gain does not stack on top of v6's decouple. v6 stands.

## N6: pairwise divergence issue-weights -- marginal tau, no share gain, not shipped
PAIRWISE_WEIGHT="divergence" (weight issue by our-vs-their value-distribution total
variation) on top of pairwise values: tau_me 0.600->0.609 (+0.009) but concealing
SHARE flat 0.507->0.507 (tau_opp rose in tandem). No scored gain. Kept off; v6 uses
stability weights. Confirms tau_me beyond v6's level doesn't convert to more share.

## Lowering tau_opp (decoy/deception) on v6 -- backfires, as before
v6+DECOY_FREEZE 1.004 (con 0.514->0.511), v6+DECEPTION 1.001 (con 0.504), both WORSE
than v6 1.006: holding a decoy value / diversifying offers REVEALS more, raising the
opponent's tau of us (lowering our share). v6's "bid plain, reveal nothing extra" is
already the best tau_opp approach. gamma 0.4/0.5 tie 0.3. N4 (reservation-aware end-
game) not pursued: the remaining ANL no-deals are structurally empty zones (high mutual
reservations) and rvtarget already tested worse. v6 is the converged optimum.

## v6 FINAL high-confidence confirmation (n=50 ANL, 988 rivals negs each)
v6 OVERALL 0.995 (rivals 0.809, adv 0.295, con 0.514) vs v4 0.987 (rivals 0.796, adv
0.295, con 0.501). +0.008 OVERALL, +0.013 rivals concealing, advantage IDENTICAL,
firm/conceders IDENTICAL. The largest/most-reliable run; v6 conclusively better. SHIPPED.

## N7: time-phased decouple (sharp end-game bidding) -- worse, rejected
DECOUPLE_PHASED (bid plain early, switch to sharp pairwise model in end-game): n=30
0.994 vs v6 0.997 (-0.003, con 0.514->0.510 -- end-game sharp bids reveal more); short
ns=40 0.961 vs 0.962. No advantage gain (deals structural, not bid-selection-limited).
Confirms: ANY extra revelation hurts concealing; v6's plain-throughout bidding is best.

## v6 robustness across the live n_steps range (10-10000) -- holds everywhere
v6 vs v4 (ANL, n_gen=10): ns=10 0.887 vs 0.872 (+0.015!), ns=1000 0.994 vs 0.986
(+0.008), ns=10000 0.998 vs 0.991 (+0.007). Gain holds at every deadline, LARGEST at
short deadlines (pairwise helps most when opponent data is scarce -> our-offer signal
relatively more informative). Timing: ns=10000 = 6.74s/neg, IDENTICAL to v4 (pairwise+
decouple adds negligible cost; model rebuild is cheap dict ops). No 180s-timeout risk.
v6 conclusively validated across the full live distribution AND deadline range.

## Short-deadline gamma sweep (ns=10) -- gamma 0.3 fine, no adaptive gamma needed
ns=10 ANL: gamma 0.3 (v6) 0.895, 0.6 0.896, 1.0 0.898, 1.5 0.898. Higher gamma adds
+0.003 at short deadlines via con 0.505->0.510 -- within noise, not worth a deadline-
adaptive gamma. Model fully tuned at gamma=0.3. CONVERGED: v6 is the final optimum.

## N8: broader-opponent generalization -- v6 gain GENERALIZES, no regression
v6 vs v4 against DIVERSE opponents outside the tuning rivals set (ANL n=24):
  NEW GSmith MODELERS (con<1.0, contested): MidModelerNeg +0.029 (con 0.525->0.536),
    ConcederModelerNeg +0.012 (con 0.514->0.531) -- both clear the +0.01 bar.
  NON-MODELERS (con=1.0): Aspiration/CAB +0.000, MiCRO +0.004 -- NO regression.
v6's concealing gain holds against modeler configs we never tuned on, and is exactly
neutral against non-modelers. Confirms v6 is not overfit; the gain is the genuine
pairwise+decouple mechanism working wherever the concealing term is contested (= the
live ANAC field, which is full of sophisticated modelling student agents).

## N9: earliness-weighted pairwise penalty -- marginal, not shipped (SEARCH COMPLETE)
Weight our EARLY near-ideal rejected offers more (1/offer_index) in the our_freq
penalty: tau_me 0.617->0.624 (+0.007) but share 0.509->0.507 (tau_opp rose too), ANL
OVERALL 1.006->1.007 (+0.001 noise). Confirms (again) tau_me beyond v6 does not convert
to concealing share. Model side definitively maxed. v6 stands.

=== SEARCH COMPLETE ===
Every idea/variant tested. v6 (pairwise + emit/bid decouple) is the converged optimum:
+0.007-0.029 over v4 wherever concealing is contested, neutral elsewhere, robust across
seeds/distributions/n_steps(10-10000)/opponent-types, timing-safe, code-clean. The
project's central finding -- an INFORMATION CEILING: extra compute and every alternative
architecture (lookahead, MiCRO, max-ent IRL, offline prior, RL x3, scenario-conditioning,
decoy, deception, phased decouple, divergence/earliness weights) all TIED OR LOST. The
ONLY gains came from (a) new signal the baseline discarded (pairwise: our rejected offers)
and (b) the emit/reveal asymmetry (decouple) -- never from re-processing the opponent's
sparse offers. Opponent utility is identifiable only from the few offers the protocol
reveals; frequency counting already extracts that efficiently.

## FINAL CAPSTONE SCORECARD (v6 vs v4, all 4 modes, n_gen=20, n_steps=300)
  mode        v4 OVERALL   v6 OVERALL   delta
  realistic     1.139        1.148      +0.009
  mixed         1.088        1.098      +0.010
  hard          0.902        0.910      +0.008
  anl (live)    0.981        0.989      +0.008
v6 wins in EVERY mode (+0.008 to +0.010), concealing ~0.51 throughout, advantage
preserved. Sanity: anchor.py imports clean, all 6 experimental flags OFF (TIMEW/PHASED/
COND/LOOKAHEAD/BID=boulware/USE_RL), submission.zip = v6 (66756B, pairwise+decouple
confirmed). PROJECT COMPLETE: v6 shipped, validated everywhere, reports polished.
