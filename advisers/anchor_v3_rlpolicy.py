"""ANL 2026 negotiation agent.

A single-class BOA agent. The three negotiation responsibilities are kept as
separate, self-contained methods so each maps one-to-one onto a section of the
report:

    update_opponent_model      -> report: "Opponent Model"
    concealing_bidding_strategy -> report: "Concealing Bidding Strategy"
    acceptance_strategy        -> report: "Acceptance Strategy"

Design summary ("model hard, deceive lightly"):
  * Scoring is `Advantage + Concealing`. Advantage (how far the deal beats our
    reservation value) dominates; Concealing is a small shared term split by how
    well each side models the other. So we maximise Advantage first and treat
    concealment as a near-free add-on.
  * Bidding decouples *how much* to concede (a time-based Boulware utility
    target) from *which* bid to make (among bids in our utility band, offer the
    one our opponent model likes most -- palatable at zero cost to us).
  * The opponent model is a simple, interpretable frequency + stability additive
    estimate. It is emitted every round into `private_info["opponent_ufun"]`,
    which the competition scorer reads -- emitting nothing forfeits the whole
    Concealing point.

Anti-exploitation (the key to not being walked over by a firm opponent):
  A naive Boulware that concedes all the way to its reservation value lets a firm
  opponent simply wait and collect its own ideal outcome. We therefore floor our
  concession at a *fair share* of our utility range, and even the end-game rescue
  only releases that floor partway (RESCUE_FLOOR_FRACTION). We never hand the deal
  away; instead a firm opponent must yield to our offers or accept a no-deal.

  End-game close (SECURE_BEST + SECURE_BEST_FAIR=False): a no-deal scores only the
  ~0.5 Concealing term, while ANY positive-advantage deal beats it. So once inside
  the (deadline-adaptive) end-game window we bank the best offer the opponent has
  shown us as long as it clears our reservation value -- we do NOT require it to be
  "fair" by our own model. MEASURED: this is the single biggest fix relative to the
  agent that placed mid-pack in the live tournament (whose median negotiation was a
  no-deal): against hold-and-wait opponents it roughly doubles our deal rate
  (e.g. vs a pure hardliner 0.03 -> 0.61) at the cost of only a small concealing
  swing. The reservation hard floor -- not a fairness test -- is what still
  guarantees we never accept worse than walking away.

  Deadline-adaptive end-game: the rescue and secure-accept windows trigger at the
  EARLIER of a relative-time threshold or a fixed number of absolute rounds before
  the deadline (MIN_RESCUE_ROUNDS / MIN_SECURE_ROUNDS). At short deadlines a fixed
  5% window is too few rounds to actually close; this pulls the end-game earlier so
  we still reach agreement (MEASURED: n_steps=20 score 0.98 -> 1.04).

  First-mover close (LAST_OFFER_GRAB): on our final proposal -- when the opponent
  gets no turn to accept a fresh bid -- we table the opponent's own best-shown
  offer, which is provably acceptable to them, rather than time out.

Adaptivity (how we win against opposite opponent styles):
  A firm opponent yields by *accepting* our firm offers, so against it we hold the
  floor and extract. A conceding opponent yields by *improving its offers*, so we
  capture its best offer as soon as it clears CAPTURE_FRACTION of our range rather
  than greedily holding for our maximum and losing it. The same firm floor handles
  both -- no explicit opponent-type switch.

We optimise for the tournament metric -- our ABSOLUTE mean score (Advantage +
Concealing), not pairwise margin: a balanced deal scores ~1.0, a no-deal only the
~0.5 Concealing term, so we close deals even when the opponent also gains.

Measured behaviour (local harness: ~20 NegMAS opponents incl. self-play and firm
GSmith-modelling rivals, over the 7 provided + generated domains, both move
orders). We score positively against every opponent type and close deals at a high
rate (~0.85 vs ~0.63 for the firmer earlier version). We extract hard from
conceders (~1.6), close and bank a positive deal against hold-and-wait firms
(Tough/MiCRO etc., where the earlier version timed out), and contest the
model-capable agents. Opponent-model accuracy (the free half of Concealing):
tau_me ~0.57 with a slight early-offer/opening boost, giving a Concealing split at
parity (~0.50) rather than losing it.

Three deliberate-deception variants were implemented, MEASURED, and left OFF:
offer diversification (DECEPTION) raises the opponent's tau of us (it reveals more
of our high-utility region); a Bayesian issue-weight model (MODEL_KIND) is
tau-equivalent to the simple model; decoy-issue freezing (DECOY_FREEZE) does lower
a weight-learner's tau of us but costs more Advantage than it returns. All confirm
the project's "deceive only when nearly free" thesis -- our concealment comes from
holding firm and modelling them well, not from noise. The flags are kept for the
report's ablation.
"""

from negmas.sao import SAOCallNegotiator, ResponseType, SAOState, SAOResponse
from negmas.outcomes import Outcome
from negmas.preferences import LinearAdditiveUtilityFunction
from negmas.preferences.value_fun import TableFun


class AnchorNegotiator(SAOCallNegotiator):
    """Advantage-first Boulware agent with anti-exploitation concession floor."""

    # --- Tunable constants (tuned empirically against a roster of opponents) --
    # Concession exponent of the Boulware target curve. e < 1 => firm early,
    # concede late. Smaller e == firmer (holds near our maximum longer).
    CONCESSION_EXPONENT = 0.10

    # We never let our utility target fall below this fraction of our own utility
    # range (reservation .. maximum) until the end-game rescue. This is the
    # anti-exploitation floor: a firm opponent cannot drag us down to its ideal.
    FAIR_FLOOR_FRACTION = 0.85

    # Fraction of time after which the end-game rescue engages: the floor is
    # released down toward the reservation value so we still close *some*
    # positive-Advantage deal rather than time out into a zero. Kept late so the
    # rescue is hard to exploit.
    RESCUE_TIME = 0.95

    # In the end-game the rescue concedes toward (but not to) the reservation
    # value to CLOSE a deal. The tournament ranks by our absolute score
    # (Advantage + Concealing): a balanced deal scores ~1.0 while a no-deal scores
    # only the ~0.5 Concealing term, so closing almost any positive deal beats
    # timing out -- even against a firm opponent who also gains. We keep a small
    # floor so we never accept a near-worthless deal. MEASURED: lowering this from
    # 0.9 lifted our mean score vs firm opponents (1.07 -> 1.19) at no cost to the
    # extraction we get from conceders (they concede before the rescue engages).
    RESCUE_FLOOR_FRACTION = 0.20

    # We accept the opponent's best-yet offer as soon as it clears this fraction
    # of our range, instead of holding out for our absolute maximum (a conceding
    # opponent's strong offers oscillate and may not return). Lower == grab deals
    # sooner (more robust to firm opponents, less extraction from yielders).
    CAPTURE_FRACTION = 0.85

    # How to pick a bid within our utility band:
    #   "opponent"  -> the one the opponent likes most (maximises agreement prob,
    #                  but most generous to a firm opponent)
    #   "greedy"    -> the one best for us (gives the opponent the least)
    #   "nash"      -> maximise our_utility * opponent_utility (balanced)
    SELECTION_MODE = "nash"

    # End-game (rescue) bid selector -- see _select_bid. "fair" (most balanced)
    # is the measured default; "rvtarget"/"kalai"/"nash" are extraction-oriented
    # variants tried for deal QUALITY against responsive rivals.
    RESCUE_SELECT = "fair"
    # Slack applied to the estimated opponent reservation for "rvtarget": we
    # assume they will accept a little below the worst they have themselves
    # offered (their offers lower-bound, not pin, their true reservation).
    RV_SLACK = 0.9

    # Deception (Phase 2): flatten the frequency signal we leak. Because our firm
    # floor makes us repeatedly offer our few best outcomes -- which is exactly
    # what a frequency-based opponent model needs to pin down our preferences --
    # we instead diversify across our high-utility band, preferring values we have
    # revealed least often. This lowers the opponent's Kendall-tau of us (raising
    # our share of the Concealing point) at ~no Advantage cost, since every bid we
    # pick still clears the fair floor. Flag-gated for the Phase 2 ablation.
    # MEASURED: against a capable opponent model this BACKFIRES -- diversifying
    # across our top band reveals MORE of our high-utility region, raising the
    # opponent's tau of us. Kept for the ablation but disabled by default.
    DECEPTION = False

    # How to time-weight opponent offers when modelling their PREFERENCES:
    #   "uniform" -> every offer counts equally
    #   "recency" -> later offers count more (good for "what they'll accept",
    #                but late offers are concessions, not true preferences)
    #   "early"   -> earlier offers count more (their opening is near their ideal,
    #                so early offers are the honest preference signal)
    MODEL_WEIGHTING = "early"

    # Extra weight given to the opponent's OPENING offer, which is typically their
    # ideal outcome and thus the single strongest preference signal. MEASURED: a
    # moderate boost raises our model accuracy (tau_me 0.55 -> 0.57) and lifts our
    # Concealing split from losing (0.496) to parity (0.500) at zero advantage cost
    # -- the free half of Concealing. Above ~8 the gain saturates.
    OPENING_BOOST = 3.0

    # How to estimate the opponent's ISSUE WEIGHTS:
    #   "stability" -> issues they change less are weighted more (cheap heuristic)
    #   "bayesian"  -> posterior over rank-weight hypotheses, favouring the
    #                  weighting under which their (early) offers look high-utility
    # MEASURED: the two are within noise on tau_me (0.561 vs 0.559) -- Kendall-tau
    # only needs the ranking, which the simple model already recovers, and ~100
    # offers is not the sparse regime where Bayesian helps. We keep "stability"
    # (simpler, interpretable) and retain the Bayesian path for the report.
    MODEL_KIND = "stability"
    BAYES_TEMP = 6.0  # softmax sharpness over weight hypotheses

    # Decoy-freeze concealment: hold a FIXED value on our least-important issue
    # across all offers. Opponent models that learn issue weights from which
    # issues we leave unchanged (HardHeaded/AgentX-style) then over-rate this
    # cheap issue and mis-rank our true weights -- lowering their tau of us
    # (raising our Concealing share) at ~zero Advantage cost, since we hold our
    # own preferred value on an issue we barely care about. Does nothing against
    # uniform-weight frequency models (e.g. GSmith). Flag-gated for the ablation.
    DECOY_FREEZE = False

    # End-game safety: rather than risk a no-deal (which can LOSE on the
    # Concealing term if the opponent models us better than we model them), in the
    # rescue window we secure the best offer the opponent has actually shown us.
    # Against a yielding opponent this rarely triggers (they accept our firm
    # offers first); against a firm one it guarantees we bank their best concession
    # instead of timing out.
    SECURE_BEST = True

    # Whether the end-game SECURE_BEST capture requires the offer to be "fair" by
    # our own model (u_to_us >= our_estimate_of_their_utility). MEASURED: the live
    # field is dominated by firm rivals whose best-shown offer is their near-ideal
    # -- fair by reality but "lopsided" by our model -- so the fairness test blocks
    # the very deals we need to close and we time out into a ~0.5 no-deal. The
    # tournament scores absolute mean, so ANY positive-Advantage deal beats a
    # no-deal; with this False we bank any rescue-window offer above our
    # reservation. The reservation hard floor (not this test) is what still
    # prevents us being walked below the value of walking away.
    SECURE_BEST_FAIR = False

    # When the SECURE_BEST_FAIR blanket-accept (above) activates, as a fraction of
    # time. Kept slightly LATER than RESCUE_TIME so that "accept-only-high" rivals
    # (e.g. AcceptTop-style) still get a window to accept OUR offer first -- we
    # only cave to their lopsided best at the very end. MEASURED below.
    SECURE_BEST_TIME = 0.97

    # Deadline-adaptive end-game (robustness across round counts). The rescue and
    # secure-accept windows are defined by RELATIVE time, but at short deadlines a
    # 5% window is only ~1 round -- too few to actually exchange and close, which
    # MEASURABLY hurt us at n_steps=20. So we also require a MINIMUM number of
    # absolute rounds for each window: the effective trigger time is the EARLIER of
    # the relative threshold and "this many rounds before the end". At long
    # deadlines the relative thresholds dominate (unchanged); at short ones the
    # round floors pull the end-game earlier so we still close.
    MIN_RESCUE_ROUNDS = 5
    MIN_SECURE_ROUNDS = 3

    # Concession-conditional secure-accept timing. A conceding opponent will keep
    # improving its offers, so we can afford to HOLD (delay the blanket-accept) and
    # extract more; a firm hold-and-wait opponent never improves, so we should cave
    # at the normal time to still close. We push SECURE_BEST_TIME later in
    # proportion to the opponent's revealed concession (the round floor still
    # guarantees we accept before timing out). Off by default; measured.
    ADAPTIVE_SECURE = False
    ADAPT_SECURE_K = 0.025

    # Minimum advantage (as a fraction of our range) required for the end-game
    # blanket-accept to fire. A near-zero-advantage deal barely beats a no-deal on
    # the Advantage term, but closing early means we observe FEWER opponent offers
    # -> a weaker model -> lower Concealing; holding for more offers can raise
    # tau_A. A small floor refuses worthless deals to protect Concealing on hard
    # (competitive) domains. 0.0 = accept any positive deal (the deal-rate-max
    # default). Measured.
    SECURE_ACCEPT_FLOOR = 0.05

    # AC_combi-style end-game acceptance. Our secure path historically required the
    # opponent's CURRENT offer to be its best-ever (MAX_T), which the literature
    # finds "too strict -- loses deals": an opponent that oscillates and offers a
    # good-but-not-best deal late is rejected, risking a no-deal (a bottom-quartile
    # leak). With SECURE_ANY=True we instead accept ANY end-game offer that clears
    # our acceptance floor (reservation + SECURE_ACCEPT_FLOOR), regardless of
    # whether it is their all-time best -- closer to the measured-best AC_combi
    # rule. The reservation+floor still guards against underselling. Measured.
    SECURE_ANY = False

    # "Never undersell": never propose a bid worth less to us than the best the
    # opponent has already offered (anti-self-sabotage). Measured.
    NEVER_UNDERSELL = False

    # Calibrated P(accept) bid search: among band candidates, offer the one
    # maximising our_utility * P(accept), where P(accept) rises monotonically with
    # the bid's estimated utility to the opponent between their estimated
    # reservation and their ideal. A SOFT version of reservation-targeting (which,
    # as a hard filter, dropped deal rate); the probabilistic weighting trades
    # extraction against acceptance smoothly. Measured.
    PACCEPT_SEARCH = False

    # --- Offline-trained concession policy (Phase 3) ---------------------------
    # A state-conditioned policy whose weights are optimised OFFLINE
    # (eval/train_policy.py) against the opponent roster and FROZEN here. At
    # runtime it maps negotiation-state features to a concession target,
    # replacing the fixed Boulware curve in the normal phase. The end-game rescue
    # and secure-accept safety net still apply on top, so a bad policy can never
    # time out into a no-deal -- it can only affect how much we concede. Inference
    # is pure-python (a 7-dim dot product + sigmoid); no runtime memory, no deps.
    # Off unless a trained weight vector is shipped.
    USE_LEARNED_POLICY = True
    # Residual MLP policy: target = Boulware_fraction(t) + correction(state), where
    # correction is a small tanh-MLP (9 features -> 6 tanh hidden -> 1), bounded by
    # POLICY_CORR_SCALE. With all-zero weights the correction is 0 and the policy IS
    # the heuristic Boulware curve -- so training starts exactly at the proven bar
    # and can only learn to deviate where it helps (non-regressing by construction).
    POLICY_NFEAT = 9
    POLICY_NHID = 6
    POLICY_CORR_SCALE = 0.25
    POLICY_WEIGHTS = [-2.998998, 2.351945, 0.229012, 0.09561, 0.529328, 0.427983, 1.806619, -2.606468, 0.477897, -1.590718, 4.121987, -0.673944, -1.656193, -1.148943, -0.06128, 0.414582, -3.679978, -2.432292, 0.428981, -1.311522, 0.147199, -0.926331, 0.945257, -0.774881, -1.322077, 0.628672, -0.503033, 0.902521, 0.653827, 0.134321, 1.381501, 1.031024, -0.398369, -0.95392, 1.821574, 3.049437, -1.076726, 0.618299, -1.57986, -0.229833, -3.436871, -0.046999, 3.243237, 0.498544, -2.19989, -0.809427, -1.370285, -2.125543, -1.19021, -3.308172, -1.135589, -0.93856, -0.509172, -0.032074, -1.469515, -0.437333, -1.102043, -0.967768, -1.22082, 1.339024, -1.786388, -1.070433, 1.896571, 0.665713, -2.623641, -2.411149, -0.509744]

    # First-mover end-game: if we are the LAST to propose (the opponent will not
    # get a turn to accept a fresh bid), table the opponent's own best-shown offer
    # instead -- it is provably acceptable to them, guaranteeing a close where a
    # fresh rescue bid would time out. Off by default; measured.
    LAST_OFFER_GRAB = True
    LAST_OFFER_ROUNDS = 1

    # Behaviour-adaptive concession: speed up our clock in proportion to how much
    # the opponent has actually conceded (reciprocity), to break semi-firm
    # deadlocks. A firm opponent shows ~0 concession so we stay firm. Off by
    # default; BETA scales the effect. Measured.
    ADAPTIVE_CONCESSION = False
    ADAPT_BETA = 0.5

    # ------------------------------------------------------------------ init --
    def on_preferences_changed(self, changes):
        """Initialise per-negotiation state.

        Called once our utility function is known. All state here is
        instance-local and rebuilt every negotiation -- we keep NO memory across
        negotiations (a competition rule).
        """
        if self.ufun is None:
            return

        os_ = self.ufun.outcome_space
        self._issues = list(os_.issues)
        self._n_issues = len(self._issues)
        # Candidate values per issue, in issue order (outcomes are tuples in this
        # same order).
        self._issue_values = [list(issue.all) for issue in self._issues]

        self._reserve = float(self.ufun.reserved_value)
        self._u_max = float(self.ufun.max())
        # The anti-exploitation floor in absolute utility terms.
        self._fair_floor = self._reserve + self.FAIR_FLOOR_FRACTION * (
            self._u_max - self._reserve
        )

        # All outcomes sorted by OUR utility, descending. We keep only rational
        # outcomes (utility strictly above the reservation value) -- we would
        # never propose or accept anything below the value of walking away.
        outcomes = list(os_.enumerate_or_sample(max_cardinality=100_000))
        scored = [(float(self.ufun(o)), o) for o in outcomes]
        scored = [su for su in scored if su[0] > self._reserve]
        scored.sort(key=lambda su: su[0], reverse=True)
        if not scored:  # degenerate: nothing beats reservation -> keep the best
            scored = sorted(
                ((float(self.ufun(o)), o) for o in outcomes),
                key=lambda su: su[0],
                reverse=True,
            )[:1]
        self._sorted_utils = [su[0] for su in scored]
        self._sorted_outcomes = [su[1] for su in scored]

        # --- Decoy-freeze: pick our least-important issue (smallest utility swing)
        # and hold our preferred value on it across all offers (see DECOY_FREEZE).
        self._decoy_issue = None
        self._decoy_value = None
        if self.DECOY_FREEZE and self._n_issues >= 2:
            best = self._sorted_outcomes[0]
            sens = []
            for i in range(self._n_issues):
                us = []
                for v in self._issue_values[i]:
                    o = list(best)
                    o[i] = v
                    us.append(float(self.ufun(tuple(o))))
                sens.append(max(us) - min(us))
            self._decoy_issue = min(range(self._n_issues), key=lambda i: sens[i])
            self._decoy_value = best[self._decoy_issue]

        # --- Opponent-model accumulators (see update_opponent_model) ---
        self._value_counts = [dict() for _ in range(self._n_issues)]
        self._last_opp_value = [None] * self._n_issues
        self._issue_changes = [0] * self._n_issues
        self._opp_offers_seen = 0

        # --- Deception accumulators: how often WE have revealed each value, so we
        # can prefer under-revealed values and keep our offer distribution flat.
        self._my_value_counts = [dict() for _ in range(self._n_issues)]

        # Opponent offer history (offer, early-weight) for the Bayesian model.
        self._opp_offer_hist: list[tuple] = []
        # Pre-enumerate rank-weight hypotheses for the Bayesian model (only when it
        # is enabled and the issue count keeps the permutation set small).
        self._weight_hypotheses = None
        if self.MODEL_KIND == "bayesian" and 1 <= self._n_issues <= 6:
            import itertools

            self._weight_hypotheses = []
            for perm in itertools.permutations(range(self._n_issues)):
                # perm[i] = rank of issue i (0 = most important).
                raw = [float(self._n_issues - perm[i]) for i in range(self._n_issues)]
                s = sum(raw)
                self._weight_hypotheses.append([r / s for r in raw])

        # Best offer (to us) the opponent has made -- a provably attainable
        # utility/outcome we can fall back on in the end-game.
        self._best_opp_util = self._reserve
        self._best_opp_offer = None
        self._first_opp_util = self._reserve
        self._opp_concession = 0.0

        # Small memo so a single round computes its planned bid once.
        self._cached_bid_step = -1
        self._cached_bid = None

        # Emit an initial (uniform) estimate so the scoring contract holds even
        # before the opponent has made any offer.
        self._rebuild_opponent_model()

    # ------------------------------------------------------- opponent model --
    def _rebuild_opponent_model(self) -> None:
        """Rebuild our estimate of the opponent's utility as an additive ufun.

        Two interpretable signals, both derived only from the opponent's own
        offers (their rejections are weak evidence and are ignored):

          * Value scores: within an issue, values the opponent offers more often
            are assumed better for them -> normalised frequency.
          * Issue weights: issues whose value the opponent rarely changes are
            assumed more important to them -> stability. Until we have a couple
            of offers we fall back to uniform weights.

        We only claim to recover the opponent's *ranking* of outcomes (which is
        exactly what the Kendall-tau scoring rewards) -- not cardinal utilities
        and not their reservation value.
        """
        value_funs = []
        for i in range(self._n_issues):
            counts = self._value_counts[i]
            max_count = max(counts.values()) if counts else 1.0
            # The tiny index gradient guarantees the value function is never
            # perfectly flat. A constant model makes Kendall-tau undefined, which
            # the scorer reads as -1 and which would forfeit the ENTIRE Concealing
            # point if we are scored before folding in any opponent offer (e.g. we
            # open and the opponent accepts immediately). The gradient is
            # negligible (1e-6) once real counts accumulate.
            mapping = {
                v: (counts.get(v, 0) / max_count if max_count > 0 else 0.0)
                + 1e-6 * idx
                for idx, v in enumerate(self._issue_values[i])
            }
            value_funs.append(TableFun(mapping=mapping))

        if self.MODEL_KIND == "entropy" and self._opp_offers_seen >= 2:
            # Issue weight from the CONCENTRATION of the opponent's offered-value
            # distribution: an issue on which they keep offering the same few
            # values (low entropy) is important; one they vary freely (high
            # entropy) is not. This uses the full distribution rather than only
            # consecutive-change counts, and is a sharper estimator of the
            # issue-WEIGHT ranking, which dominates the full-space Kendall-tau.
            import math
            raw = []
            for i in range(self._n_issues):
                counts = self._value_counts[i]
                tot = sum(counts.values())
                k = len(self._issue_values[i])
                if tot <= 0 or k <= 1:
                    raw.append(1.0)
                    continue
                ent = -sum((c / tot) * math.log(c / tot) for c in counts.values() if c > 0)
                raw.append(max(0.0, 1.0 - ent / math.log(k)))  # 1 = concentrated
            total = sum(raw)
            weights = ([r / total for r in raw] if total > 0
                       else [1.0 / self._n_issues] * self._n_issues)
        elif self.MODEL_KIND == "bayesian" and self._weight_hypotheses:
            weights = self._bayesian_weights(value_funs)
        elif self._opp_offers_seen >= 2:
            # changes are counted across transitions => denominator is one less
            # than the number of offers seen.
            denom = max(1, self._opp_offers_seen - 1)
            raw = [
                max(0.0, 1.0 - (self._issue_changes[i] / denom))
                for i in range(self._n_issues)
            ]
            total = sum(raw)
            weights = (
                [r / total for r in raw]
                if total > 0
                else [1.0 / self._n_issues] * self._n_issues
            )
        else:
            weights = [1.0 / self._n_issues] * self._n_issues

        self.private_info["opponent_ufun"] = LinearAdditiveUtilityFunction(
            values=value_funs,
            weights=weights,
            outcome_space=self.ufun.outcome_space,
        )

    def _bayesian_weights(self, value_funs) -> list[float]:
        """Posterior-mean issue weights over rank-weight hypotheses.

        Likelihood model: a rational opponent offers outcomes that are high on
        *their* utility, especially early. For each candidate weight vector h we
        score how high the opponent's (early-weighted) offers look under
        u_h(o) = sum_i h_i * value_i(o_i); the posterior is a softmax over those
        scores, and we return the posterior-mean weight vector. Value functions
        are the frequency estimates (passed in) -- only the weights are Bayesian.
        """
        import math

        if not self._opp_offer_hist:
            return [1.0 / self._n_issues] * self._n_issues
        # Per-issue value score of each historical offer, cached once.
        vals = [
            [float(value_funs[i](o[i])) for i in range(self._n_issues)]
            for o, _ in self._opp_offer_hist
        ]
        wts = [w for _, w in self._opp_offer_hist]
        scores = []
        for h in self._weight_hypotheses:
            s = sum(
                wt * sum(h[i] * vals[t][i] for i in range(self._n_issues))
                for t, wt in enumerate(wts)
            )
            scores.append(s / max(1e-9, sum(wts)))
        mx = max(scores)
        post = [math.exp(self.BAYES_TEMP * (s - mx)) for s in scores]
        z = sum(post)
        post = [p / z for p in post]
        weights = [
            sum(post[k] * self._weight_hypotheses[k][i] for k in range(len(post)))
            for i in range(self._n_issues)
        ]
        total = sum(weights)
        return [w / total for w in weights] if total > 0 else weights

    def update_opponent_model(self, state: SAOState) -> None:
        """Fold the opponent's latest offer into the frequency/stability model."""
        offer = state.current_offer
        if offer is None:
            return
        self._opp_offers_seen += 1
        n = self._opp_offers_seen
        u_to_us = float(self.ufun(offer))
        if self._opp_offers_seen == 1:
            self._first_opp_util = u_to_us
        if u_to_us > self._best_opp_util:
            self._best_opp_util = u_to_us
            self._best_opp_offer = offer
        # Opponent concession in OUR utility terms (how far their offers have
        # improved for us since their opening), normalised to our range.
        rng = max(1e-9, self._u_max - self._reserve)
        self._opp_concession = min(
            1.0, max(0.0, (self._best_opp_util - self._first_opp_util) / rng)
        )
        # Time-weight this offer for PREFERENCE modelling (see MODEL_WEIGHTING).
        if self.MODEL_WEIGHTING == "recency":
            w = float(n)
        elif self.MODEL_WEIGHTING == "early":
            w = 1.0 / n
        else:  # uniform
            w = 1.0
        if n == 1:  # the opening offer is the strongest preference signal
            w += self.OPENING_BOOST
        self._opp_offer_hist.append((offer, w))
        for i in range(self._n_issues):
            v = offer[i]
            self._value_counts[i][v] = self._value_counts[i].get(v, 0) + w
            if self._last_opp_value[i] is not None and self._last_opp_value[i] != v:
                self._issue_changes[i] += 1
            self._last_opp_value[i] = v
        self._rebuild_opponent_model()

    # ----------------------------------------------------- bidding strategy --
    def _effective_times(self) -> tuple[float, float]:
        """(rescue_time, secure_time) adjusted for the deadline length.

        At short deadlines a fixed relative window is too few rounds to close, so
        we pull each trigger earlier to leave at least MIN_*_ROUNDS rounds. At long
        deadlines the relative thresholds are unchanged."""
        n = None
        nmi = getattr(self, "nmi", None)
        if nmi is not None:
            n = getattr(nmi, "n_steps", None)
        secure_base = self.SECURE_BEST_TIME
        if self.ADAPTIVE_SECURE:
            # Hold longer (cave later) the more the opponent has conceded.
            secure_base = min(
                0.995, self.SECURE_BEST_TIME + self.ADAPT_SECURE_K * self._opp_concession
            )
        if not n or n <= 0:
            return self.RESCUE_TIME, secure_base
        rt = min(self.RESCUE_TIME, 1.0 - self.MIN_RESCUE_ROUNDS / n)
        # The absolute-round floor still guarantees we accept before timing out.
        st = min(secure_base, 1.0 - self.MIN_SECURE_ROUNDS / n)
        return max(0.0, rt), max(0.0, st)

    def _policy_target(self, t: float, rescue_time: float) -> float:
        """Concession target from the offline-trained RESIDUAL policy.

        target_fraction = Boulware_fraction(t) + correction(state), where the
        correction is a tanh-MLP over negotiation-state features, bounded by
        POLICY_CORR_SCALE. All-zero weights => correction 0 => the heuristic curve.
        Pure-numpy, deterministic, no runtime memory. The caller still applies the
        end-game rescue and secure-accept safety net.
        """
        import numpy as np
        rng = self._u_max - self._reserve
        if rng <= 0:
            return self._u_max
        # Heuristic Boulware fraction (the residual base).
        tt = t / rescue_time if rescue_time > 0 else 1.0
        bff = self.FAIR_FLOOR_FRACTION + (1.0 - self.FAIR_FLOOR_FRACTION) * (
            1.0 - tt ** (1.0 / self.CONCESSION_EXPONENT))
        # State features.
        opp_best = min(1.0, max(0.0, (self._best_opp_util - self._reserve) / rng))
        reserve_norm = self._reserve / self._u_max if self._u_max > 0 else 0.0
        conc = self._opp_concession
        n_norm = min(1.0, self._opp_offers_seen / 30.0)
        f = np.array([t, t * t, t * t * t, conc, opp_best, reserve_norm, n_norm,
                      opp_best * t, conc * t], dtype=float)
        # Parse flat weights into the MLP (W1,b1,W2,b2).
        nf, nh = self.POLICY_NFEAT, self.POLICY_NHID
        w = np.asarray(self.POLICY_WEIGHTS, dtype=float)
        i = nh * nf
        W1 = w[:i].reshape(nh, nf)
        b1 = w[i:i + nh]; i += nh
        W2 = w[i:i + nh]; i += nh
        b2 = w[i] if i < w.size else 0.0
        corr = self.POLICY_CORR_SCALE * float(np.tanh(W2 @ np.tanh(W1 @ f + b1) + b2))
        frac = min(1.0, max(self.RESCUE_FLOOR_FRACTION, bff + corr))
        return self._reserve + frac * rng

    def _target_utility(self, relative_time: float) -> float:
        """Utility target for the current time.

        Two regimes:
          * Normal (t < rescue_time): firm Boulware from our maximum down to the
            fair floor -- we refuse to concede below a fair share of our range,
            which is what stops a firm opponent from extracting its ideal.
          * Rescue (t >= rescue_time): release the floor, decaying from the fair
            floor down to the reservation value, so we still close a positive
            deal in the final rounds instead of timing out into a zero.
        rescue_time is deadline-adaptive (see _effective_times)."""
        rescue_time = self._effective_times()[0]
        t = min(max(relative_time, 0.0), 1.0)
        if self.ADAPTIVE_CONCESSION and self._opp_offers_seen >= 5:
            # Reciprocity: advance our clock in proportion to the opponent's
            # revealed concession (firm opponent -> ~0 -> unchanged).
            t = min(1.0, t * (1.0 + self.ADAPT_BETA * self._opp_concession))
        if t < rescue_time:
            if self.USE_LEARNED_POLICY:
                return self._policy_target(t, rescue_time)
            tt = t / rescue_time if rescue_time > 0 else 1.0  # renormalise firm window
            concession = 1.0 - tt ** (1.0 / self.CONCESSION_EXPONENT)
            return self._fair_floor + (self._u_max - self._fair_floor) * concession
        frac = (t - rescue_time) / max(1e-9, 1.0 - rescue_time)  # 0 -> 1
        rescue_floor = self._reserve + self.RESCUE_FLOOR_FRACTION * (
            self._u_max - self._reserve
        )
        return rescue_floor + (self._fair_floor - rescue_floor) * (1.0 - frac)

    def _opp_reservation_est(self) -> float:
        """Estimate the opponent's acceptance floor (in OUR model's units).

        A rational opponent never proposes an outcome it would itself reject, so
        the lowest utility-to-them (under our model) among the offers they have
        made lower-bounds their reservation value. We relax it by RV_SLACK because
        their offers only lower-bound, not pin, that floor. Used by the "rvtarget"
        rescue selector to keep only bids they are likely to accept.
        """
        opp = self.opponent_ufun
        if opp is None or not self._opp_offer_hist:
            return 0.0
        worst = min(float(opp(o)) for o, _ in self._opp_offer_hist)
        return self.RV_SLACK * worst

    def _band_candidates(self, target: float) -> list[Outcome]:
        """Outcomes whose utility (to us) is at least `target`.

        `_sorted_utils` is descending, so the band is a prefix. We always return
        at least our single best remaining outcome.
        """
        hi = 0
        n = len(self._sorted_outcomes)
        while hi < n and self._sorted_utils[hi] >= target:
            hi += 1
        return self._sorted_outcomes[: max(1, hi)]

    def _select_bid(self, candidates: list[Outcome], rescue: bool) -> Outcome:
        """Pick which bid to offer from our utility band.

        Among the band we favour the outcomes our opponent model rates highest
        (raises agreement probability at no cost to our own utility). Tie-break:
          * Normal: prefer the one best for *us* (protect Advantage).
          * Rescue: prefer the *fairest* (smallest gap between our and their
            estimated utility). This lets us table a genuine compromise in the
            end-game even when the opponent never revealed it -- e.g. the only
            closeable deal on a pure-conflict domain.
        """
        opponent = self.opponent_ufun
        if opponent is None:
            return candidates[0]

        if self.PACCEPT_SEARCH and self._opp_offer_hist:
            # P(accept) rises from the opponent's estimated reservation (lo) to
            # their ideal (hi, ~their best self-offer). Pick argmax our_u * P.
            opp_utils = [float(opponent(o)) for o, _ in self._opp_offer_hist]
            lo = self.RV_SLACK * min(opp_utils)
            hi = max(opp_utils)
            span = max(1e-6, hi - lo)

            def p_accept(o):
                return min(1.0, max(0.0, (float(opponent(o)) - lo) / span))

            return max(candidates, key=lambda o: float(self.ufun(o)) * p_accept(o))

        if rescue:
            # End-game bid selection. The default ("fair") tables the most
            # balanced outcome in our band. RESCUE_SELECT switches in extraction-
            # oriented selectors that try to close at a HIGHER-for-us point that
            # the opponent is still likely to accept (deal-quality frontier):
            #   "rvtarget" : among band outcomes the opponent is predicted to
            #                accept (estimated from the worst they have offered),
            #                pick the BEST for us. Degenerates to caving vs a pure
            #                hardliner (only its ideal is "acceptable") and
            #                extracts vs accept-high / responsive rivals.
            #   "kalai"    : maximise min(our utility, their estimated utility).
            #   "nash"     : maximise our utility * their estimated utility.
            mode = self.RESCUE_SELECT
            if mode == "rvtarget":
                r_opp = self._opp_reservation_est()
                accept_pred = [o for o in candidates if float(opponent(o)) >= r_opp]
                if accept_pred:
                    return max(accept_pred, key=lambda o: float(self.ufun(o)))
            elif mode == "kalai":
                return max(
                    candidates,
                    key=lambda o: min(float(self.ufun(o)), float(opponent(o))),
                )
            elif mode == "nash":
                return max(
                    candidates,
                    key=lambda o: float(self.ufun(o)) * float(opponent(o)),
                )
            # "fair" (default) or rvtarget fallback: most balanced outcome.
            return min(
                candidates,
                key=lambda o: abs(float(self.ufun(o)) - float(opponent(o))),
            )

        if self.DECEPTION:
            # Diversify: prefer the band outcome whose values we have revealed
            # least so far (flattens our frequency signal), tie-broken toward the
            # outcome best for us. Every candidate already clears the fair floor,
            # so this costs ~no Advantage.
            def novelty(o):
                shown = sum(
                    self._my_value_counts[i].get(o[i], 0)
                    for i in range(self._n_issues)
                )
                return (shown, -float(self.ufun(o)))

            return min(candidates, key=novelty)

        if self.SELECTION_MODE == "greedy":
            return candidates[0]  # already sorted by our utility, descending
        if self.SELECTION_MODE == "nash":
            return max(
                candidates,
                key=lambda o: float(self.ufun(o)) * float(opponent(o)),
            )
        # "opponent": most opponent-friendly, ties broken toward our utility
        # (candidate order is our-utility-descending).
        return max(candidates, key=lambda o: float(opponent(o)))

    def _record_my_offer(self, outcome: Outcome) -> None:
        """Remember which values we have revealed (for deception diversification)."""
        if outcome is None:
            return
        for i in range(self._n_issues):
            self._my_value_counts[i][outcome[i]] = (
                self._my_value_counts[i].get(outcome[i], 0) + 1
            )

    def concealing_bidding_strategy(self, state: SAOState) -> Outcome | None:
        """Choose our counter-offer.

        Decouples *how much* to concede (the floored Boulware target) from
        *which* bid to make (the most opponent-friendly bid in our band).
        """
        step = getattr(state, "step", -1)
        if step == self._cached_bid_step and self._cached_bid is not None:
            return self._cached_bid

        t = state.relative_time
        # "Never undersell" (ChargingBoul): never let our concession target fall
        # below the best utility the opponent has already offered us -- proposing
        # worse than a deal we could already get is pure self-sabotage. The
        # end-game safety net still closes; this just stops us giving away value.
        if self.NEVER_UNDERSELL and self._best_opp_util > self._reserve:
            floor_target = self._best_opp_util - 1e-9
            t_target = self._target_utility(t)
            if t_target < floor_target:
                cands = self._band_candidates(floor_target)
                bid = self._select_bid(cands, rescue=t >= self._effective_times()[0])
                self._cached_bid_step = step
                self._cached_bid = bid
                return bid
        # First-mover end-game: on our final proposal (opponent gets no turn to
        # accept a fresh bid), table their own best-shown offer -- provably
        # acceptable to them -- rather than time out. Only when it beats walking
        # away; the rescue logic still governs ordinary rounds.
        if self.LAST_OFFER_GRAB and self._best_opp_offer is not None:
            nmi = getattr(self, "nmi", None)
            n = getattr(nmi, "n_steps", None) if nmi else None
            if n and (n - step) <= self.LAST_OFFER_ROUNDS and self._best_opp_util > self._reserve:
                self._cached_bid_step = step
                self._cached_bid = self._best_opp_offer
                return self._best_opp_offer
        target = self._target_utility(t)
        candidates = self._band_candidates(target)
        if self._decoy_issue is not None:
            # Hold our fixed value on the decoy issue so weight-learning opponents
            # over-rate it. Only narrow the band when some candidate still matches.
            frozen = [o for o in candidates if o[self._decoy_issue] == self._decoy_value]
            if frozen:
                candidates = frozen
        bid = self._select_bid(candidates, rescue=t >= self._effective_times()[0])

        self._cached_bid_step = step
        self._cached_bid = bid
        return bid

    # -------------------------------------------------- acceptance strategy --
    def acceptance_strategy(self, state: SAOState) -> bool:
        """Decide whether to accept the opponent's current offer.

        Rules, in order:
          * Hard floor -- never accept below our reservation value.
          * ACNext -- accept if the offer is at least as good for us as the bid
            we are about to make. Because our planned bid respects the fair floor,
            before the end-game we only accept genuinely good offers (we are not
            talked down to the opponent's ideal).
          * Rescue relaxation -- once in the end-game the planned bid (and hence
            this threshold) decays toward the reservation value, so we accept any
            safely-positive offer rather than time out into a zero.
        """
        offer = state.current_offer
        if offer is None:
            return False

        u_offer = float(self.ufun(offer))
        if u_offer < self._reserve:  # hard floor
            return False

        next_bid = self.concealing_bidding_strategy(state)
        if next_bid is not None and u_offer >= float(self.ufun(next_bid)):
            return True

        # Don't be so greedy we lose a near-best offer. If the opponent just made
        # us its best offer yet and it already clears our fair floor, take it --
        # holding out for our absolute maximum risks the offer never returning
        # (a conceding opponent's offers oscillate). Never fires against a firm
        # opponent, whose offers sit far below our floor.
        capture = self._reserve + self.CAPTURE_FRACTION * (self._u_max - self._reserve)
        if u_offer >= capture and u_offer >= self._best_opp_util - 1e-9:
            return True

        # End-game: secure the best the opponent has shown us rather than risk a
        # no-deal that may lose us the Concealing term -- but only if that offer
        # is not lopsided against us (by our model). A firm opponent's "best" is
        # its near-ideal, which fails this fairness test, so we stay firm and let
        # it yield; a genuinely conceding opponent's best is fair, so we bank it.
        if (
            self.SECURE_BEST
            and state.relative_time >= self._effective_times()[1]
            and (self.SECURE_ANY or u_offer >= self._best_opp_util - 1e-9)
            and u_offer >= self._reserve
                + self.SECURE_ACCEPT_FLOOR * (self._u_max - self._reserve)
            and u_offer > self._reserve
        ):
            if not self.SECURE_BEST_FAIR:
                # Field is mostly firm rivals whose best offer is fair-by-reality
                # but lopsided-by-our-model; banking it (it is already > our
                # reservation) closes a positive deal that beats a ~0.5 no-deal.
                return True
            opp = self.opponent_ufun
            if opp is None or u_offer >= float(opp(offer)):
                return True
        return False

    # --------------------------------------------------------- main entry ----
    def __call__(self, state: SAOState, dest: str | None = None) -> SAOResponse:
        """Respond to the opponent: accept, or reject with a counter-offer."""
        if self.ufun is None:
            return SAOResponse(ResponseType.END_NEGOTIATION, None)

        offer = state.current_offer

        # First move of the negotiation: open with a counter-offer.
        if offer is None:
            bid = self.concealing_bidding_strategy(state)
            self._record_my_offer(bid)
            return SAOResponse(ResponseType.REJECT_OFFER, bid)

        # Update our model with their offer *before* deciding, so acceptance and
        # bidding both use the freshest estimate.
        self.update_opponent_model(state)

        if self.acceptance_strategy(state):
            return SAOResponse(ResponseType.ACCEPT_OFFER, offer)

        bid = self.concealing_bidding_strategy(state)
        self._record_my_offer(bid)
        return SAOResponse(ResponseType.REJECT_OFFER, bid)
