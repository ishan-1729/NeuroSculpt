> **Provenance.** Raw synthesis from the automated research-direction brainstorm of 2026-11-03 (5 generators, 10 prior-art checks, 3 judges). It is the decision record for the project objective; the binding design is the spec. In this record, "v1" and "v1.1" mean the earlier draft (`docs/archive/earlier_draft_spec.md`) and "v2" means the current spec. Its 2026 citations were found by automated search and are unverified.

# NeuroSculpt: What Is Reading the Weights Worth?

## 1. Title and mission

**Title:** *What Is Reading the Weights Worth? A ground-truth, budget-matched test of LLM weight-space repair*

**Mission.** Takehi et al. ("Sparks of Weight-Space Programming", 2 Nov 2026) report that an LLM can read raw network weights and write edits that fix the network. They did not compare against search at a matched forward-pass budget, ran no information ablations, and used one model. NeuroSculpt v2 tests that claim on small networks with planted defects whose exact repair Δ* is known. A pinned LLM gets a frozen network's weights and a metered `evaluate(patch)` API, which is its only route to data. It proposes weight patches. We measure four things:
- whether it beats the strongest search baseline at the same query budget;
- how many black-box queries its white-box access is worth (the exchange rate Q*);
- whether the advantage comes from reading the network's *computation*. The test is exact function-preserving reparameterizations, plus a defect that leaves the weight statistics unchanged;
- how much of any gain search can recover once it is handed the LLM's chosen support or edit template.

Every outcome is reported, and a negative result directly limits how the weight-space-programming claims can be read.

## 2. Why this beats the v1 objective (with trade-offs)

**Why it is better:**
- **It tests the inspiration directly.** v1 asked "structured basis vs random basis inside 8 columns of W2." That sits close to SHARPEN, RepTran, CARE and Arachne, and the review judged it narrow. v2 asks whether white-box weight access is worth anything relative to black-box search. That is the open question the Takehi post leaves, and it is one day old.
- **Every outcome is informative.** A positive Q* is a quotable headline. A null, or an advantage that disappears under reparameterization, constrains the claims of the weight-space-programming line.
- **The ground truth is exact.** On planted damage, Δ* is known. We can therefore tell a causal repair apart from a symptom patch (cos(Δ, Δ*), edit mass on the damaged entries), which v1's MNIST-C task could not do.
- **Almost all of the v1.1 scaffold is reused** (§6): the frozen model, functional patches, query counting, preservation gate, sealed test, many-draw nulls and preregistration. v1's CMA-ES and random-subspace machinery becomes the search side of the comparison.

**Honest trade-offs:**
- **Scoop risk is high.** The gap is obvious and many groups will try to fill it, so the preregistration and preprint have to be posted quickly.
- **External, nondeterministic dependency.** Results depend on which LLM is used and drift as models change. A negative result with our pinned models limits "LLMs of this capability".
- **A second cost axis.** Tokens cost money (roughly low hundreds to about 1k USD) and add a budget dimension that is not charged as queries. It is reported as a Pareto frontier instead.
- **New infrastructure,** which is the main schedule risk: an LLM harness, a metered tool sandbox, exact reparameterization code and transcript auditing.
- **It is an evaluation, not a new method.** It measures something; it does not discover a phenomenon. The "readable" 1.8k-parameter primary model is far smaller than Takehi's targets, so the claims are scoped to it, with one larger-model replication.

## 3. Central question and contribution boundary

**Central question.** Take small networks with planted defects whose exact repair Δ* is known, with all data access going through a counted `evaluate()` API. Is a pinned LLM's read access to the weights worth more than b black-box queries? Concretely:
1. Does LLM-with-weights beat the strongest preregistered search arm at a matched budget?
2. What is Q*(b)?
3. Does the advantage survive (a) withholding the weights or swapping in another seed's weights, (b) exact permutation and rescaling reparameterizations, (c) a defect that is invisible to weight statistics, and (d) removing semantic priors?
4. When search re-fits the LLM's chosen support or edit template, how much of the gain does it recover?
5. Is the advantage confined to defects written into θ0 (endogenous), or does it also appear under exogenous input shift (MNIST-C)?

**Closest prior work and the exact overlap:**

| Work | What it already covers | What it lacks |
|---|---|---|
| Takehi et al. 2026, https://rikiyat.github.io/blog/weight-space-programming/ | An LLM reads and edits weights; sweeps 0–50 forward passes for the LLM | No search baseline at matched budget, no weights-withheld or reparameterization controls, one model |
| Weight Oracles, Kabra et al., https://arxiv.org/abs/2610.07334 | A fine-tuned LLM reads small-transformer weights, simulates forwards, detects backdoors | No editing, no permutation or rescaling test, no black-box baseline at matched budget |
| EvoLLM, https://arxiv.org/abs/2402.18381 | An LLM acts as an evolution strategy over network weights from score history | Never shown the weights. This is our LLM-noW arm |
| LLM HPO warm-start, https://arxiv.org/abs/2606.21641 | Budget-matched LLM vs classical search; finds the LLM gain is a warm-start default | Opaque parameters. Our template re-fit (H6) transplants this logic |
| LLM-for-EC survey with BBOB study, https://arxiv.org/abs/2509.08269 | Equal-budget LLM vs CMA-ES; CMA-ES is strong | Black-box only |
| Evolution or Illusion?, https://arxiv.org/abs/2609.19799 | Rankings change with budget, so single-budget claims are unsafe | This is why Q* is reported as a curve over budgets |
| Prior knowledge or search?, https://arxiv.org/abs/2605.19782 | Separates prior knowledge from search through information ablations | Different domain (code optimization) |
| Le Scao & Rush, https://arxiv.org/abs/2103.08493 | "How many data points is a prompt worth": the exchange-rate framing | Q* is a new instance of this framing, cited as such |
| Goedecke critique, https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/ | Argues Takehi's wins reduce to simple data-free heuristics | We make those heuristics required baselines |
| SHARPEN https://arxiv.org/abs/2604.00422, RepTran https://arxiv.org/abs/2607.11193, CARE, Arachne | Localize-then-search repair with random-localization ablations | No LLM arm. These are our search baselines |
| Block-level weight-space structure, https://arxiv.org/abs/2609.26147 | Permutation and rescaling as function-preserving controls in weight-space analysis | Never applied to an LLM reader or editor |
| CLP data-free backdoor removal, https://arxiv.org/abs/2208.03111 | Weights-only repair works when the failure is written into θ0 | Motivates our endogenous vs exogenous contrast (H7) |

**What is new.** No prior study combines all four of: (a) an LLM given a network's weights, (b) proposing repair patches, (c) compared against CMA-ES, random search and EvoLLM-style search at a matched, metered forward budget, (d) with weights-withheld, wrong-weights and exact reparameterization controls. Specific new elements:
1. **Q*(b)**, an exchange rate between white-box access and black-box queries, measured against exact ground truth.
2. **The reparameterization test**, split into permutation-only (marginal weight statistics preserved), positive per-unit rescaling (statistics changed), and rescale-then-renormalize.
3. **A statistics-invariant "stealth" defect**: outgoing columns of hidden units are swapped, so every marginal weight statistic is unchanged but the computation is broken. Only reading what the units compute can locate it, so data-free norm or SVD heuristics fail by construction.
4. **A decomposition of any LLM gain into localization, template and value-setting**, each re-fitted by search.
5. **An endogenous vs exogenous moderator.**

**What is not claimed:** a new repair method, the exchange-rate concept itself, the idea that LLMs can read weights, or anything about the LLMs used in [R1] or LLM-scale targets.

## 4. Hypotheses

One primary hypothesis, H1. Its pre-declared primary contrast is: **LLM-W (model M1) minus the primary search comparator, at b = 64 FBE, on Task A, scored as the sealed-test fraction of recoverable gap closed.** The primary search comparator is the strongest search arm at b = 64 on calibration seeds in P1, chosen and frozen before any analysis seed is run. The unit of analysis is the (seed × damage family) cell, with LLM samples nested inside cells. Test: one-sided Wilcoxon signed-rank over cells, α = 0.05, plus a hierarchical bootstrap 95% CI. SESOI = 10 pp of recoverable gap. All secondary hypotheses use Holm correction, and equivalence claims use TOST with a ±5 pp margin.

| ID | Hypothesis | What falsifies it |
|---|---|---|
| **H1 (primary)** | At b = 64 FBE on planted damage, LLM-W closes at least 10 pp more of the recoverable gap than the strongest preregistered search arm, under the same patch language, gate and layer hint. | Median cell difference below SESOI, Wilcoxon p ≥ 0.05, or TOST equivalence/negative. |
| H2 (exchange rate) | Q*(64) > 128, i.e. weight access is worth more than one extra budget's worth of queries. Q*(b) is the search budget at which the comparator's median sealed gain reaches LLM-W's median gain at b, estimated from search curves up to 4096 FBE, with a bootstrap CI and right-censoring at 4096. | The CI of Q*(64) includes values ≤ 64. |
| H3 (weights are used) | LLM-W beats both LLM-noW (same API, score history, no weights; the EvoLLM-style arm) and LLM-wrongW (another seed's weights, patch applied to the true model) by at least SESOI. At b = 0, LLM-W beats the best data-free heuristic (norm-outlier repair, top-SVD-direction removal, z = 0). | LLM-noW or LLM-wrongW within the TOST margin of LLM-W; or LLM-W at b = 0 no better than data-free heuristics. |
| H4 (computation, not surface) | (a) LLM-W's gain is invariant to hidden-unit permutation (TOST ±5 pp). (b) It is invariant to positive rescaling, with c log-uniform in [¼, 4], and to rescale-then-renormalize. (c) On the stealth column-swap family, LLM-W still beats data-free heuristics by at least SESOI. | A drop beyond the margin under permutation means the LLM depends on presentation. A drop under rescaling but not under renormalization means it reads magnitudes. No advantage on the stealth family means it reads surface statistics, not computation. |
| H5 (reading, not priors) | Semantic stripping keeps at least 70% of LLM-W's advantage over LLM-noW. Stripping applies a secret fixed pixel permutation (folded into W1) and a class relabeling, and gives no MNIST or digit description. | The stripped advantage falls below 70% of the unstripped advantage, so digit-shape priors drive it. |
| H6 (decomposition, two-sided) | CMA-ES restricted to the LLM's chosen support, or re-fitting the LLM's edit template ("swap columns i,j", "scale column j", "bias shift on unit i") by search, recovers less than 50% of LLM-W's gain. That would mean the LLM sets values, not just locations. | Re-fit recovers at least 90%: the LLM contributes localization or a template only. This is still publishable, and is reported as its own outcome. |
| H7 (endogenous vs exogenous) | LLM-W's advantage over LLM-noW is larger on Task A (defect in θ0) than on Task B (MNIST-C input shift, where θ0 carries no information about the shift). Seed-level interaction test. | No interaction, or a larger advantage on Task B. |
| H8 (cause vs symptom) | On Task A, LLM-W edits have higher cos(Δ, Δ*) and higher repair on held-out affected classes than search edits at a matched gain. Held-out classes are affected classes deliberately left out of the adaptation pool. The search null is the distribution of 200 random-restart solutions. | LLM-W inside the random-restart null on both metrics. |

## 5. MVP experimental design

**Models.**
- **Primary, "readable" MLP:** 7×7-downsampled MNIST, 49→24→16→10, ReLU, about 1.8k parameters. Its weights fit in context as text (about 6–10k tokens at 4 significant figures). Six analysis seeds plus two calibration seeds, which are used for pilots and comparator selection and excluded from analysis.
- **Secondary (P4):** the existing 784→128→64→10 MLP. Too large to read as text, so the weights are available only inside a sandbox that holds weight arrays and no data.

**LLMs.** M1 is a pinned frontier API model and M2 a smaller or open-weight model, so capability can serve as a covariate. Model IDs, temperature and dates are fixed and logged, with full transcripts and token counts. M2 runs only the primary arms (§7).

**Tasks.**
- **Task A, planted damage in W2** (24×16), exact Δ* = −damage. Four families:
  - F1: block sign flip
  - F2: outgoing columns of 2 hidden units replaced by noise of matched column norm
  - F3: full-rank noise on an 8×8 submatrix, which raises the norm and is detectable by statistics
  - F4: **stealth** swap of two to three hidden units' outgoing W2 columns. Marginal statistics are exactly preserved, but the function is broken.

  Severity is set so that clean accuracy drops by 15–40 pp. The harness reports ‖Δ*‖/‖W2‖ (review item M11). For H8, the adaptation pool covers only half of the classes the damage affects.
- **Task B, exogenous shift:** one non-geometric MNIST-C corruption downsampled to 7×7, chosen by the v1.1 headroom pilot (M8a) using calibration data only.

**Information parity.** Every arm is told the same things: the architecture, that the defect is in W2 (Task A), the task description (absent in the stripped arm), and the same `evaluate(patch)` API. The API returns aggregate target CE and accuracy, per-class accuracy, and preservation KL on held-out pools. One call is 1 FBE. The patch language allows any additive edit to W2, b2, W3 and b3, and only queries are charged.
- The LLM's Python tool holds weights only. Any loop it writes over `evaluate` is metered automatically, because there is no other data channel.
- Forward passes on inputs the LLM synthesizes itself are allowed, since they compute over weights, not data. They are logged and analysed as a covariate.
- Each arm submits at most K = 3 final candidates. The rest follows v1.1: top-K rescoring on the selection split (charged), a one-sided paired-bootstrap gate, and `NO_IMPROVING_EDIT`.
- The Task A gate is: no per-class selection accuracy falls more than 1 pp below the damaged model's. The Task B gate is v1.1's clean-retention gate (0.5 pp).

**Arms.**
- **LLM arms** (3 samples per cell):
  - LLM-W
  - LLM-noW (EvoLLM-style)
  - LLM-wrongW
  - LLM-W-perm
  - LLM-W-rescale
  - LLM-W-renorm
  - LLM-W-blind

  Patches from the reparameterized arms are mapped back exactly, with unit tests requiring float64 allclose and identical argmax on all test inputs.
- **Search arms**, all on the same API and gate:
  - CMA-ES on the full W2 + b2 support
  - CMA-ES in a d = 8 random subspace, with a 200-draw null (the v1 B0 machinery)
  - random search
  - CMA-ES on a support localized by data-free norm outliers, with localization charged
  - CMA-ES on S1/S2-localized supports, with localization forwards and labels charged
  - **hybrids:** CMA-ES on the LLM's support, and a re-fit of the LLM's template
- **Data-free baselines:** z = 0, norm-outlier renormalization, top-SVD-direction removal.
- **Cheap labeled baselines:** logit-bias (b3) search, which is the v1 H4 class-prior control, and last-layer refit (charged).
- **Ceilings,** labeled as gradient methods: CMA-ES on the oracle support, masked Adam on W2, full fine-tuning.

**Metrics.**
- **Primary:** sealed-test fraction of recoverable gap closed, (acc_edit − acc_damaged)/(acc_clean − acc_damaged), under the gate. For Task B the denominator is the FULL-FT recoverable gap.
- **Secondary:**
  - Q*(b) curves
  - cos(Δ, Δ*), ‖Δ − Δ*‖/‖Δ*‖, and edit mass on the damaged entries
  - held-out-class repair
  - preservation KL
  - an edit-type taxonomy
  - tokens and wall-clock per session (Pareto frontier against gain; not charged as queries)
  - parse and refusal rates

**Statistics.** The unit is the (seed, family) cell: 24 cells for Task A, 6 for Task B. A hierarchical bootstrap resamples seeds, then LLM samples, then test examples. Wilcoxon tests run over cells. Equivalence claims use TOST at ±5 pp, and secondary hypotheses are Holm-corrected. Sample size is checked against pilot variance before the freeze. If the minimum detectable effect exceeds SESOI, seeds are added; this is prespecified.

**Budgets.**
- LLM arms run at b ∈ {0, 16, 64} FBE. Search curves extend to 4096 FBE, which costs seconds on CPU.
- About 800 M1 sessions: W and noW over the full Task A grid; ablations at b = 64; Task B with W, noW and perm at b ∈ {16, 64}.
- About 180 M2 sessions on the primary arms.
- At 30–60k tokens per session this is roughly 30–55M tokens, under a preregistered spending cap.
- All non-LLM compute runs on CPU.

## 6. What carries over from v1.1 and what changes

**Carries over unchanged:**
- the frozen-model discipline and functional patches (`edits.py`, with no-mutation and bitwise z = 0 tests)
- the query-counted evaluation and FBE accounting (`search.py`), now the LLM's only data channel
- CMA-ES and random search, and the 200-draw random null (C2)
- six or more seeds
- the preservation gate with a one-sided bootstrap, top-K rescoring and `NO_IMPROVING_EDIT` (M10)
- the closed-form, logit-bias, last-layer, FULL-FT and ADAM-MASK baselines and their labeling as gradient methods (M7)
- Task A, now the core task, with ‖Δ*‖/‖W_S‖ reporting and non-spectral damage families (M11)
- the headroom pilot for Task B (M8a); held-out adaptation and preservation pools; the MNIST-C index-alignment check; the labeled-examples column (M9)
- float64 deterministic evaluation
- the sealed test, a preregistration config (`configs/study_v1.yaml` becomes `study_v2.yaml`), and `research_decisions.md` logging
- the SESOI, Wilcoxon, hierarchical-bootstrap and TOST statistical template
- S1/S2 localization code, reused as query-spending localization arms

**Changes:**
- **Mission, central question and H1–H5** are replaced by §3 and §4.
- **The v1 non-goal "no LLM edit controller" is lifted, for the LLM arms only.** Every other arm remains gradient-free search.
- **The B0/B1/B2 basis factorial is demoted.** The B0 d = 8 subspace becomes one search arm, and B1/B2 and the basis-vs-basis question drop out of the primary claims.
- **The primary model becomes the readable 49-24-16-10 MLP on 7×7 MNIST.** The 784 MLP moves to a P4 sandbox replication.
- **New code:**
  - readable-model data and training
  - a damage generator with exact Δ* (including the stealth family)
  - `reparam.py` (permutation, rescaling, renormalization, exact patch inverse)
  - an LLM harness: prompt templates, metered API client, weights-only tool sandbox, transcript and token logging, patch parser
  - a Q* estimator
  - semantic stripping
- **Task B shrinks** from the primary result to the exogenous arm of the H7 moderator.

## 7. Staged plan with exit gates

**P0, harness (days 1–4).** Train the readable MLP (8 seeds). Build the damage generator, reparameterization module, metered API, transcript logging and data-free heuristics.
*Exit gate:*
- z = 0 is bitwise identical to the frozen model
- applying Δ* restores clean accuracy exactly
- reparameterization round-trips pass (float64 allclose ≤ 1e-10, identical argmax on every test input)
- the readable model reaches at least 90% clean accuracy
- an API call cannot reach data except through the meter (an adversarial test script tries)

**P1, search-only headroom and comparator selection (days 4–7).** Run search curves to 4096 FBE and all data-free and ceiling arms on the calibration seeds. Run the Task B headroom pilot.
*Exit gate:*
- at b = 64, the best search arm closes ≤ 70% of the gap on at least 3 of 4 families; otherwise raise severity
- data-free heuristics close ≤ 20% of the gap on F4
- the primary search comparator and the Task B corruption are frozen

**P2, LLM pilot and freeze (week 2).** About 30 sessions on calibration seeds only: Task A, LLM-W and LLM-noW at b = 0 and b = 64, on M1 and M2.
*Exit gate:*
- parse rate ≥ 90%
- token cost per session measured, and the spending cap set
- variance-based power check
- the sandbox audit finds no unmetered data path
- `study_v2.yaml` frozen and **the preregistration posted publicly with a timestamp**, as a defence against being scooped

Analysis seeds are untouched until this point.

**P3, main grid (week 3).** Run all Task A arms on M1, then Task B, then the M2 primary arms. Open the sealed test once, after all cells are complete.
*Exit gate:* the primary analysis runs exactly as preregistered, and deviations are logged in `research_decisions.md`.

**P4, decomposition, replication and write-up (week 4).**
- the H6 support and template re-fits (cheap, searching over LLM outputs)
- the 784-MLP sandbox replication on two damage families
- optional: an activation-probe LLM arm without weights (MAIA-style)
- the preprint

*Exit gate:* every preregistered hypothesis is reported with its outcome, regardless of sign.

## 8. Top risks and the fallback result

| Risk | Mitigation |
|---|---|
| Scooped within weeks | Post the preregistration in P2, keep the MVP to the readable MLP, and put out a preprint in week 4 |
| LLM nondeterminism and model drift | Pin IDs and dates, 3 samples per cell nested in the analysis, sample variance treated as data |
| Search-in-disguise or harness leakage | The meter is the only data channel; LLM-written optimizer loops are charged automatically; synthetic-input forwards are logged as a covariate |
| A "reading" win is really a rediscovered heuristic | Data-free heuristics are required baselines; the stealth family F4; the H6 template re-fit |
| Task A too easy for search (small Q*) or too hard for everyone | P1 headroom gate on severity; full-rank and stealth families |
| Priors or contamination (MNIST knowledge) | Semantic stripping (H5), wrongW |
| The readable model does not represent real networks | 784-MLP sandbox replication, claims explicitly scoped |
| Token cost over budget | Preregistered cap; ablations only at b = 64; M2 only on primary arms |

**The result that stays publishable if H1 fails:** "With exact ground truth and a matched, metered budget, white-box weight access for pinned LLMs of capability X is worth fewer than b queries (Q* ≤ b) relative to CMA-ES." This would be accompanied by:
- a decomposition saying whether any partial gain is localization or template (recoverable by search) or nothing;
- the H4 and H7 outcomes, which may show reading of surface statistics, or value only for endogenous defects;
- a reusable benchmark: planted-damage suite with exact Δ*, reparameterization tests, the stealth family and a metered harness, against which future weight-programming claims can be checked.

A positive H1 together with a failed H4 is also a clear finding: the LLM reads magnitudes, not computation.

## 9. Runner-up directions and why they lost

- **Localization exchange rate (contrarian-1, 18.5).** The most confound-resistant design, with G-matched placebo nulls and planted-locality instrument checks. But Wang & Veitch (https://arxiv.org/abs/2502.11447), Hase et al. and a 2026 code-repair placebo study already make its central claim, and its G covariate is nearly tautological. *Grafted:* localization cost charged to search arms, and planted locality as a positive control.
- **Closed-form vs search across an information ladder (practical-shift-2, 18).** Cheap and actionable, but its H1 is arithmetic (CMA-ES cannot beat the exact optimum of the same quadratic), and H4 and H5 replicate Mintun et al., Santurkar et al. and the label-shift literature. It reads as a careful benchmark, not a contribution. *Grafted:* the closed-form, labeled and charged baselines.
- **Weights-only repair subspace and LOCO ceiling (theory-geometry-2, 17.5).** The most feasible proposal, but its theory is Ky Fan, and GOLD (https://arxiv.org/abs/2603.21928) and Robust Weight Signatures partly answer it. *Grafted:* the endogenous vs exogenous scope (via CLP), which became H7.
- **Mechanism or patch (llm-weight-programming-2, 17).** The deepest question, but it carries the heaviest infrastructure (Tracr/InterpBench, a MAIA arm, four defect families) and MAIA and Mechanistic Unlearning pre-empt parts of it. *Grafted:* cos(Δ, Δ*) and held-out-class repair (H8); the MAIA-style arm is optional in P4.
- **Repairability frontier (theory-geometry-1, 17).** Standard results assembled (PRDNN, REASSURE, EMMET, Elenberg), the rank result is NP-hard where the Jacobian varies per input, and it drops the weight-programming hook. Its region-crossing gap is the one original piece, and too niche to anchor the project.
- **Others.**
  - mechanistic-2 (16): a scope jump to transformers, with a hard-to-beat baseline arena.
  - mechanistic-1 (15.5): circular predictor-oracle agreement, essentially a restatement of Hase et al. 2023.
  - contrarian-2 (15): the headline is algebraically forced at W2.
  - practical-shift-1 (14): pre-empted by Labiosa et al. (https://arxiv.org/abs/2608.03842), and its divergence rule is degenerate for input corruptions.