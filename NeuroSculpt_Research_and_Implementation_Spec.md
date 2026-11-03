# NeuroSculpt

**What Is Reading the Weights Worth? A Ground-Truth, Budget-Matched Test of LLM Weight-Space Repair**  
**Research and Implementation Specification — v1.0**  
**Prepared:** 3 November 2026  
**Status:** Proposed research design; not an implemented or experimentally validated system  
**Audience:** Research scientist / frontier coding agent reviewing, revising, and implementing the project

> **Mission.** Takehi et al. report that a large language model (LLM) can read raw neural-network weights and write edits that repair the network [R1]. NeuroSculpt tests that claim under conditions where it can fail cleanly. The networks are small, their planted defects have an exactly known repair \(\Delta^*\), and the data is reachable only through a metered query API. We measure whether white-box weight access helps a pinned LLM beat the strongest black-box search at the same query budget, and how many queries that access is worth. We then test whether any advantage comes from reading the network's *computation*, or only from cheap weight statistics, priors, free compute or localization hints that a search or a simple heuristic could exploit as well.

> **Interpretation rule.** Every numerical budget and acceptance threshold in this document is a *proposed experimental-design choice*, not a reported result. The implementing agent may revise choices with a written rationale; it must not present hypotheses as findings.

> **Background.** An earlier draft asked whether structured low-dimensional edit bases beat random bases for gradient-free repair. Its review (`docs/spec_review.md`) found that question narrow and close to existing search-based repair work, and a judged brainstorm (`docs/research_direction_brainstorm.md`) selected the present question instead. No code was written for the earlier draft; it is kept for reference in `docs/archive/earlier_draft_spec.md`. This spec reuses its reviewed methodology and has itself been through one adversarial review, whose accepted fixes are logged in `docs/research_decisions.md`.

## 1. Executive specification

### 1.1 Central question

Take small networks with planted defects whose exact repair \(\Delta^*\) is known, with all data reachable only through a counted `evaluate` API. **Is a pinned LLM's read access to the weights worth more than \(b\) black-box queries?** Concretely:

1. Does an LLM given the weights beat the strongest preregistered black-box arm at a matched query budget?
2. What is the exchange rate \(Q^*(b)\): the black-box budget needed to match the LLM's gain at budget \(b\)?
3. Does the advantage survive four controls: (a) withholding the weights, or substituting a decoy network's weights; (b) exact function-preserving reparameterizations; (c) a stealth defect that W2-only weight statistics cannot locate, sampled so that cross-layer balance heuristics close at most 20% of its gap; (d) removing semantic priors?
4. When search is handed the LLM's chosen support or edit template, how much of the LLM's gain does it recover?
5. Descriptively: does the advantage differ between defects written into \(\theta_0\) (Task A) and exogenous input shift (Task B)?

\(Q^*\) measures the value of white-box access in **metered-data** terms only. It says nothing about compute or dollar efficiency: an LLM session costs orders of magnitude more compute than CMA-ES at 4,096 queries [R20], and both costs are reported.

### 1.2 Non-goals

- Do **not** claim to invent model repair, weight-space learning, LLM-driven optimization, or the exchange-rate framing [R12]. Do not claim that LLMs can read weights in general.
- Do **not** reproduce [R1]'s targets or make claims about the LLM it used. Every claim is scoped to the pinned LLMs and the small target network used here; a negative result is conditional on the pinned models' capability.
- Do **not** fine-tune an LLM, build a general-purpose interpretability agent, or depend on LLM-scale target networks.
- Do **not** give any arm a data channel that bypasses the meter, and do not let Claude Code (or any agent building the harness) inspect candidates or pick edits from evaluation feedback.
- Do **not** require a cloud deployment, authentication system or web UI.

### 1.3 Minimum viable research product (MVRP)

1. A "readable" target MLP whose full weights fit in an LLM context as text, with reproducible training, splits and checkpoints.
2. A planted-damage generator with exact \(\Delta^*\), uniformly sampled damage sites and a balance-matched stealth defect.
3. A metered evaluation API that is the only route to data for every arm, inside a sandbox with an adversarial leakage test.
4. Exact reparameterization code with a proven-exact patch inverse, and canonicalized repair metrics.
5. An LLM harness: versioned prompts, the sandbox, billed-token and transcript logging, a patch parser and a deterministic template extractor.
6. Black-box search arms (including a discrete template search), data-free heuristics, cheap labeled baselines and gradient ceilings under the same API, selection rule and gate.
7. The per-cell \(Q^*\) estimator, preregistered statistics, results tables, plots and a technical report that states every outcome regardless of sign.

### 1.4 Default scope (adjustable only with justification)

| Item | Default | Why |
|---|---|---|
| Target model | MLP 49 → 24 → 16 → 10, ReLU, on MNIST area-averaged to 7×7 (`avg_pool2d`, kernel 4, on [0,1] images); 1,770 parameters | Full weights fit in context as text (~10–16k tokens in the fixed number format of §6.1) |
| Defect site | \(W_2\in\mathbb R^{16\times24}\), with natural exact repair \(\Delta^*=-D\) | Ground-truth repair, so causal repair can be told apart from symptom patching |
| Patch language | Any additive edit to \(W_2, b_2, W_3, b_3\) | Identical for every arm |
| LLMs | M1: a pinned frontier API model. M2: a smaller or open-weight model | Capability as a covariate; model ID, effort/thinking settings, API version and dates pinned (§6.1) |
| Query budgets | LLM arms at \(b\in\{0,16,64\}\) FBE; search curves on \(q\in\{1,2,4,\dots,4096\}\) | Search is CPU-cheap, so \(Q^*\) can be estimated as a curve |
| Seeds | 10 analysis seeds + 3 calibration seeds | The seed is the inferential unit (§7.2); 10 seeds give a minimum exact one-sided p of 1/1024 |
| LLM samples | 3 sessions per cell for primary arms, 2 for ablation arms | LLM variance is treated as data |
| Random null | 200 random restarts for every stochastic search arm | A null distribution, not a single comparator |
| Hardware | CPU for all non-LLM compute; one local sandbox for every LLM arm | Target model is tiny |
| Sessions and cost | ~1,380 sessions (§8.2); billed tokens and dollars measured in P2 and capped | Tokens are a second cost axis, reported as a Pareto frontier, not charged as queries |
| Timeline | ~6 weeks for one researcher plus a coding agent | The harness is most of the work (§10) |

## 2. Scientific context and contribution boundary

### 2.1 Inspiration and its critics

Takehi et al., *Sparks of Weight-Space Programming* (2 November 2026), study one LLM (reported as GPT-6 Astra) reading, writing and fixing about twenty target networks [R1]. Its repair cases are a Kinetix actor fixed in one reply with no forward passes, and a Qwen3.5-0.8B fix using 10/25/50 forward passes plus practice data; a Gemma repair failed. The 0–50 forward-pass sweep belongs to a write-from-scratch Go task, not a repair. Search is banned in some experiments, and no search baseline is reported. There are no weights-withheld or reparameterization controls, although read tasks include an input/output-layer and label-stripping ablation. R1's one-shot repair setting corresponds to \(b=0\) here, and its stripping ablation is the precedent for H5.

A public critique argues that the tasks are within reach of a competent researcher using simple methods, such as weight-norm heatmaps for identification and contrastive activation steering for the Qwen fix [R11]. An independent partial replication finds that turning thinking off, or removing a mechanically computed activation direction, nearly matches R1's Qwen edit [R25]. NeuroSculpt makes such simple methods required baselines.

### 2.2 Prior work the project must acknowledge

**Citation status.** Entries marked † were checked against their sources only by automated agents on 3 November 2026. Confirm each by hand before citing it in any write-up (§13, step 1). The novelty claim in §2.3 is dated 3 November 2026; re-run the literature search at the P2 freeze.

| Work | What it already covers | What it lacks |
|---|---|---|
| Takehi et al. 2026† [R1] | One LLM reads, writes and fixes many networks | No matched-budget search baseline; no weights-withheld or reparameterization controls |
| Weight Oracles† [R2] | A fine-tuned LLM reads small-transformer weights and detects backdoors; statistical detectors degrade out of distribution | No editing; no matched black-box baseline |
| Learning to Interpret Weight Differences (DIT)† [R24] | LLMs trained to describe weight differences | No repair; no budget-matched comparison |
| EvoLLM [R3] | An LLM acts as an evolution-strategy operator over discretized candidate parameter vectors and their fitness; discretized encodings beat raw floats; smaller LLMs sometimes win | Optimizes from scratch, with no trained network to diagnose. LLM-noW is EvoLLM-inspired, with \(\theta_0\) withheld |
| AgenticBBO-Bench† [R26] | LLM agents beat numerical optimizers in 4 of 5 domains under a finite budget | Black-box only; motivates keeping LLM-noW strong |
| Rodrigues et al.† [R4] | Budget-matched LLM vs classical HPO; the LLM gain is a warm-start default | Opaque parameters. Our template re-fit (H6) applies the same logic |
| Zhang et al. survey† [R5] | LLMs for evolutionary optimization; BBOB comparison against classical optimizers; LLMs degrade with dimension | Black-box only |
| Oved et al.† [R6, R20] | Rankings among LLM evolutionary strategies change with budget; comparisons should be matched by tokens, not proposals | Motivates per-budget \(Q^*\) curves and the explicit cost caveat of §1.1 |
| Redko et al.† [R7] | Separates prior knowledge from search through information ablations; LLMs act as greedy optimizers in pure black-box settings | Different domain (code optimization) |
| Weight-space symmetries [R8, R21, R22] | Positive rescaling and permutation are exact function-preserving symmetries of ReLU networks | Never used as controls for an LLM reader or editor (cf. R1, R2) |
| Balanced layers [R23] | Gradient training approximately conserves per-unit incoming-minus-outgoing norm | Implies a data-free cue for the stealth swap; addressed in §4.4 and §5.1 |
| CLP data-free backdoor removal [R9] | Weights-only repair works when the failure is written into \(\theta_0\) | Motivates the Task A vs Task B description |
| SHARPEN† [R13], RepTran† [R14], CARE [R15], Arachne [R16] | Localize-then-search repair, with random-localization ablations | No LLM arm. They motivate our localize-then-search arms (CMA-NORM, CMA-PCA, CMA-S1/S2) |
| Le Scao & Rush [R12] | "How many data points is a prompt worth?": the exchange-rate framing | \(Q^*\) is a new instance of this framing, cited as such |
| Hase et al. [R17] | Causal localization need not predict where an edit works | Motivates the template re-fit and charged localization |

### 2.3 Precise candidate contribution

As of 3 November 2026, no prior study combines all four of: (a) an LLM given a network's weights, (b) proposing repair patches, (c) compared against CMA-ES, discrete template search, random search and an LLM-without-weights arm at a matched, metered query budget, (d) with withheld-weights, decoy-weights and exact reparameterization controls. The new elements are:

1. **\(Q^*(b)\)**, a per-cell exchange rate between white-box access and black-box queries, measured against exact ground truth, in metered-data terms only.
2. **Reparameterization controls**: every LLM-W session sees a fresh random permutation of both hidden layers; separate arms see random positive rescaling and a canonical row-normalized form.
3. **A stealth defect** (F4) that W2-only statistics cannot locate, with swap sets sampled so that cross-layer balance heuristics close at most 20% of its gap (P1 gate).
4. **A decomposition of any LLM gain** into localization and edit template, each re-fitted by search (the template extracted automatically), with value setting as the residual.
5. **A descriptive endogenous vs exogenous comparison** (Task A vs Task B).

**Not claimed:** a new repair method, the exchange-rate concept, the idea that LLMs can read weights, compute or dollar efficiency, or anything about LLM-scale target networks. **A rigorous negative result is a full result**: "for pinned LLMs of capability X, white-box access is worth fewer than \(b\) metered queries relative to the best black-box arm" limits how weight-space-programming claims can be read.

## 3. Research questions and falsifiable hypotheses

**Primary contrast (pre-declared).** LLM-W on model M1 minus the primary comparator, at \(b=64\) FBE, on Task A, scored as the sealed-test fraction of recoverable gap closed \(R\) (§4.1). The primary comparator is the strongest **pure black-box** arm (§6.2) at \(b=64\), chosen by selection-split \(R\) on the 3 calibration seeds in P1 and frozen before any analysis seed is run.

**Inference unit and tests.** The unit is the **analysis seed** (10). For each seed, average \(R\) over the 4 damage families and the LLM samples (and over restarts for search). The primary test is an exact one-sided Wilcoxon signed-rank test over seeds of the paired difference, α = 0.05, with a family-stratified seed-level bootstrap BCa 95% CI. A mixed model (family fixed; seed and seed × family random; samples nested) is a preregistered sensitivity analysis. The smallest effect size of interest (SESOI) is 10 pp of recoverable gap.

**Confirmatory secondary family** (Holm, \(m=6\)): H2, H3a, H3b, H3c, H4b, H5. Each contributes exactly one p-value. H2 uses a one-sided seed-cluster bootstrap p-value for \(Q^*(64)>128\). H3c and H4b each use the larger of their two component p-values (intersection–union). H5 is gatekept on H3a: it is tested at its Holm level only if H3a is rejected, and otherwise counts as not rejected, so \(m\) never changes. **Estimation-only** (effect and CI, no confirmatory test): H4a, H6, H7, H8. Equivalence claims are made only where the P2 power check shows a ±5 pp TOST is attainable; otherwise they are reported as estimates.

| ID | Hypothesis | Supported if | Falsified or not supported if |
|---|---|---|---|
| **H1 (primary)** | At \(b=64\) on Task A, LLM-W closes at least 10 pp more of the recoverable gap than the primary comparator, under the same disclosure, patch language, selection rule and gate. | Wilcoxon \(p<0.05\) and the median seed difference ≥ SESOI. | Otherwise; a TOST-equivalent or negative difference is reported as such. |
| H2 (exchange rate) | \(Q^*(64)>128\) against the per-budget black-box envelope (§4.5). | The one-sided bootstrap p-value for \(Q^*(64)>128\) passes its Holm level. | Otherwise. Separately report whether the one-sided 95% lower bound exceeds 64 ("worth anything"). |
| H3a (weights are used) | LLM-W beats LLM-noW at \(b=64\) by ≥ SESOI. | Holm-adjusted test passes and effect ≥ SESOI. | Otherwise. |
| H3b (true weights) | LLM-W beats LLM-wrongW at \(b=64\) by ≥ SESOI. | As H3a. | Otherwise. |
| H3c (beyond heuristics and free compute) | At \(b=0\), LLM-W beats the best data-free heuristic, including BALANCE-MATCH; and LLM-W-nosynth beats LLM-noW at \(b=64\). | Both pass (one Holm member, max p). | Either fails. |
| H4a (rescaling) | LLM-W-rescale's gain is within ±5 pp of LLM-W's. | Estimate and CI reported; TOST only if powered. | — |
| H4b (computation, not statistics or trial swaps) | On balance-matched F4: at \(b=64\), LLM-W beats the best of TEMPLATE-SEARCH, ELIGIBLE-ENUM and LLM-noW at the same budget by ≥ SESOI; and at \(b=0\), it beats BALANCE-MATCH, NORM-RENORM and SVD-REMOVE by ≥ SESOI. | Both pass (one Holm member, max p). | Otherwise: the LLM's F4 gains are explained by metered trial swaps or by statistics a heuristic can read. |
| H5 (reading, not priors) | With semantics stripped, weights still help: LLM-W-strip beats LLM-noW-strip by ≥ SESOI. Gatekept on H3a. | Holm-adjusted test passes. | Otherwise. The loss of advantage from stripping, \((W-\text{noW})-(W\text{-strip}-\text{noW-strip})\), is reported as an estimate and attributed to priors *plus* synthetic proxy data, since the pixel permutation also blocks synthesizing digit-like inputs. |
| H6 (decomposition) | Search re-fits of the LLM's support (HYB-SUPPORT) and template (HYB-TEMPLATE), each with a fresh 64 FBE, recover a fraction \(\phi\) of LLM-W's gain. | Reported in bands: \(\phi<0.5\) "LLM sets values"; \(0.5\le\phi<0.9\) "partial"; \(\phi\ge0.9\) "localization or template only". | — |
| H7 (Task A vs B) | The LLM-W minus LLM-noW advantage differs between Task A and Task B. | Descriptive: both differences in absolute pp and in \(R\), with CIs. | — |
| H8 (cause vs symptom) | At matched gain, LLM-W edits are closer to the true repair than black-box edits and generalize better to held-out affected classes. | Estimated: the per-seed fraction of cells where LLM-W exceeds the gain-matched search distribution's 95th percentile on the canonical cosine, on \(h_2\) agreement and on held-out-class repair (§7.1), with CIs. | — |

**Pre-registration.** Freeze the arm list, LLM settings, prompt templates, number format, seed lists, splits, damage families and severities, disclosure text, budgets, the primary comparator and per-budget envelope, metric definitions, session table, caps and the analysis plan in `configs/study.yaml` before any analysis seed is run. Post the preregistration publicly with a timestamp in P2. Later deviations are logged in `docs/research_decisions.md`.

## 4. Formal setup

### 4.1 Notation and recoverable gap

The target MLP has \(h_1=\mathrm{ReLU}(W_1x+b_1)\), \(h_2=\mathrm{ReLU}(W_2h_1+b_2)\), \(f=W_3h_2+b_3\). The clean model is \(\theta_c\). A planted defect replaces \(W_2\) by \(W_2+D\), giving \(\theta_0\), the damaged model handed to every arm.

**Affected classes.** A class is *affected* if its calibration accuracy drops by at least 5 pp under the damage. Every cell needs at least 2 affected classes; otherwise its sites are resampled. Of \(n\) affected classes, \(\lceil n/2\rceil\) drawn at random are *in-pool* (present in the adaptation batch), and the rest (at least 1) are *held out* for the H8 generalization estimate.

For a patch \(\Delta\), the **fraction of recoverable gap closed** on a split is

\[
R(\Delta)=\frac{A(\theta_0+\Delta)-A(\theta_0)}{A(\theta_c)-A(\theta_0)},
\]

with accuracy \(A\) on the **in-pool affected classes**. Compute numerator and denominator jointly within each bootstrap resample. Report \(R\) unclipped, always alongside absolute pp. If a cell's sealed-test denominator is below 5 pp, flag it; the primary analysis keeps flagged cells, and a sensitivity analysis drops them. Values above 1 are flagged together with any change in predicted-class frequency.

**Task B** has \(\theta_0=\theta_c\), so its denominator is the gap to the in-language ceiling ADAM-PATCH (§6.3), computed once per seed, with the same 5 pp floor.

### 4.2 Disclosure, information parity and the metered API

**Disclosure (identical for every arm).** The architecture; that the defect is in \(W_2\) (Task A); the family library: "the defect is one of: sign flips of a set of entries; replacement of some columns by noise; additive noise on a submatrix; a permutation of some columns". Which family applies is never disclosed. The task description is identical except in the stripped arms. The prompts state that the API reports per-class accuracy on the target pool.

```python
evaluate(patch) -> {"target_ce", "target_acc", "per_class_acc", "preserve_kl"}  # 1 FBE
evaluate_batch(patches) -> list                                              # 1 FBE per patch
forward(x) -> logits                       # synthetic inputs only; capped and logged (below)
```

- One `evaluate` scores the fixed adaptation batch once and costs **1 FBE**. The batch holds the in-pool affected classes as the target pool and a preservation pool of the other classes.
- `per_class_acc` covers the target pool, so together with the pool's composition it reveals which classes are damaged and in-pool. Every arm receives it, and the grey-box arm CMA-PCA (§6.2) exploits it, so the hint is not exclusive to the LLM.
- **Synthetic compute.** LLM-W-family arms get a code tool in the sandbox, holding their own presented weight view, with a preregistered CPU-time budget per session (default 10 CPU-minutes). That budget, not a count of calls, bounds any forward passes they compute. `forward(x)` evaluates the arm's own presented view, unpatched (the decoy for LLM-wrongW). This free weight-only compute is part of LLM-W, and **H1 is explicitly a claim about LLM-W including it**. LLM-W-nosynth has no code tool and no `forward`: its weights are in the prompt only, and it calls `evaluate` as a plain tool (H3c).
- **Patch language.** Any additive edit to \(W_2, b_2, W_3, b_3\), plus harness-applied **index-only operators** on \(W_2\) that need no weight values: permute a column set (pairs or cycles), flip the signs of a row set × column set, zero a column set. Every arm may use them, so template repairs are expressible without reading \(\theta_0\). Out-of-language patches are rejected without charge.
- Black-box search arms minimize

\[
J(\Delta)=L_T(\theta_0+\Delta)+\lambda\,K_P(\theta_0,\theta_0+\Delta),\qquad K_P=\mathbb E_{x\sim\mathcal D_P}D_{\mathrm{KL}}\big(p_{\theta_0}(\cdot\mid x)\,\|\,p_{\theta_0+\Delta}(\cdot\mid x)\big),
\]

with \(L_T\) the target cross-entropy and \(\lambda\) tuned on calibration seeds only.

### 4.3 Function-preserving reparameterizations (PROVEN, exact)

For a hidden layer with incoming \((W_{\text{in}},b_{\text{in}})\) and outgoing \(W_{\text{out}}\):

- **Permutation.** Replacing \((W_{\text{in}},b_{\text{in}},W_{\text{out}})\) by \((W_{\text{in}}[\pi,:],\,b_{\text{in}}[\pi],\,W_{\text{out}}[:,\pi])\) leaves the function unchanged for any permutation \(\pi\), since ReLU acts elementwise.
- **Positive rescaling.** Replacing them by \((CW_{\text{in}},Cb_{\text{in}},W_{\text{out}}C^{-1})\), \(C=\operatorname{diag}(c)\), \(c_i>0\), leaves the function unchanged, since \(\mathrm{ReLU}(c\,t)=c\,\mathrm{ReLU}(t)\).
- **Canonical form.** Rescaling with \(c_i=1/\|(W_{\text{in}}[i,:],b_{\text{in}}[i])\|\) gives every incoming row (including its bias) unit norm. With two hidden layers, canonicalize layer 0 first, then layer 1 on the already-transformed parameters. In that order, rescaling both layers first by any random \(c\) and then canonicalizing gives the same result, so the canonical form is deterministic; the reverse order is not. Near-zero rows (norm below \(10^{-8}\)) are left unscaled.

**Presentation.** Every LLM-W-family session is presented under a fresh random permutation of **both** hidden layers. LLM-W-rescale additionally applies independent random \(c\) on both hidden layers, drawn log-uniform in \([1/4,4]\). LLM-W-canon applies the canonical form to both hidden layers. The harness builds each transform once from \(\theta_0\), stores it with the session, and maps patches back with the stored inverse. Every transform here is linear in the parameters once its \(\pi\) and \(c\) are fixed, so \(f(T\theta_0+\Delta')=f(\theta_0+T^{-1}\Delta')\) exactly. Unit tests require `allclose(rtol=0, atol=1e-10)` on logits and identical argmax on real test inputs, for the forward map and the patch inverse, on both hidden layers and on damaged models from every family.

Because damage-site sampling is permutation-equivariant (§5.1) and so is training, a fixed and a fresh permutation are equal in distribution. A permutation arm would therefore test only presentation-order variance, so there is none; fresh permutations remove any index-order cue instead.

### 4.4 Stealth defect and non-uniqueness of the repair (PROVEN)

**Equivalence.** Family F4 permutes the outgoing \(W_2\) columns of 2–3 first-layer units, \(W_2\mapsto W_2[:,\pi]\). Every W2-only statistic is unchanged: each row's multiset of entries and the full multiset are preserved. Since \(W_2[:,\pi]\,h=W_2\,h[\pi^{-1}]\), the damaged network computes exactly what the clean \(W_2\) computes with incoming rows \(W_1[\pi^{-1},:]\), \(b_1[\pi^{-1}]\). The defect is a mismatch between what units compute and where their outputs are sent. **When the permuted units are drawn uniformly, W2-only statistics cannot locate it.**

**Balance cue.** Gradient training approximately conserves \(\|w_{\text{in},i}\|^2+b_i^2-\|W_2[:,i]\|^2\) per unit [R23], so incoming and outgoing norms are strongly correlated unit by unit. A review pilot on a comparable 7×7 MLP measured a log-norm correlation of 0.985–0.999. A swap therefore breaks the balance of the swapped units, and a data-free heuristic (BALANCE-MATCH, §6.3) can find it. Three consequences follow:

1. F4 swap sets are drawn uniformly from the *eligible* sets: sets whose units match **pairwise** in both incoming log-norm (bias included) and outgoing log-norm, \(|\Delta\log\|\cdot\||\le\tau\) with \(\tau\) preregistered. The number of eligible sets is reported per seed and must be at least 20.
2. Restricting to eligible sets makes enumerating them a partial locator. The grey-box arm ELIGIBLE-ENUM (§6.2) therefore tries eligible sets through the meter, and H4b compares LLM-W against it. P1 verifies that BALANCE-MATCH closes ≤ 20% of the F4 gap on calibration seeds.
3. Reading balance or eligibility is legitimate white-box use and counts toward H1. H4b is what asks for more than that.

**Non-uniqueness.** The patch language contains every incoming and outgoing parameter of the 16 \(h_2\) units. For any positive \(c\in\mathbb R^{16}\) and permutation \(P\), the patch \((PCW_2^{c}-W_2^{0},\,(PC-I)b_2,\,W_3C^{-1}P^\top-W_3,\,0)\) also restores \(\theta_c\)'s function exactly, and directions through dead units or flat activation directions change nothing on the data. So \(\Delta^*=-D\) is one member of a continuum of exact repairs. \(R\) is the primary metric, and cause-vs-symptom metrics are computed only after canonicalization (§7.1).

### 4.5 Exchange rate \(Q^*(b)\)

- **Per-budget envelope.** For each \(q\) on the grid \(\{1,2,4,\dots,4096\}\), the envelope arm is the pure black-box arm with the highest calibration-seed selection-split \(R\) at that \(q\), frozen in P1. \(g_S^{(c)}(q)\) is that arm's sealed-test \(R\) in analysis cell \(c\), averaged over restarts. \(g_S^{(c)}(0)=0\).
- **Per-cell estimate.** Isotonize \(g_S^{(c)}\) with PAVA on the log₂ grid, interpolate linearly in \(\log_2 q\), and set \(Q^{*}_c(b)=\min\{q:g_S^{(c)}(q)\ge\bar R^{(c)}_{\text{LLM}}(b)\}\), with \(\bar R^{(c)}_{\text{LLM}}\) the cell's mean over LLM samples. If the curve never reaches it, \(Q^*_c\) is right-censored at 4,096 and flagged.
- **Aggregation.** Pool all cells and estimate the Kaplan–Meier median of \(Q^*_c\), with a seed-cluster bootstrap (BCa) CI; the H2 p-value comes from the same bootstrap. Also report per-family KM medians, the fraction of cells with \(Q^*_c>b\) and the censoring fraction.
- \(Q^*\) is also reported against the single primary comparator, as a secondary view.

### 4.6 Candidate selection and the gate

- **Selection ledger.** Every arm submits **exactly** \(K=3\) final candidates: LLM arms their own choice (padded with \(\Delta=0\) if fewer); search arms their top 3 by \(J\), or unevaluated proposals when \(q<3\); data-free heuristics their top 3 variants. Candidates are rescored on the selection split through a **separate selection ledger, identical for every arm and not counted in \(b\)**. This makes \(b=0\) arms and data-free baselines well defined without giving any arm a larger free best-of-K.
- **Task A gate.** The candidate passes if the one-sided 95% paired-bootstrap upper bound on the drop in **pooled unaffected-class accuracy**, relative to \(\theta_0\), is at most 1 pp. The selection split has at least 500 examples per class. P1 requires \(\Delta^*\) and CMA-ORACLE to pass in ≥ 95% of calibration cells; otherwise the gate is recalibrated on \(\Delta^*\) before freezing.
- **Task B gate.** Clean-retention drop at most 0.5 pp, with the same bound.
- **Choice.** Among passing candidates, take the highest raw selection-split accuracy on in-pool affected classes (tie-breaker: smallest \(\|\Delta\|_F\)). If none beats \(\Delta=0\) by a preregistered margin in raw pp, keep \(\theta_0\) and record `NO_IMPROVING_EDIT`. Selection never uses \(\theta_c\).
- **Intent to treat.** Parse failures, refusals, cap-truncated and infrastructure-failed sessions yield \(\theta_0\) (\(R=0\)) and enter every analysis. Failure rates are reported per arm.

## 5. Tasks and data protocol

### 5.1 Task A: planted damage (primary)

Damage is planted in \(W_2\in\mathbb R^{16\times24}\). **Damaged row and column index sets are sampled uniformly at random, regardless of contiguity; F4 samples uniformly within its eligible sets (§4.4). Both rules are permutation-equivariant.**

| Family | Damage | Visible to W2-only statistics? |
|---|---|---|
| F1 | Sign flip of the entries in a random row set × random column set | Partly |
| F2 | Outgoing columns of 2 random first-layer units replaced by noise with matched column norm | Mostly no (norms matched) |
| F3 | Full-rank noise on a random 8×8 row × column set | Yes (raises the norm) |
| F4 | **Stealth** permutation of 2–3 balance-matched units' outgoing columns (§4.4) | No, by construction |

- **Severity.** F1–F3 are set to target a 25 pp drop in affected-class accuracy on calibration seeds. F4 has no continuous severity knob, so it rejection-samples eligible swap sets whose calibration drop falls within 15–40 pp. The harness reports \(\|\Delta^*\|_F/\|W_2\|_F\).
- **P1 checks** (calibration seeds; black-box arms, heuristics and ceilings only; no LLM result is ever used):
  1. **Reported, never acted on:** the best pure black-box arm's gap closed at \(b=64\). A value above 70% or below 15% is a finding, not a reason to change severity.
  2. **Identifiability bound:** on at least 3 of 4 families, the best pure black-box arm closes at least 80% of the gap by 4,096 FBE. This is the only check that may move severity, within 15–40 pp, by a preregistered rule.
  3. BALANCE-MATCH closes ≤ 20% of the F4 gap, and every seed has at least 20 eligible F4 sets.
  4. \(\Delta^*\) and CMA-ORACLE pass the gate (§4.6) in ≥ 95% of calibration cells.
- **Ground truth** (\(D\), \(\pi\), \(\Delta^*\), \(\theta_c\)) is visible only to the harness. The damage generator stores \(\pi\) and \(\Delta^*=W_2^{c}-W_2^{0}\) and asserts that \(\theta_0+\Delta^*\) reproduces \(\theta_c\)'s logits.

### 5.2 Task B: exogenous shift (descriptive)

One non-geometric MNIST-C corruption [R10], area-averaged to 7×7 exactly as in §1.4, chosen by a headroom pilot on calibration data (drop of roughly 5–30 pp, stable across seeds). \(\theta_0=\theta_c\), and the denominator is the ADAM-PATCH ceiling. Only LLM-W and LLM-noW run, at \(b=64\). MNIST-C's authors note that fine-tuning on one corruption trivially recovers accuracy, so Task B is few-shot adaptation using MNIST-C images, not a robustness benchmark evaluation.

### 5.3 Splits and leakage protection

| Split | Used for | Forbidden use |
|---|---|---|
| `base_train` (50k MNIST-train IDs) | Learn \(\theta_c\) | Any overlap with the pools below |
| `calibration` (4,000 of the 10k held-out IDs, in two disjoint halves of 2,000) | Half A: damage severity and affected-class rule. Half B: \(\lambda\) tuning, CMA-S1/S2 localization, LASTLAYER | Final method ranking |
| `adaptation` (1,000 held-out IDs, behind the API) | All `evaluate` calls (the 256 target + 512 preservation batch is drawn from it) | Direct access by any arm |
| `selection` (5,000 held-out IDs, ≥ 500 per class) | The selection ledger and the gate | Repeated trial-and-error on the sealed test |
| `sealed_test` (official 10k MNIST test split) | Final metrics, scored once | Any optimization, selection or stopping |

- Split by **original image ID**; record SHA-256 hashes of every index list.
- **Pairing audit (Task B).** Take clean views from MNIST-C's `identity/` folder; assert label and pixel equality against raw MNIST and run a per-index correlation check; regenerate the corruption if a check fails.
- **Labeled-data arms.** CMA-S1/S2 and LASTLAYER read labeled calibration data outside the meter. They are reported, with "labeled examples used" as its own column, but they are **ineligible** as the primary comparator or the \(Q^*\) envelope.
- **Licensing.** MNIST-C data (Zenodo record 3239543) is CC BY 4.0 and requires attribution; pin the download by SHA-256. MNIST's license is commonly cited as CC BY-SA 3.0 but is not stated at the original source; do not redistribute images.

## 6. Arms

### 6.1 LLM arms and harness

| Arm | Sees | Samples | Purpose |
|---|---|---|---|
| LLM-W | True weights (fresh permutation) + API + code tool with CPU-time budget and `forward` on its own view | 3 | The claim under test |
| LLM-noW | API and its own patch/score history; \(\theta_0\) withheld (EvoLLM-inspired) [R3]; index-only operators available | 3 | Is weight access used? (H3a) |
| LLM-wrongW | A decoy: another analysis seed's \(\theta_0\) with the same family planted at independently drawn sites, freshly permuted. Code tool and `forward` act on the decoy only; patches are applied to the true model | 2 | True weights vs generic priors (H3b) |
| LLM-W-nosynth | As LLM-W, with weights in the prompt only: no code tool and no `forward` | 2 | Free weight-only compute (H3c) |
| LLM-W-rescale | As LLM-W, with random positive rescaling of both hidden layers | 2 | Magnitude reading (H4a) |
| LLM-W-canon | As LLM-W, in canonical row-normalized form | 2 | Diagnostic: absolute scale and balance removed (§4.4) |
| LLM-W-strip | As LLM-W, with a secret fixed pixel permutation folded into \(W_1\), class relabeling, and no MNIST or digit description | 2 | Semantic priors (H5) |
| LLM-noW-strip | As LLM-noW, with the same stripping | 2 | Stripped baseline for H5 |

**Interpretation of the presentation arms.** Random rescaling and the canonical form both erase the per-unit balance cue. Rescaling also randomizes absolute magnitudes, while the canonical form fixes them. A drop under both relative to LLM-W suggests the LLM uses magnitude or balance cues. A drop under rescaling only suggests sensitivity to unusual magnitudes. On balance-matched F4 the balance cue is already removed, so these arms are read on F1–F3.

**Harness requirements.**

- **Pinned settings.** Model ID, explicit effort or thinking settings (and thinking display, archived where available), `max_tokens`, API version and beta headers, and dates. Sampling parameters that a provider rejects (for example temperature on current frontier APIs) are omitted; sampling variance comes from provider defaults and is captured by the per-cell samples. Thinking summaries are not raw reasoning.
- **Number format.** One fixed format for every arm and presentation: 4 significant digits per entry, in a compact per-row-exponent layout. Per-entry precision then does not change under rescaling, so an H4a drop cannot come from quantization. Measure token counts with the pinned model's tokenizer in P0.
- **Templates.** Ablation arms share the LLM-W prompt text verbatim except for the one manipulation they test. Every template is versioned in `prompts/`.
- **Caps.** Identical per-session caps on turns (40), output tokens and context for every LLM arm. Use prompt caching with a frozen prefix (weights + prompt) and verify cache reads. Stops at a cap are logged.
- **Sandbox.** One local sandbox (a container or WSL2) used identically by every LLM arm and model: no network, read-only filesystem, CPU/time/memory limits. It holds only the arm's weight view, enforces the CPU-time budget on code execution, and exposes only `evaluate`, `evaluate_batch` and (where allowed) `forward` as RPCs to a host-side meter. The adversarial leakage test targets that RPC surface: builtins, introspection, other RPC methods, oversized and out-of-language patches.
- **Output.** At most \(K=3\) final candidates, plus a structured declaration of the intended template (logged; not used for the primary analysis).
- **Run order and failures.** Sessions run in a randomized order interleaved across arms, tasks and models, with concurrency ≥ 4. Infrastructure failures (rate limits, 5xx, timeouts past SDK retries) resume from the last successful turn with the meter ledger unchanged. A session is restarted only if it has made no `evaluate` call. After 3 failed resumptions it counts as failed (\(R=0\)). No session is re-run because of its outcome.

### 6.2 Black-box arms (same API, disclosure, selection ledger and gate)

**Pure black-box arms** never read \(\theta_0\). They are eligible as the primary comparator and for the \(Q^*\) envelope:

| Arm | What it does |
|---|---|
| CMA-FULL | sep-CMA-ES on the full \(W_2+b_2\) support (400 coordinates), starting at \(\Delta=0\) |
| CMA-SUB | CMA-ES in a \(d=8\) random subspace of \(W_2\), 200-restart null |
| RANDOM | Random search in a bounded ball |
| TEMPLATE-SEARCH | Tries instances of the index-only operator library (§4.2) in a preregistered random order at 1 FBE each, keeps improving ones and composes them greedily, then spends any remaining budget on CMA-FULL refinement. At small \(q\) it covers only part of the library (column-pair swaps alone number 276); that coverage limit is part of what black-box search costs |
| LOGIT-BIAS | CMA-ES on \(b_3\) (class-prior control) |

**Grey-box arms** read weight statistics. They are reported, ineligible as the comparator, and LLM-W is also compared descriptively with the best of them:

| Arm | What it does |
|---|---|
| CMA-NORM | CMA-ES on columns localized by data-free norm outliers |
| CMA-PCA | CMA-ES on the \(W_2\) rows of the \(h_2\) units with the largest \(|W_3[k,:]|\) over the target-pool classes \(k\) |
| TEMPLATE-STAT | TEMPLATE-SEARCH with instances ordered by weight statistics (norm imbalance, BALANCE-MATCH residuals) instead of at random |
| ELIGIBLE-ENUM | Tries the eligible F4 sets of §4.4 through the meter in a preregistered random order (H4b) |

**Labeled-data arms** (reported, ineligible as comparator): **CMA-S1 / CMA-S2**, CMA-ES on supports localized by activation contrast or causal restoration on calibration half B.

**Operator library** (shared by TEMPLATE-SEARCH, TEMPLATE-STAT and the extractor; it matches the disclosed families): column-set permutations (pairs and 3-cycles); row-set × column-set sign flips; column-set zeroing. F3 (additive noise) has no exact template.

H6 hybrids (each with a fresh 64 FBE; also reported at equal total queries):

| Arm | What it does |
|---|---|
| HYB-SUPPORT | CMA-ES restricted to the support of LLM-W's selected patch |
| HYB-TEMPLATE | Search re-fit of the template extracted from LLM-W's patch; for a "dense" patch, H6 uses HYB-SUPPORT only in that cell |

**Template extractor.** A deterministic least-squares fit of each patch against the operator library, with a preregistered residual threshold; otherwise the patch is labeled "dense". It is built and unit-tested in P0b, its threshold is frozen in P2, and it runs automatically, so no person or coding agent reads patches.

**Small budgets.** For \(q\ge16\), the CMA population size is \(\max(4,\lfloor q/4\rfloor)\), with \(\sigma_0\) from calibration. Budgets \(q<16\) run as RANDOM for curve purposes.

### 6.3 Baselines and ceilings

| Code | Kind | Queries | Role |
|---|---|---|---|
| ZERO | No edit | 0 | Floor |
| NORM-RENORM | Data-free: renormalize outlier-norm columns | 0 | Data-free heuristic (H3c, H4b) |
| SVD-REMOVE | Data-free: remove the top singular direction of the anomalous component | 0 | Data-free heuristic (H3c, H4b) |
| BALANCE-MATCH | Data-free: per-unit residuals of \(\log(\|w_{\text{in}}\|^2+b^2)-\log\|W_2[:,i]\|^2\), then linear-assignment re-pairing of W2 columns | 0 | Cross-layer heuristic (H3c, H4b; F4 gate) |
| LASTLAYER | Closed-form refit of \(W_3\) on calibration half B | labeled | Cheap labeled baseline |
| CMA-ORACLE | CMA-ES on the true damaged support | metered | Ceiling (uses ground truth; harness-only) |
| ADAM-PATCH | Adam on \(W_2,b_2,W_3,b_3\) with labeled data, fixed epochs | — | Gradient ceiling inside the patch language; Task B denominator |
| FULL-FT | Adam on all parameters | — | Gradient ceiling |

## 7. Metrics and statistics

### 7.1 Metrics

- **Primary:** sealed-test \(R\) on in-pool affected classes (§4.1), after selection and gating, with absolute pp alongside.
- **Exchange rate:** per-cell \(Q^*_c(b)\) and its aggregate (§4.5).
- **Cause vs symptom (H8), canonicalized.** Before comparing \(\Delta\) with \(\Delta^*\):
  1. map \(\theta_0+\Delta\) to the nearest member of the \(h_2\) symmetry orbit of \(\theta_c\) (closed-form per-unit scale, Hungarian matching over permutations);
  2. project out directions that are flat on calibration inputs (dead units and the null space of the \(h_1\) activation covariance below a preregistered eigenvalue threshold).

  Report the canonical cosine with \(\Delta^*\), the canonical relative distance, and a function-space metric: \(R^2\) of \(h_2\) pre-activations against \(\theta_c\)'s on sealed-test inputs. Report repair on held-out affected classes as well.
- **Gain matching for H8.** Primary: for each cell, compare LLM-W with black-box restarts whose selection-split gain is within ±5 pp of LLM-W's. Sensitivity: the envelope arm at the grid \(q\) nearest \(Q^*_c(64)\), excluding censored cells.
- **Secondary:**
  - preservation KL and per-class regressions;
  - extracted template labels;
  - billed tokens (uncached input, cache writes, cache reads, output, thinking), turns and wall-clock per session, plus a dollar estimate, as Pareto frontiers against gain;
  - parse, refusal, cap and failure rates by arm;
  - synthetic forwards used.

### 7.2 Statistical protocol

- **Unit and test.** The seed (10 analysis seeds) is the inferential unit. Use exact Wilcoxon signed-rank tests on per-seed means, family-stratified seed-level BCa bootstrap CIs, and Holm correction across the confirmatory secondary family (§3). The mixed model is a sensitivity analysis.
- **Power check before freeze.** Using P2 pilot variance on the 3 calibration seeds (and P1 search variance), compute the minimum detectable effect for H1, H3a and H4b, using the upper 80% confidence bound of the pilot SD. If it exceeds the SESOI, add up to 4 analysis seeds by a preregistered rule; their sessions are already inside the cap. Equivalence (TOST ±5 pp) claims are made only where this check shows them attainable.
- **Nulls and selection.** Stochastic search arms report their 200-restart null, and LLM-W's percentile in it is reported per cell. Never select the best seed, family, prompt or LLM using sealed-test numbers.

## 8. Budgets

### 8.1 Query budget

- LLM arms run at the budgets in the session table; black-box curves run on \(q\in\{1,2,4,\dots,4096\}\), seconds of CPU per run.
- Localization queries and every hybrid's search count in \(b\). The selection ledger is separate and identical for every arm (§4.6). A partial forward counts as one query.

### 8.2 Session table and token budget

| Model | Task | Arms | Budgets (FBE) | Cells | Samples | Sessions |
|---|---|---|---|---|---|---|
| M1 | A | LLM-W | 0, 16, 64 | 40 | 3 | 360 |
| M1 | A | LLM-noW | 16, 64 | 40 | 3 | 240 |
| M1 | A | wrongW, nosynth, rescale, canon, W-strip, noW-strip | 64 | 40 | 2 | 480 |
| M1 | B | LLM-W, LLM-noW | 64 | 10 | 3 | 60 |
| M2 | A | LLM-W, LLM-noW | 64 | 40 | 3 | 240 |
| | | | | | **Total** | **1,380** |

Up to 4 extra seeds (§7.2) add at most 552 sessions. Their cost is reserved inside the cap.

- **Tokens are billed tokens.** A tool-using session re-sends its context every turn, so billed input greatly exceeds the unique-token count. Prompt caching of the frozen prefix and `evaluate_batch` keep this down. Log per-turn usage from the API.
- **Cost estimate.** The P2 pilot measures billed tokens and dollars per session per template. The spending cap is that measurement times the session table above (including reserved seeds) plus a 25% margin, and is preregistered before P3. Expect an order of magnitude of a few thousand USD; the harness refuses to start sessions past the cap.
- **Keys.** API keys are supplied by the researcher through environment variables and are never committed or logged.

## 9. Implementation architecture

### 9.1 Repository layout

```text
src/neurosculpt/
├── models.py        # readable MLP sizes (49, 24, 16, 10); checkpoints with hashes
├── data.py          # avg_pool2d 7x7 downsampling; splits base_train/calibration/adaptation/selection/sealed_test
├── edits.py         # functional patches, no mutation, bitwise z = 0
├── damage.py        # F1-F4 generators, uniform site sampling, balance-matched swaps, exact Delta*
├── reparam.py       # permutation, rescaling, canonical form, exact patch inverse
├── meter.py         # evaluate / evaluate_batch / forward RPC surface, cost and selection ledgers
├── sandbox/         # container or WSL2 definition and RPC bridge
├── llm_harness.py   # pinned settings, prompts, caching, caps, transcripts, billed tokens, parser, resume
├── templates.py     # index-only operator library, TEMPLATE-SEARCH/-STAT, ELIGIBLE-ENUM, template extractor
├── search.py        # CMA-FULL/SUB/NORM/PCA/S1/S2, RANDOM, LOGIT-BIAS, hybrids, budget-dependent CMA settings
├── localization.py  # activation-contrast (S1) and causal-restoration (S2) localization
├── exchange.py      # per-cell Q*(b): PAVA, log2 interpolation, censoring, KM median, bootstrap
├── metrics.py       # R, canonicalized repair metrics, h2 agreement
└── objectives.py, experiment.py, plotting.py
prompts/             # versioned prompt templates per arm
configs/study.yaml, configs/smoke.yaml
```

### 9.2 Numerics and invariants

All non-LLM evaluation runs on CPU in float64, with deterministic algorithms and a fixed thread count. Required unit tests:

- \(\Delta=0\) reproduces the damaged model's logits bitwise.
- \(\theta_0+\Delta^*\) reproduces \(\theta_c\)'s logits for every family, including 3-cycles in F4.
- Reparameterization forward maps and patch inverses are exact (§4.3): rtol = 0, atol = 1e-10, identical argmax, both hidden layers, damaged models.
- The base checkpoint is never mutated.
- The adversarial leakage test passes against the RPC surface.
- Every `evaluate` call is charged exactly once, and the selection ledger is identical across arms.
- Splits are disjoint by original ID.
- The template extractor recovers planted templates on synthetic patches.

### 9.3 Output schema per run

One row per candidate and one per final selection, containing:

- **Provenance:** study ID, config hash, checkpoint hash, split hashes, seed, family, task, arm, presentation transform ID.
- **Session:** LLM settings, session index, budget, queries used, selection-ledger queries, synthetic forwards, billed tokens by type, turns, wall-clock, dollar estimate.
- **Patch and scores:** the patch (sparse), extracted template, \(R\) and absolute pp on search/selection/test, canonical cosine and distance, \(h_2\) agreement, per-class accuracies, preservation KL.
- **Outcome:** gate outcome, `NO_IMPROVING_EDIT` flag, labeled examples used, stop, parse-failure or infrastructure-failure reason.

Transcripts are stored separately and linked by session ID.

## 10. Staged plan

| Stage | Build | Exit gate |
|---|---|---|
| P0a — non-LLM harness (week 1) | Readable MLP (13 seeds); data and splits; damage generator; `reparam.py`; meter and ledgers; data-free heuristics | \(\Delta=0\) bitwise; \(\Delta^*\) restores \(\theta_c\); reparameterization exactness; clean accuracy ≥ 90% per seed; F4 balance matching works |
| P0b — sandbox and LLM harness (week 2) | Sandbox and RPC bridge; LLM harness with caching, caps, resume and concurrency; template extractor; mock-LLM smoke run | Adversarial leakage test passes; mock sessions run end to end; extractor tests pass |
| P1 — black-box headroom (week 3, first half) | All black-box curves, baselines and ceilings on calibration seeds; Task B headroom pilot | §5.1 P1 checks pass (identifiability bound; BALANCE-MATCH ≤ 20% on F4 and ≥ 20 eligible sets per seed; \(\Delta^*\) and CMA-ORACLE pass the gate in ≥ 95% of cells); primary comparator and envelope frozen |
| P2 — LLM pilot and freeze (week 3, second half) | 2–3 sessions per distinct prompt template on calibration seeds, for M1 and M2 | Parse rate ≥ 90% per template; billed cost per template measured and cap set; power check done; `study.yaml` frozen; preregistration posted publicly with timestamp; literature search re-run |
| P3 — main grid (weeks 4–5) | The session table in randomized interleaved order; sealed test opened once at the end | Primary analysis runs exactly as preregistered; deviations logged |
| P4 — decomposition and write-up (week 6) | H6 hybrids; H8 canonicalized metrics; preprint | Every preregistered hypothesis reported with its outcome, regardless of sign |

Analysis seeds are untouched until the P2 freeze. A replication on the 784 → 128 → 64 → 10 MLP (weights offered only through the sandbox) is an **UNTESTED extension** and would need its own preregistration. **Stop rule:** if a stage repeatedly fails, isolate it with a synthetic fixture and simplify the design; a sound negative result beats an impressive but invalid one.

## 11. Risks, confounders and mitigation

| Risk | Why it matters | Mitigation |
|---|---|---|
| Scooped within weeks | The gap is obvious | Post the preregistration in P2; keep the MVP to the readable MLP |
| LLM nondeterminism and model drift | Results depend on the pinned LLM | Pin settings and dates; multiple samples per cell nested in the analysis; scope claims to the pinned models |
| LLM exploits discrete templates that search cannot propose | Inflates H1 | Disclosed family library and index-only operators for every arm; TEMPLATE-SEARCH eligible as comparator |
| Black-box arms secretly read weights | The comparison would not be black-box | Pure black-box and grey-box arms separated; only pure arms are comparator-eligible |
| Free synthetic compute is search in disguise | Inflates H1 | Sandbox CPU-time budget; H1 stated as including it; LLM-W-nosynth without a code tool (H3c) |
| `per_class_acc` is a localization hint | Unequal exploitation | Given to every arm; CMA-PCA exploits it |
| Index-order or contiguity cues | A text reader sees patterns that search cannot use | Permutation-equivariant site sampling; fresh permutation for every LLM-W session |
| Stealth defect readable from norm balance or eligibility | H4b would test a statistic, not computation | Balance-matched swaps; BALANCE-MATCH and ELIGIBLE-ENUM baselines; ≥ 20 eligible sets per seed |
| A "reading" win is a rediscovered heuristic | Overclaims understanding | Data-free heuristics as baselines; H4b; H6 template re-fit |
| Gate rejects even the exact repair | `NO_IMPROVING_EDIT` everywhere | Pooled unaffected-class gate; P1 check on \(\Delta^*\) and CMA-ORACLE; ≥ 500 selection examples per class |
| Underpowered or pseudo-replicated inference | False confidence | Seed as unit; 10 seeds; power check with priced seed addition |
| Survivorship bias from failed sessions | Inflates LLM arms | Intent to treat (\(R=0\)); randomized interleaved order; infrastructure-failure policy |
| Token cost far above unique-token estimates | Budget overrun | Billed-token accounting; caching; batched evaluate; cap from the P2 measurement |
| Query-matched comparison read as efficiency | Misleading headline | §1.1 caveat; tokens, wall-clock and dollars reported [R20] |
| Unverified or misdescribed citations | Novelty claims could be wrong | Hand-verify every † reference; re-run the search at the P2 freeze |

## 12. Scientific integrity rules

- Ground truth (\(D\), \(\pi\), \(\Delta^*\), \(\theta_c\)), sealed-test data and labels are used only where the declared protocol permits.
- LLM arms run as automated, preregistered API sessions. **Neither Claude Code nor any agent building the harness may inspect candidates, steer sessions, or pick edits from evaluation feedback.** Templates are extracted by a frozen deterministic extractor. Claude Code writes and debugs code only.
- Every arm reaches data only through the metered API, except the labeled-data arms, which are labeled as such and ineligible as comparators.
- No session is re-run or discarded because of its outcome. Infrastructure failures follow §6.1, and failed sessions score \(R=0\).
- Gradient methods appear only as labeled ceilings.
- No result may be called state of the art without a fair benchmark against relevant established methods.
- Write-ups separate **PROVEN** facts (§4.3, §4.4), **OBSERVED** results, **HYPOTHESIZED** mechanisms and **UNTESTED** extensions, and include seeds, split hashes, prompts, model settings and all commands needed to reproduce tables and plots.

## 13. Instructions to the implementing agent

1. **Literature verification.** Confirm every † reference by hand: existence, authors, venue and what it actually claims. Complete RepTran's metadata. Narrow §2.3 if any verified work already covers it, and re-run the search at the P2 freeze.
2. **Mathematical audit.** Re-derive §4.3–§4.5. Check the affected-class rule, the denominator floor, the gate and the \(Q^*\) estimator's censoring.
3. **Harness audit.** Build the meter and sandbox so no data path bypasses the meter; write the adversarial leakage test first.
4. **Implement stage by stage** with the exit gates of §10. Record decisions in `docs/research_decisions.md`.
5. **Evaluate without cherry-picking.** Freeze choices before analysis seeds; open the sealed test once; report every hypothesis with its outcome.

### Questions to answer before implementing

1. Does the readable MLP reach ≥ 90% clean accuracy with area averaging, and how many tokens do its weights take in the fixed number format?
2. Does the P1 headroom gate hold with severity inside the band, including the lower bounds that make \(Q^*\) identifiable?
3. Can at least 20 balance-matched F4 swap sets be found for every seed at the preregistered threshold?
4. Do \(\Delta^*\) and CMA-ORACLE pass the gate in ≥ 95% of calibration cells?
5. What billed cost per session does P2 measure, and is the cap affordable?
6. Which comparison falsifies H1, and what precise claim remains valid if it fails?

## 14. Final deliverable checklist

- [ ] `README.md` with goal, scope and honest limitations.
- [ ] Dependency-locked environment; one-command smoke run with a mock LLM and no API calls.
- [ ] Checkpoints with hashes; deterministic, audited splits.
- [ ] Damage generator with exact \(\Delta^*\), uniform sites and balance-matched F4; reparameterization module with exactness tests.
- [ ] Sandbox with a passing adversarial leakage test; LLM harness with versioned prompts, billed-token logging and full transcripts; template extractor.
- [ ] Every arm of §6 run under the same API, disclosure, selection ledger and gate; 200-restart search nulls.
- [ ] Preregistration (`configs/study.yaml`) frozen and posted publicly before the main grid.
- [ ] A results table of \(R\), absolute pp, \(Q^*\), canonicalized repair metrics, held-out-class repair, KL, queries, billed tokens, dollars and wall-clock, including failed sessions.
- [ ] \(Q^*\) curves, Pareto frontiers (gain vs queries, gain vs tokens and dollars) and per-family plots.
- [ ] `docs/findings.md` separating PROVEN, OBSERVED, HYPOTHESIZED and UNTESTED.
- [ ] A clear answer to H1–H8, including negative answers.

### Suggested final report title

**What Is Reading the Weights Worth? A Ground-Truth, Budget-Matched Test of LLM Weight-Space Repair**

---

## References and implementation resources

† = checked against the source only by automated agents on 3 November 2026; confirm by hand before citing.

**[R1]**† R. Takehi, J. Teoh, A. Zhang, O. Khattab. *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*. Blog post, 2 November 2026. https://rikiyat.github.io/blog/weight-space-programming/

**[R2]**† A. Kabra, C. Venhoff, C. Schroeder de Witt. *Weight Oracles: Reading Neural Network Weights with Language Models*. arXiv:2610.07334, October 2026.

**[R3]** R. T. Lange, Y. Tian, Y. Tang. *Large Language Models As Evolution Strategies* (EvoLLM). arXiv:2402.18381.

**[R4]**† Rodrigues, Vas, DCosta, Prabhakaran. *When Is an LLM Worth It for Hyperparameter Optimization? A Budget-Matched Study on Tabular Data*. arXiv:2606.21641, June 2026.

**[R5]**† Zhang, Cheng, Yi, Tan. *A Systematic Survey on Large Language Models for Evolutionary Optimization: From Modeling to Solving*. IEEE Computational Intelligence Magazine; arXiv:2509.08269 (v1 2025, v7 2026).

**[R6]**† Oved, Pony, Naparstek, Barzelay. *Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search*. arXiv:2609.19799, September 2026.

**[R7]**† Redko, Fazlyev, Sozykin, Ivanova, Burnaev, Shvetsov et al. *Prior Knowledge or Search? A Study of LLM Agents in Hardware-Aware Code Optimization*. arXiv:2605.19782.

**[R8]** B. Neyshabur, R. Salakhutdinov, N. Srebro. *Path-SGD: Path-Normalized Optimization in Deep Neural Networks*. NeurIPS 2015, arXiv:1506.02617; and L. Dinh, R. Pascanu, S. Bengio, Y. Bengio. *Sharp Minima Can Generalize for Deep Nets*. ICML 2017, arXiv:1703.04933.

**[R9]** R. Zheng, R. Tang, J. Li, L. Liu. *Data-Free Backdoor Removal Based on Channel Lipschitzness* (CLP). ECCV 2022. arXiv:2208.03111

**[R10]** N. Mu, J. Gilmer. *MNIST-C: A Robustness Benchmark for Computer Vision*. arXiv:1906.02337. Data: Zenodo record 3239543 (CC BY 4.0).

**[R11]**† S. Goedecke. *LLMs are not superhuman at reading and writing neural network weights*. Blog post, 3 November 2026. https://www.seangoedecke.com/llms-are-not-superhuman-at-reading-and-writing-neural-network-weights/

**[R12]** T. Le Scao, A. M. Rush. *How Many Data Points is a Prompt Worth?* NAACL 2021. arXiv:2103.08493

**[R13]**† X. Sun et al. *Shapley-Guided Neural Repair Approach via Derivative-Free Optimization* (SHARPEN). arXiv:2604.00422, April 2026.

**[R14]**† *RepTran*. ISSRE 2026. arXiv:2607.11193. *(Title and authors still to be confirmed.)*

**[R15]** B. Sun, J. Sun, L. H. Pham, J. Shi. *Causality-Based Neural Network Repair* (CARE). ICSE 2022.

**[R16]** J. Sohn, S. Kang, S. Yoo. *Arachne: Search-Based Repair of Deep Neural Networks*. ACM TOSEM 32(4), article 85, 2023. doi:10.1145/3563210

**[R17]** P. Hase et al. *Does Localization Inform Editing?* NeurIPS 2023.

**[R18]** N. Hansen. *The CMA Evolution Strategy: A Tutorial*. arXiv:1604.00772; *pycma* documentation, https://cma-es.github.io/apidocs-pycma/index.html

**[R19]** PyTorch documentation. `torch.func.functional_call`. https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html

**[R20]**† Oved, Pony, Naparstek, Barzelay. *LLM Evolutionary Search Strategies Should be Compared by Tokens*. arXiv:2610.23977, October 2026.

**[R21]** S. K. Ainsworth, J. Hayase, S. Srinivasa. *Git Re-Basin: Merging Models modulo Permutation Symmetries*. ICLR 2023. arXiv:2209.04836

**[R22]** A. Navon et al. *Equivariant Architectures for Learning in Deep Weight Spaces* (DWSNets). ICML 2023. arXiv:2301.12780

**[R23]** S. S. Du, W. Hu, J. D. Lee. *Algorithmic Regularization in Learning Deep Homogeneous Models: Layers are Automatically Balanced*. NeurIPS 2018. arXiv:1806.00900

**[R24]**† Goel, Kim, Shavit, Wang. *Learning to Interpret Weight Differences in Language Models* (DIT). arXiv:2510.05092.

**[R25]**† *OpenWSP*: independent partial replication of [R1]. https://github.com/openwsp/openwsp

**[R26]**† Chen et al. *AgenticBBO-Bench*. arXiv:2610.12183, October 2026.

**Reference caveat:** These sources establish context and methodological precedents; they do not establish that any arm here will succeed. Check current links, software compatibility and licensing before implementation.
