# Review of the earlier draft spec

> This reviews the earlier draft of the project (then numbered v1.0, now `docs/archive/earlier_draft_spec.md`). Its required-tier fixes were folded into that draft and carried into the current spec's methodology. Section numbers below refer to the earlier draft.

**Date:** 2026-11-03. **Method:** five independent review lenses (literature, math, design, mechanism, feasibility)
produced 74 findings. An adversarial verifier then checked each lens. Below are the findings that survived,
merged, with severities as the verifiers corrected them. Raw per-finding output is in the session's
workflow journal. **Every item is a proposal.** None is adopted until it is logged in
[research_decisions.md](research_decisions.md).

## Verdict

The integrity scaffolding is sound and unusually careful: a sealed test, pre-registration, mandatory random
controls, query accounting, honest handling of negative results, and a correct d/q projection identity.
**As written, though, the study cannot cleanly answer its own primary hypothesis.** Three problems cause this:

1. The behaviour basis B2 differs from the random basis B0 in **structure** (rank-1, one fixed output direction)
   as well as **information**. Nothing separates the two.
2. The random null is a **single draw per seed over 3 seeds**, so "beats RAND on most seeds" is close to a coin flip.
3. The pre-declared primary contrast, NEUROSCULPT vs RAND, changes **support and basis together**, so it tests neither H1 nor H2.

The novelty framing also misses 2026 work that already runs localize → CMA-ES repair with a random-localization
ablation. All of this can be fixed before the freeze. Compute is negligible (240 FBE ≈ 40 GFLOP ≈ 1 s on CPU),
so the real cost of the fixes is coding and analysis time.

## Critical

**C1. B2 is a rank-1 family, so structure is confounded with information (§6 B2, H1).**
span{u e_{s_j}^T} = {u v^T : v on S}. Every B2 edit has rank 1 with a fixed left factor u, while B0 edits are
generically rank 8. Orthogonalisation does nothing for B2, because the vectors are already orthonormal. A B2 win
could be a rank-1 regularisation effect.
→ Add **B0-u**: the same u e_{s_j}^T form with a random unit u, over many draws. Judge the information claim as
B2 vs B0-u, and report B2 vs B0 as the combined effect.

**C2. The random null is far too weak (§9.2, §10.4–10.5).**
One random support and basis draw per seed, with 3 seeds, cannot support inference.
→ Draw ≥100–200 random supports and bases per seed (seconds of compute) and report NEUROSCULPT's **percentile
in the null distribution**. Use ≥6–10 base seeds, since an MNIST MLP trains in seconds. Pre-register a smallest
effect of interest (e.g. 0.5 pp), a seed-level test (Wilcoxon or hierarchical bootstrap) and a TOST margin for
declaring a null.

## Major

**M1. The primary contrasts are mislabelled (§10.4, line 418).**
→ H1 is B2 vs B0 on the same support: S2 is the primary stratum (CAUS-RAND vs NEUROSCULPT) and S0 the secondary
(RAND vs BEHAV). H2 is S2 vs S0 at a fixed basis, preferably also under ADAM-MASK on every support.
NEUROSCULPT vs RAND becomes descriptive only. State whether B1 counts toward H1.

**M2. Novelty and prior art (§2.2–2.3).** These need citing and positioning:
- SHARPEN (arXiv:2604.00422, Apr 2026): Deep-SHAP and activation-contrast localization, CMA-ES repair, and a
  top-k vs random vs bottom-k neuron ablation, which is effectively H2.
- RepTran (arXiv:2607.11193): beats random weight selection using differential evolution.
- CARE (ICSE 2022), Arachne (TOSEM 2023), PRDNN (PLDI 2021).
- Surgical fine-tuning (ICLR 2023), BUFR (ICLR 2022), FOA (ICML 2024), DFR (ICLR 2023), AlphaEdit (ICLR 2025),
  BBT (ICML 2022).

→ Narrow the contribution to **basis/subspace choice at a fixed support, with diagnostic costs charged to the
method, under natural corruption shift**. Reframe H2 as a budget-matched replication in a new setting.
Add a **FULL-SUPPORT** arm (the same optimizer over all 512 support coordinates), because that is what every prior
search-based repair method does.

**M3. B2's contrast is weakly motivated (§6).**
"Incorrect minus correct" uses corrupted inputs on both sides, so the shared corruption shift cancels and the
clean/corrupt pairing goes unused. A single class-agnostic u cannot send different classes' errors to their
correct classes. B2 also forces d = k, so the d- and k-ablations in §9.3 are undefined for it.
→ Promote the paired **pre-activation residual** r = a2(x) − a2(x̃) as the primary B2, with class-conditional
directions or a top-r SVD so that d ≠ k is possible. Use label shuffling as the control for the correct/incorrect
variant and pair shuffling for the paired variant.

**M4. B1 at d = k = 8 is degenerate (§6 B1, §9.3).**
B1 uses every singular mode, so it can only rescale singular values within range(W_S). The dominant-vs-trailing
ablation is impossible at this setting.
→ Use d < k for that ablation, or pre-specify the u_i v_j^T cross-products.

**M5. Localization ≠ editability (§5).**
ΔW[:,j] acts through h1_j(x). A neuron that the corruption **silences** (h1_j(x̃) = 0) cannot be fixed by an
outgoing edit, yet it still shifts clean inputs. S1 and S2 score silenced and spuriously activated neurons alike.
Random supports may also hit dead neurons, which makes the null artificially easy to beat.
→ Use a signed S1 and log P(h1_j(x̃) > 0) for selected columns. Add an activation-energy/selectivity support
(e.g. E_T[h1_j(x̃)²]/E_P[h1_j(x)²]) and an activation-matched random support. Pre-register the prediction
"S2 ≤ S0 when the selected columns are mostly silenced". Note that the S2 pre-screen comes from S1's top 32, so
S1 is not optional.

**M6. The support-relative radius confounds H2 (§7.3).**
ρ as a fraction of ‖W2[:,S]‖_F gives different supports different absolute and functional budgets.
→ Add a robustness run that matches the activation-weighted norm √tr(ΔW Σ_S ΔW^T).

**M7. Missing cheap and ceiling baselines (§9.1).**
Pre-register these as required arms:
- Closed-form **ridge least-squares** edit on S that maps h1_S(x̃) to a2(x) with a clean-covariance penalty
  (forward-only, ROME-style).
- **Mean/bias shift**, and a **b3 logit-bias** search. This is the H4 class-prior control. Report the
  mean-shift fraction ‖ΔW h̄_S‖²/E‖ΔW h_S‖².
- **Last-layer refit**.
- **FULL-FT** ceiling. MNIST-C's authors note that fine-tuning "trivially" recovers any single corruption, so
  report the fraction of the recoverable gap each method closes.
- Promote **Adam-in-basis** (the subspace oracle) and the **gradient-capture fraction** ‖P_B g_S‖²/‖g_S‖²
  (random expectation d/q ≈ 1.6%) to required, harness-only diagnostics. Split H1 into H1a (span quality at
  convergence) and H1b (efficiency at 240).

**M8. Edit site and corruption choice (§1.4, §8).**
For input-level corruptions, Surgical FT predicts that early layers work best, and a b1 or W1-row edit can cancel
an additive shift that a W2 column edit cannot. The example corruptions are poor choices: translation is geometric
and probably unrepairable at W2, while shot noise likely leaves little headroom. The CNN numbers are not
transferable to the MLP; measure the MLP directly.
→ Make two pilots mandatory before the freeze, using calibration data and base-model metrics only.
(a) A **headroom pilot** over all 15 corruptions: pick a non-geometric corruption with a moderate, seed-stable drop.
(b) A **site-ceiling pilot**: matched-mask Adam at W2[:,S], b1, W1 rows and W3. Keep W2 for the factorial and
report its ceiling as a fraction of the best site's.

**M9. Data protocol gaps (§8).**
- The adaptation and preservation pools must be **held out before base training**, e.g. 10k MNIST-train IDs.
  Otherwise the preservation data is memorised, and KL, CE and the gate understate the damage.
- Verify that MNIST-C is index-aligned with MNIST, which is undocumented: take clean views from MNIST-C's
  `identity/` folder, assert label and pixel equality, and run a per-index correlation check.
- **Labeled-data asymmetry:** S2 and B2 use a larger labeled calibration pool that RAND never sees. Pre-register a
  labeled-example policy (e.g. append calibration data to the data-free methods' objective) and report
  "labeled examples used" as its own column. S1 needs unlabeled pairs only; SPEC needs no data.

**M10. Selection and budget accounting (§4.3, §7.3, §12.2).**
- **NO_SAFE_EDIT can never trigger**, because z = 0 is always evaluated and always passes the gate. Redefine it
  as "no gated candidate beats z = 0 on selection by margin δ".
- Scoring all ~240 candidates on the selection split turns it into a second search set, with a winner's curse,
  correlated multiple testing and uncharged forwards. → Rescore only the top-K (e.g. 5) by search J, and apply
  the gate with a one-sided paired-bootstrap bound.
- ρ/λ sweeps, the selection forwards and the S2 calibration size are uncharged or undefined. Task A
  localization has no prescreen and costs about 170 FBE. A partial forward counts as one query; log MACs as a
  secondary measure.

**M11. Task A design (§8, §10.3).**
One W2 column has norm ≈ 0.35‖W_S‖_F, but ρ_max = 0.10, so exact inversion may be infeasible for every method.
Large low-rank damage aligns with the top singular vectors and hands B1 a privileged view. The h_{1,j} ← 0 probe
is itself a column-ablation edit. Task A also appears in no stage of §12.3.
→ Have the harness report ‖Δ*‖/‖W_S‖ and extend the ρ grid to reach it. Use full-rank or non-spectral damage
families. Label the probe "edit-probe localization" and keep it out of H2 evidence. Add Task A to P2/P3 as the
oracle positive control.

## Minor

- **Coordinate search depends on the basis axes**, so span quality is confounded with axis alignment. Make
  CMA-ES (rotation-invariant) or random search primary, or add random within-span rotations.
- **Preservation KL** is second-order at θ0 and nearly collinear with CE for confident examples. "Scale by
  baseline score" is undefined for terms that are 0 at θ0 (K_P, μ‖ΔW‖²). Test H3 at a radius where the
  unpenalized search actually loses selection accuracy, and name an H3 arm (λ = 0, gate kept and removed).
- **B2 ↔ gradient boundary.** dL/dW2 is computable from forward quantities. Label every basis with an
  information taxonomy (weights only / forward unlabeled or paired / label-grouped / error-propagated = oracle).
- **Numerics.** Run evaluation on CPU in float64 with deterministic settings. Test z = 0 bitwise on the same code
  path, and use allclose plus equal argmax across paths. Cache h1 and precompute B_j h1 for about 100× faster
  evaluation.
- **Licensing row is wrong.** MNIST-C data is CC BY 4.0 and its code is Apache-2.0. The GitHub repo was
  archived in Apr 2026, so pin the Zenodo zip (record 3239543) by SHA-256. MNIST's CC BY-SA 3.0 status is unverified
  at the source. Keep the employment-contract check.
- **References.** R4 is software documentation: cite Hansen & Ostermeier (2001) or arXiv:1604.00772 for
  CMA-ES theory. Complete R9 as Du, Zhang, Wang, Liu, Qian, *CAAI TIT* 11(3):816–834, 2026. Add arXiv:1906.02337
  to R11. §14 conflates two separate 2026 Shapley papers. R2, R3, R5–R8 and R10–R13 checked out.
- **H5** stays untestable until an alignment method is defined (e.g. Hungarian matching on activation
  correlation).

## Scope guidance

The verifiers warned that adopting every proposed arm as one flat factorial reintroduces the scope creep the spec
deliberately avoids. Tier the changes instead:

- **Required for v1:** C1, C2, M1, M7 (closed-form + bias + FULL-FT + oracle diagnostics), M8 pilots, M9 and M10.
  B0-u, many null draws, ADAM-MASK on every support, and a FULL-SUPPORT arm.
- **Secondary:** paired-residual B2 variants, an activation-energy support or S3, the functional-budget H2
  check, and the Task A damage families.
- **Optional:** AlphaEdit-style projected basis, alternative-site search arms, and H5.
