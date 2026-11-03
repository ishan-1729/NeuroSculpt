# NeuroSculpt

**Structured, Gradient-Free Weight-Space Search for Targeted Neural Network Repair**  
**Research and Implementation Specification — v1.1**  
**Prepared:** 3 November 2026 (v1.0); revised 3 November 2026 (v1.1)  
**Status:** Proposed research design; not an implemented or experimentally validated system  
**Audience:** Research scientist / frontier coding agent reviewing, revising, and implementing the project

> **Mission.** Test whether behavior-informed, low-dimensional edits to a frozen neural network can repair a specific failure more efficiently than matched random edits, while preserving unrelated behavior. Discover when this works, when it fails, and which structural assumptions explain the outcome.

> **Interpretation rule.** Every numerical budget and acceptance threshold in this document is a *proposed experimental-design choice*, not a reported result. The implementing agent may revise choices with a written rationale; it must not present hypotheses as findings.

> **v1.1 revision.** This version adopts the *required tier* of the multi-lens review in `docs/spec_review.md`; each change is logged in `docs/research_decisions.md`. Main changes: a structure-matched random basis (B0-u) and a 200-draw random null per seed; ≥6 base seeds; per-hypothesis primary contrasts; repositioning against 2026 search-based repair work (SHARPEN, RepTran) and a full-support search arm; closed-form, bias, last-layer and full-fine-tune baselines plus a harness-only subspace oracle; mandatory headroom and edit-site pilots; adaptation pools held out from base training; a labeled-data policy; top-K selection rescoring with a redefined `NO_IMPROVING_EDIT`; and corrected licensing and references. Secondary-tier items are marked **[secondary]** and run only after the required comparisons.

## 1. Executive specification

### 1.1 Central question

Given a trained model with parameters \(\theta_0\) and a clearly specified failure distribution, can we find a small, structured update \(\Delta\theta\) without computing training-loss gradients, such that performance improves on the failure distribution and changes minimally elsewhere?

The core contribution being tested is **subspace construction**, not the invention of zeroth-order optimization. The project compares structured directions informed by the network's weights and behavior against equally sized random directions, under matched model-query budgets. A secondary question asks whether causal activation localization yields *editable* parameter supports better than simpler heuristics.

### 1.2 Non-goals

- Do **not** claim to invent sparse model editing, causal tracing, derivative-free optimization, or model repair; these are established research areas.
- Do **not** try to reproduce the November 2026 weight-space-programming article's large-language-model results or claim comparable gains.
- Do **not** train a frontier model, develop a general-purpose interpretability agent, or optimize all network parameters.
- Do **not** require a cloud deployment, authentication system, full-featured web UI, distributed platform, or long hyperparameter search.
- Do **not** call a standard Adam update 'gradient-free' or silently use autograd in the proposed search method. Gradient-based methods are required *as explicit baselines*.

### 1.3 Minimum viable research product (MVRP)

1. A deterministic PyTorch model and reproducible dataset splits.
2. Two scenarios: **controlled parameter damage** (mechanistic sanity check) and **naturally occurring distribution-shift failure** (main generalization test).
3. Several fixed-support / fixed-dimension search bases, a simple zeroth-order optimizer, a structure-matched random control and a many-draw random null distribution.
4. Cheap closed-form and bias baselines, gradient-based ceilings, and a harness-only subspace oracle that put any gain in context.
5. Preservation-aware selection and an honest held-out evaluation with uncertainty.
6. Reproducible scripts, unit tests, compact plots, results tables, an editable weight patch, and a technical report.

### 1.4 Default scope (adjustable only with justification)

| Item | Default | Why |
|---|---|---|
| Model | MLP: 784 → 128 → 64 → 10, ReLU, no batch normalization | Fast, interpretable, supports exact layer interventions |
| Base data | MNIST | Small, standardized, easy to diagnose |
| Shift task | One non-geometric MNIST-C corruption, chosen by a preregistered headroom pilot (§8.3), adapted on pools held out from base training, sealed official test | Genuine non-planted adaptation rather than artificial recovery |
| Edited matrix | Second-layer weight matrix \(W_2 \in \mathbb R^{64\times128}\) for the factorial; a site-ceiling pilot (§8.3) reports how W2 compares with b1, W1 rows and W3 | Edits the mapping from first-layer features into second-layer features; kept fixed so the factorial stays clean |
| Editable support | \(k=8\) selected input-neuron columns of \(W_2\) | Exactly 512 candidate matrix entries, before low-dimensional restriction |
| Search dimension | \(d=8\) coefficients | Tractable zeroth-order optimization |
| Main optimizers | CMA-ES (rotation-invariant) as primary; coordinate search secondary; random-search control | Coordinate search depends on basis axes, which confounds span quality with axis alignment |
| Main budget | Up to ~240 full-batch-equivalent evaluations per method/seed, including diagnostic and selection calls in end-to-end comparisons | A sample-efficiency choice, not a compute limit (240 FBE ≈ 40 GFLOP); predefine before experiments |
| Seeds | ≥6 independent base-model seeds (10 preferred) | An MNIST MLP trains in seconds; 3 seeds cannot support inference |
| Random null | 200 random draws per seed for every random component (support, basis) | Gives a null distribution and a percentile, not a single noisy comparator |
| Hardware | CPU-capable; optional single consumer GPU | No reliance on LLM-scale infrastructure |
| Interface | CLI and saved plots first; optional Streamlit page only after validation | Prioritize experiments over visual polish |

**Critical distinction:** choosing a 512-entry support does **not** mean independently optimizing 512 variables. The proposed search manipulates only \(d=8\) coefficients in a fixed basis inside that support. Report *both* support size and the number of entries actually changed.

---

## 2. Scientific context and contribution boundary

### 2.1 Inspiration

Takehi et al., *Sparks of Weight-Space Programming* (2 November 2026), report exploratory experiments in which an LLM reads or modifies neural-network weights directly. One experiment targets excessive repetition in a small reasoning model, using a highly localized edit to improve measured benchmark completion and accuracy under the authors' settings [R1]. This provides motivation, **not proof**, that a conventional search procedure will find equally useful edits.

### 2.2 Prior work the project must acknowledge

- **Intrinsic dimension:** Li et al. study optimization in low-dimensional random subspaces [R2]. Therefore, success with a small edit vector alone is not a novelty claim.
- **Derivative-free / zeroth-order optimization:** MeZO and standard evolutionary strategies optimize through forward evaluations [R3, R4]. The novelty cannot simply be 'no backpropagation.'
- **Model editing:** ROME, task arithmetic, and other approaches directly manipulate parameters [R5, R6]. Sparse/local edits and rank-one parameter updates are established.
- **Causal interpretation caveats:** Hase et al. show that a component's importance in causal localization need not predict where a beneficial weight edit lies [R7]. Zhang and Nanda show sensitivity of activation-patching conclusions to methodology [R8].
- **Search-based neural network repair:** locate-then-search repair of localized weights is established. CARE uses causal neuron localization plus particle-swarm search [R14]; Arachne uses differential evolution on localized weights [R15]; PRDNN reduces single-layer repair to a linear program [R16]. Two separate 2026 Shapley-guided papers exist: NSRepair (interpretable fault localization plus multi-objective GA) [R9] and SHARPEN (Deep-SHAP and activation-contrast localization, CMA-ES repair, and a top-k vs random vs bottom-k neuron ablation) [R17]. RepTran also beats random weight selection with differential evolution [R18]. All of these optimize the **raw coordinates** of the localized support; none compares structured low-dimensional bases inside a fixed support, and none charges localization forwards to a matched budget.
- **Small-parameter adaptation to shift:** surgical fine-tuning finds that early layers are best for input-level corruptions [R19]; bottom-up feature restoration makes the same point for measurement shift on MNIST-C-type data [R20]; forward-only test-time adaptation uses CMA-ES plus activation shifting [R21]; last-layer retraining is a strong cheap baseline [R22]; closed-form rank-one layer rewrites fix input-level concept failures in vision classifiers [R23].
- **Subspaces for derivative-free search and preservation-aware edits:** black-box tuning runs CMA-ES in random low-dimensional subspaces and finds it competitive with Adam at small d [R24]; AlphaEdit projects edits onto the null space of preserved representations [R25].

NeuroSculpt must therefore be positioned as a focused comparative study of **edit bases inside a fixed support, under matched and fully charged budgets, on natural corruption shift**. It is not the first locate-then-repair framework, and H2 (localized vs random supports) is a budget-matched replication in a new setting, not a first test.

### 2.3 Precise candidate contribution

The intended result is an **empirical method or finding** about the utility of structured weight-edit subspaces: under identical data, support, dimensionality and evaluation budgets, do spectral and behavior-guided bases yield better target/preservation trade-offs than random bases, including random bases with the *same structural form*, and than searching the raw support coordinates as prior repair work does? Separately, does causal localization of activations yield better editable supports than random or activation-contrast selection once localization cost and labeled data are charged? The project should also identify failure regimes, not only successes, and report every gain against cheap closed-form baselines and gradient-based ceilings.

A rigorous negative result is acceptable. For example: 'Causal localization significantly predicts output sensitivity but does not outperform randomly selected editable columns' would be scientifically informative if demonstrated carefully.

---

## 3. Research questions and falsifiable hypotheses

| ID | Question / hypothesis | What would falsify it? |
|---|---|---|
| H1a (primary) | At fixed support, search dimension and query budget, the behavior-informed basis B2 achieves higher held-out repair gain under zeroth-order search than the random-basis null B0. | NEUROSCULPT's seed-level gain over the B0 null falls inside the TOST equivalence margin (±0.5 pp), or is negative. |
| H1b | The B2 advantage comes from behavioral information, not only from B2's rank-1 structural form. | B2 does not beat the structure-matched random control B0-u (same \(u e_{s_j}^\top\) form, random \(u\)). |
| H1c | The advantage reflects a better span, not only easier search. | The harness-only subspace oracle (Adam in \(z\)) shows no span advantage for B2 over B0. |
| H2 | At fixed basis and budget, causally localized columns beat random columns as editable supports *once localization forwards and labeled data are charged* (a budget-matched replication of SHARPEN-style findings [R17] on natural shift). | No advantage beyond the margin on the seed-level test, including under ADAM-MASK on each support. |
| H3 | Preservation penalties improve the target-versus-retention Pareto frontier. | The named H3 arm (\(\lambda_{\mathrm{CE}}=\lambda_{\mathrm{KL}}=0\), gate kept and gate removed) matches retention at the radius where the unpenalized search measurably loses selection accuracy. |
| H4 | Improvements on a natural shift cannot be explained by overfitting the adaptation samples or by changing predicted-class frequencies. | Improvement disappears on the sealed test, or the LOGIT-BIAS baseline matches it, or the predicted-class histogram shift explains it. |
| H5 (optional) | Useful low-dimensional directions exhibit cross-seed stability. | Untestable until an alignment method (e.g. Hungarian matching on activation correlation) is defined; deferred. |

**Primary hypothesis for the MVP:** H1a. H1b and H1c are required secondary contrasts that attribute any H1a effect. H2 and H3 are secondary. H4 is a required validity check. H5 is an extension; it must not delay the initial result.

**Pre-declared contrasts.** H1a: B2 vs the B0 null on the causal support S2 (CAUS-RAND distribution vs NEUROSCULPT) as the primary stratum, and on random supports (RAND vs BEHAV) as the secondary stratum. H1b: B2 vs B0-u on S2. H2: S2 vs the S0 null at fixed basis B0, and under ADAM-MASK on each support. NEUROSCULPT vs RAND (support and basis both changed) is reported **descriptively only**. Spectral B1 is a weights-only comparator; it does not count toward H1, which is about behavior information.

**Smallest effect of interest and tests.** For every primary and secondary contrast: the metric is held-out \(G_T\), with NEUROSCULPT's percentile in the null distribution per seed; the smallest effect of interest is 0.5 pp; the test is a one-sided seed-level Wilcoxon signed-rank test (α = 0.05) plus a hierarchical bootstrap 95% CI over seeds and test examples; a null is declared by TOST with margin ±0.5 pp.

**Pre-registration:** Freeze the main comparison list, seed list, dataset splits, target corruption, model architecture, search budget and metric definitions in `configs/study_v1.yaml` before opening any final-test results. Revision history must capture later deviations.

---

## 4. Formal mathematical problem

### 4.1 Notation

Let \(f_\theta(x)\) be the logits of a trained neural network with flattened parameters \(\theta\in\mathbb R^P\). Fix the pretrained parameters \(\theta_0\). Let \(\mathcal D_T\) denote a *target* failure distribution and \(\mathcal D_P\) a *preservation* distribution. Their datasets must be split into calibration, search, selection, and sealed-test partitions, with no example leakage.

For classification, define \(p_\theta(y\mid x)=\mathrm{softmax}(f_\theta(x))_y\) and cross-entropy \(\ell_{\mathrm{CE}}(\theta;x,y)=-\log p_\theta(y\mid x)\). Define:

\[
L_T(\theta)=\mathbb E_{(x,y)\sim\mathcal D_T}\,\ell_{\mathrm{CE}}(\theta;x,y),\qquad
L_P(\theta)=\mathbb E_{(x,y)\sim\mathcal D_P}\,\ell_{\mathrm{CE}}(\theta;x,y).
\]

A preservation measure compares the edited and original predictive distributions:

\[
K_P(\theta_0,\theta)=\mathbb E_{x\sim\mathcal D_P}\,D_{\mathrm{KL}}\!\left(p_{\theta_0}(\cdot\mid x)\,\|\,p_\theta(\cdot\mid x)\right).
\]

We seek a small edit that reduces target loss without degrading preservation performance. The ideal constrained formulation is:

\[
\min_{\Delta\theta}\;L_T(\theta_0+\Delta\theta)\quad
\text{subject to}\quad
A_P(\theta_0+\Delta\theta)\ge A_P(\theta_0)-\varepsilon,\;
\|\Delta\theta\|_F\le\rho,\;
\operatorname{groupsupp}(\Delta\theta)\le k.
\]

Here, \(A_P\) is preservation accuracy, \(\varepsilon\) is the allowed absolute accuracy drop (percentage points), and the group support limits the number of selected neuron-column groups. The optimization actually occurs over a fixed low-dimensional edit family defined below.

### 4.2 Structured, low-dimensional edit parameterization

For a chosen weight matrix \(W\in\mathbb R^{m\times n}\), choose an editable column set \(S\subseteq\{1,\dots,n\}\), \(|S|=k\), and normalized, linearly independent basis matrices \(B_1,\dots,B_d\), each zero outside those columns. Then:

\[
\Delta W(z)=\sum_{j=1}^{d}z_j B_j,\quad z\in\mathbb R^d,\quad
W(z)=W_0+\Delta W(z).
\]

Each basis element may be rank one, \(B_j=u_jv_j^\top\), with \(\|B_j\|_F=1\). Treat \(B=[\mathrm{vec}(B_1),\dots,\mathrm{vec}(B_d)]\) as a matrix of flattened directions. **Orthogonalize** its columns (QR or SVD) within the supported entries when feasible, so differences between methods are not merely condition-number or magnitude differences. The rest of \(\theta_0\) remains frozen.

The edit is therefore \(\theta(z)=\theta_0+E_S Bz\), where \(E_S\) embeds supported weight entries into the full parameter vector. The exact map must be unit-tested for indexing and normalization errors.

### 4.3 Practical scalar objective and hard validation gate

Use a smooth search objective over fixed adaptation batches:

\[
J(z)=L_T(\theta(z))+\lambda_{\mathrm{CE}}L_P(\theta(z))
+\lambda_{\mathrm{KL}}K_P(\theta_0,\theta(z))
+\mu\|\Delta W(z)\|_F^2.
\]

Scale \(L_T\) and \(L_P\) by their values at \(\theta_0\). \(K_P\) and \(\mu\|\Delta W\|_F^2\) are exactly 0 at \(\theta_0\), so scale them instead by their value at a reference edit of norm \(\rho\) along the first basis direction, and record that reference. Note that \(K_P\) has zero value and zero gradient at \(\theta_0\) (its Hessian is the Fisher), so it is second order in \(\rho\) and nearly collinear with \(L_P\) on confidently correct examples: either tune one of \(\lambda_{\mathrm{CE}}, \lambda_{\mathrm{KL}}\) and fix the other at 0, or justify tuning both. Tune coefficients on the **selection** split, never the final test. An alternative implementation is an augmented penalty for exceeding an accuracy-loss tolerance, but avoid making noisy, discontinuous accuracy the sole search signal.

**Selection rule.** Rank all search candidates by \(J\) on `edit_search` and rescore only the top \(K=5\) on the selection split; those selection forwards are charged to the method's budget. A candidate passes the preservation gate when the one-sided 95% paired-bootstrap upper bound on its selection preservation drop is at most \(\varepsilon\). Among passing candidates, select the lowest selection target cross-entropy (tie-breaker: smallest \(\|\Delta W\|_F\)). Because \(z=0\) is always evaluated and always passes the gate, a gate failure alone can never occur; instead, if no passing candidate beats \(z=0\) on selection target cross-entropy by at least a preregistered margin \(\delta\), retain the original model and record `NO_IMPROVING_EDIT`. Do not hide the failure by cherry-picking a candidate on test data.

### 4.4 Why structured subspaces might work (and why they may not)

Locally, for a smooth surrogate loss \(J\):

\[
J(\theta_0+Bz)\approx J(\theta_0)
+(B^\top g)^\top z+\tfrac12 z^\top(B^\top H B)z,
\]

where \(g=\nabla_\theta J(\theta_0)\) and \(H\) is the Hessian. This is a **theoretical interpretation**, not permission to calculate \(g\) or \(H\) in the proposed gradient-free method.

For an isotropically random \(d\)-dimensional orthonormal basis **within the chosen editable support** of \(q=mk\) scalar coordinates, let \(g_S\) be the projection of the full gradient onto those supported coordinates. Then:

\[
\mathbb E\|\operatorname{Proj}_{B}g_S\|_2^2=(d/q)\|g_S\|_2^2.
\]

This is an exact expectation for uniformly random subspaces of that **supported coordinate space**; it is not generally \(d/P\) when only a small group of weights is editable. Thus low-dimensional random search may discard much of the locally useful gradient signal. A behavior-informed subspace might capture more, but **no general guarantee follows**. Moreover, weight SVD directions describe existing parameter energy, not necessarily loss curvature or repair directions. A well-designed experiment must test this, not assume it.

### 4.5 Identifiability and symmetries

Neurons in hidden layers can be permuted together with matching adjacent weights without changing the network's function. ReLU networks also admit positive scaling transformations of a neuron and compensating outgoing weights. Consequently, raw coordinate similarity across independently trained networks is not itself evidence of a shared circuit. Normalize update directions, compare functional effects, and align neurons before any cross-seed weight-space transfer claim.

---

## 5. Candidate support localization: which weights may be edited?

**Primary matrix:** second linear layer \(W_2\), mapping first-layer hidden activations \(h_1\in\mathbb R^{128}\) to second-layer preactivations \(a_2=W_2h_1+b_2\). Editing column \(j\) changes the contribution of neuron \(h_{1,j}\) to all 64 second-layer preactivations. Candidate supports contain \(k=8\) columns.

### S0. Random support (mandatory control)

Choose 8 columns uniformly without replacement, with a recorded seed. This must use the same matrix, cardinality, and downstream optimizer as the structured methods. Draw **200 random supports per base seed** to form a null distribution; report the structured method's percentile in it, not a comparison with one draw. Log the fraction of dead neurons (\(P(h_{1,j}(\tilde x)>0)<0.01\)) in each draw, since dead columns make the null artificially easy to beat. **[secondary]** An activation-matched random support (S0-act) draws columns with probability proportional to \(\mathbb E[h_{1,j}(\tilde x)^2]\).

### S1. Activation-contrast support (cheap, noncausal)

On *paired clean and corrupted inputs* \((x,\tilde x)\), compute per-neuron differences in the first hidden layer:

\[
c_j=\mathbb E_{(x,\tilde x)}\big|h_{1,j}(x)-h_{1,j}(\tilde x)\big|.
\]

Rank columns by \(c_j\), optionally normalized by activation scale. This identifies sensitivity to corruption but does **not** establish a neuron causes failure. Compare against random S0. S1 is **required**, not optional, because S2 pre-screens from S1's top 32. Also log the signed contrast \(\delta_j=\mathbb E[h_{1,j}(\tilde x)-h_{1,j}(x)]\): \(\delta_j>0\) marks a *spuriously activated* neuron and \(\delta_j<0\) a *silenced* one. S1 needs unlabeled clean/corrupt pairs only.

### S2. Causal activation-restoration support (main structured method)

For a corrupted sample \(\tilde x\), run the model normally and then re-run while replacing one first-layer activation \(h_{1,j}(\tilde x)\) with \(h_{1,j}(x)\) from its paired clean input. Define the estimated repair effect:

\[
r_j=\mathbb E\left[
\ell_{\mathrm{CE}}(\theta_0;\tilde x,y)
-\ell_{\mathrm{CE}}(\theta_0;\tilde x,y\mid h_{1,j}\leftarrow h_{1,j}(x))
\right].
\]

Positive \(r_j\) means this activation restoration helps *under this intervention and test distribution*; it does not imply editing the corresponding outgoing column will help. Pre-screen the top ~32 activation-contrast candidates for expensive activation-restoration tests, then select 8 columns by mean \(r_j\) on a **separate calibration split**. Include wrong-pair or random-activation restoration controls if budget permits. Charge all these diagnostic forwards to the total experiment cost.

**Editability caveat (mechanistic prediction).** An edit to column \(j\) changes \(a_2\) by \(\Delta W_{[:,j]}\,h_{1,j}(x)\) for *both* corrupted and clean inputs. A neuron the corruption silences (\(h_{1,j}(\tilde x)=0\)) cannot be fixed through its outgoing column at all, yet its edit still moves clean inputs. S1 and S2 score silenced and spuriously activated neurons alike, so which kind dominates the selected support depends on the corruption. For every selected column log \(\delta_j\), \(P(h_{1,j}(\tilde x)>0)\), \(\mathbb E[h_{1,j}(\tilde x)^2]\) and \(\mathbb E[h_{1,j}(x)^2]\). **Pre-registered prediction:** S2 ≤ S0 when most selected columns are silenced, and S2 may beat S0 when most are spuriously active; read H2 conditional on this signature. **[secondary]** An editability support S-edit ranks columns by \(\mathbb E_T[h_{1,j}(\tilde x)^2]/(\mathbb E_P[h_{1,j}(x)^2]+\epsilon)\).

**For controlled planted damage:** Paired clean/corrupt activation restoration is not defined in the same way because the fault lies in the *weights*, not the input. Use a declared **S2-damage ablation variant** instead: for labeled samples, intervene on first-layer neuron \(j\) by setting \(h_{1,j}\leftarrow0\) in the damaged model, and rank candidate columns by the resulting improvement in target cross-entropy minus any measured preservation degradation. This uses only the damaged model and permitted labeled calibration data; the pristine weights and true injected support remain hidden. Setting \(h_{1,j}\leftarrow0\) is exactly the column-ablation edit \(\Delta W_{[:,j]}=-W_{2[:,j]}\), so label it **edit-probe localization**, keep it out of H2 evidence, and do not pool its scores with Task B restoration scores. Use the same 32-column S1-style pre-screen as Task B (with damaged-vs-calibration activation statistics) and charge its forwards to the budget.

### Localization validity warnings

- Restoring an activation is not the same operation as changing its outgoing weights; report both intervention scores and actual edit success.
- Strong source/target activation differences can reflect benign covariate shift rather than a failure circuit.
- Rank support candidates on calibration data only; do not optimize or rank on final-test examples.
- Do not use a layer-scan that accidentally grants structured methods far more model queries than random controls without reporting the difference.

---

## 6. Candidate edit directions: how may the selected weights change?

The project should compare at least the following three bases **on the same supports**. This factorial structure separates 'where to edit' from 'which way to edit.'

### B0. Isotropic random orthonormal basis (mandatory)

Draw \(d\) Gaussian vectors in the \(mk\)-dimensional supported coordinate space, apply QR, embed the resulting unit vectors into \(W_2\). Fix the basis at the start of a run; do not resample it on every candidate evaluation. Use the same \(d\), support and maximum Frobenius step radius as other methods. Draw **200 bases per support per base seed** for the null distribution. B0 edits are generically rank \(\min(m,k)=8\).

### B0-u. Structure-matched random basis (mandatory)

Use exactly B2's form, \(\widetilde B_j=u\,e_{s_j}^\top\), but with \(u\) drawn uniformly on the unit sphere \(S^{63}\); 200 draws per support per seed. Every B0-u edit is rank 1 with a fixed output direction, like B2. B2 vs B0-u isolates behavioral information from rank-1 structure (H1b).

### Information taxonomy

Label every basis and support by the information it uses: (i) weights only (B0, B0-u, B1, S0); (ii) forward activations, unlabeled or paired (S1, B2-pair); (iii) label-grouped activations (S2, B2); (iv) output error propagated through \(W_3^\top\), which is gradient-equivalent (the oracle diagnostics in §7.4 only). For this MLP, \(\partial L/\partial W_2=\mathbb E[(\mathbb 1[a_2>0]\odot W_3^\top(p-y))\,h_1^\top]\) is computable from forward quantities, so "no autograd" is not the same as "no gradient information"; category (iv) is forbidden in the proposed method.

### B1. Spectral basis (weight-space geometry)

Compute \(W_{2[:,S]}=U\Sigma V^\top\) on the supported submatrix. Construct rank-one directions \(u_i v_i^\top\), embedded back into the full matrix. For \(d\le\operatorname{rank}(W_{2[:,S]})\), use the first \(d\) singular pairs. If exploring \(d\) above the available rank, use prespecified orthogonal cross-products \(u_i v_j^\top\) and describe the rule explicitly.

**Degeneracy at \(d=k\).** \(W_{2[:,S]}\) is \(64\times8\) with rank ≤ 8, so at the default \(d=k=8\) B1 uses *every* singular pair and spans \(\{U\operatorname{diag}(z)V^\top\}\): it can only rescale existing singular values inside \(\operatorname{range}(W_S)\). The dominant-vs-trailing ablation is therefore run only at \(d=4<k\) (top-4 vs bottom-4 pairs): high singular value does not imply high editability.

### B2. Behavior-informed basis (proposed)

Estimate an output-space direction at the edited layer from paired examples, using the *second hidden layer* \(h_2\). One simple no-gradient contrast is:

\[
u=\operatorname{normalize}\left(
\mathbb E[h_2(\tilde x)\mid\text{incorrect}]
-\mathbb E[h_2(\tilde x)\mid\text{correct, class-matched}]
\right).
\]

For each chosen first-layer input column \(s_j\), define \(\widetilde B_j=u\,e_{s_j}^\top\). These are already Frobenius-orthonormal (disjoint columns), so orthogonalization changes nothing. Class matching and enough examples are essential; otherwise apparent signal may be caused by class balance. If the contrast is unstable or degenerate, log this and fall back to a declared control, **not** to an unreported hand-picked basis.

**Structural facts.** span\(\{u e_{s_j}^\top\}=\{u v^\top: v \text{ supported on } S\}\): every B2 edit is **rank 1 with a fixed output direction**, so it cannot send different classes' errors toward different correct classes, and B2 forces \(d=k\). Both terms of the contrast use corrupted inputs, so the shared corruption shift cancels and the clean/corrupt pairing is unused; what remains mixes a severity-correlated corruption effect, missing correct-class evidence and possibly a margin/norm term. Log the projection of \(u\) onto the mean-activation direction and onto the class-mean directions. The label-shuffled B2 is the information control for this variant; B0-u is the structure control.

**[secondary] B2-pair, paired residual basis.** Let \(r=a_2(x)-a_2(\tilde x)\) be the paired *pre-activation* residual (this removes the post-ReLU mismatch). Take class-conditional means \(u_c=\mathbb E[r\mid y=c]\) or the top-\(r\) left singular vectors of \(\mathbb E[r\,h_{1,S}(\tilde x)^\top]\), so \(d\ne k\) is possible and directions can differ by class. Its information class is (ii), and pair shuffling is its control. The d- and k-ablations of §9.3 use B2-pair, since B2 forces \(d=k\).

**Important:** For a ReLU layer, an edit in preactivation space is not guaranteed to produce the same shift as the measured post-ReLU contrast. This is an empirical hypothesis, not an algebraic identity.

### Suggested primary comparisons (compact 2 × 3 factorial)

| Support \ Basis | Random B0 (200 draws) | Structure-matched B0-u (200 draws) | Spectral B1 | Behavior B2 |
|---|---:|---:|---:|---:|
| Random S0 (200 draws) | Required null | Required | Required | Required |
| Causal S2 | Required null | Required | Required | **Primary candidate** |

Activation-contrast S1 is required as S2's pre-screen and is also reported as a support with B0. Avoid building all possible support/basis/optimizer combinations before obtaining the first interpretable result.

---

## 7. Optimization without backpropagation

### 7.1 Candidate evaluation API

Implement a pure function `evaluate_candidate(z, support, basis, batch_ids, loss_config) -> metrics` which:

1. Builds \(\Delta W(z)\) without modifying or accumulating changes in the base checkpoint.
2. Evaluates target loss, preservation loss, preservation KL, norm and optional accuracy using the *same fixed adaptation examples* for all candidate comparisons.
3. Operates under `torch.no_grad()` / inference mode, with dropout disabled and no model parameter optimizer.
4. Returns deterministic values for the same candidate, batch and seed (subject to documented device nondeterminism).
5. Records the number of examples processed, forward calls and wall-clock time.

Prefer `torch.func.functional_call` or an equivalently safe state-management pattern, with tests proving `z=0` exactly reproduces the base output and the original checkpoint bytes/tensors are unchanged [R10]. Avoid expensive copying of a whole model per candidate.

**Fast path.** \(W_1\) is frozen, so cache \(h_1\) for every fixed batch and precompute \(B_j h_{1}\); a candidate evaluation is then a linear combination plus two small layers (about 1% of a full forward), and a CMA-ES population can be scored in one batched call. A partial forward still counts as **one** model-example query; log analytic MACs as a secondary cost. **Numerics.** Run editing and evaluation on CPU in float64 with `torch.use_deterministic_algorithms(True)` and a fixed thread count. Test \(z=0\) bitwise against base logits from the same code path and batching; compare the cached path with `functional_call` using `allclose` plus identical argmax.

### 7.2 Optimizers

**O0 — Random search:** sample \(z\) in a bounded \(d\)-ball, including \(z=0\). Mandatory sanity control.  
**O1 — Coordinate / pattern search:** test \(z\pm\eta e_i\), accept the better candidate, reduce \(\eta\) on stagnation. Easy to audit and strong at \(d=8\).  
**O2 — CMA-ES:** use an established implementation with seeded sampling, a bounded search radius and the same candidate-evaluation cap. CMA-ES adapts a distribution over edit coefficients; it requires **objective evaluations**, not gradients [R4].

**Primary optimizer: O2 (CMA-ES).** It is invariant to rotations of the search space, whereas O1 moves along basis axes, so with O1 a basis comparison is confounded with axis alignment. O1 is secondary; if O1 is used for any reported comparison, also run it on a random within-span rotation of each basis. Do not make the main experiment depend on perfectly tuning CMA-ES; use library defaults with a preregistered σ₀ = ρ/2. Run 5 optimizer seeds per configuration and report the mean and spread. Report actual forward-equivalent cost, not just iteration count.

### 7.3 Search bounds and candidate selection

Initialize \(z=0\). Bound \(\|\Delta W(z)\|_F\le\rho\), where \(\rho\) is a fraction of \(\|W_{2[:,S]}\|_F\) and is selected on the validation/selection split. A reasonable *starting sweep* is \(\rho/\|W_{2[:,S]}\|_F\in\{0.01,0.03,0.10,0.30\}\); the values must be treated as tunable engineering guesses, not established optimums. Task A extends the grid until it reaches the harness-reported damage ratio \(\|\Delta^*\|_F/\|W_S\|_F\) (§8).

All methods receive the same radius candidates. The radius/λ sweep is run on the selection split; its forwards are charged to every method equally and reported. **Radius confound.** A relative radius gives different supports different absolute and functional budgets, which confounds H2. **[secondary]** Re-run H2 at a matched activation-weighted norm \(\sqrt{\operatorname{tr}(\Delta W\,\Sigma_S\,\Delta W^\top)}\), with \(\Sigma_S\) the second-moment matrix of \(h_{1,S}\) on calibration data. For **search-only** comparisons, match optimization-query limits and report diagnostic overhead separately. For **end-to-end efficiency** claims, count diagnostic forwards and grant every method the same total model-example-forward budget, reducing a structured method's search allowance accordingly. Candidate selection follows §4.3; final held-out tests are scored **once after all decisions are frozen**.

### 7.4 Gradient-based reference baseline

Include a *matched-mask* Adam baseline that updates the eligible \(W_2\) columns while freezing all other parameters, run **on every support** (each S0 draw used for reporting, S1 and S2), since ADAM-MASK on S2 vs S0 is the cleanest basis-free test of H2. Also include an unrestricted **FULL-FT** ceiling (all parameters, Adam, the same search examples, early stopping on selection) and report each method's *fraction of the recoverable gap closed*, \(G_T/(A_T(\text{FULL-FT})-A_T(\theta_0))\).

**Required harness-only diagnostics (category iv; never available to the proposed method):**
- *Subspace oracle* ADAM-BASIS: Adam or L-BFGS over \(z\) in each basis, run to convergence, which gives the best achievable selection \(J\) and test \(G_T\) inside that span (H1c).
- *Gradient-capture fraction* \(\|P_B\,g_S\|_2^2/\|g_S\|_2^2\) at \(\theta_0\) for every basis, against the random expectation \(d/q=8/512\approx1.6\%\) from §4.4.
- *Converged zeroth-order reference*: the primary optimizer at 5,000 queries, so H1a (efficiency at 240) can be separated from span quality at convergence.

A forward evaluation and a backward step have different costs: compare (i) total wall time, (ii) processed examples / forward-equivalent work, (iii) peak memory and (iv) quality at a given optimization budget. Never claim a gradient-free method beats Adam 'at the same number of steps' without explaining this accounting.

### 7.5 Descent and exotic search extensions (optional)

After a successful baseline, add **two-point finite-difference descent**:

\[
\widehat{\partial_i J}(z)=\frac{J(z+he_i)-J(z-he_i)}{2h},\quad
z_{t+1}=\Pi_\rho(z_t-\eta\widehat\nabla J(z_t)).
\]

This uses \(2d\) objective calls per full finite-difference estimate. More exotic ideas—adaptive sparse support swapping, trust-region search, beam search over support groups, or Pareto evolutionary selection—belong behind explicit P1/P2 gates. They should be introduced only if the core factorial experiment is stable and suggests a specific limitation.

---

## 8. Datasets, tasks and leakage-safe data protocol

### Task A: controlled circuit/weight damage (mandatory sanity check)

1. Train the base MLPs (all preregistered seeds) on the base-training IDs only (§8 split protocol). Record clean validation/test performance and all seeds.
2. Duplicate each model and inject a fixed, localized perturbation into a small set of \(W_2\) columns. Tune perturbation strength **only on calibration data** to produce a measurable, well-defined failure without collapsing all predictions. The harness reports \(\|\Delta^*\|_F/\|W_S\|_F\) (one column alone has norm ≈ \(0.35\|W_S\|_F\)) so the radius grid can reach it; otherwise exact repair is infeasible for every method. Use at least two **basis-neutral damage families**, e.g. full-rank random-direction perturbation and per-column scaling. Large low-rank damage aligns with the top singular vectors of the damaged \(W_S\) and hands B1 a privileged view; if used, report the alignment of \(u_1\) with \(\Delta^*\).
3. Hide injected support, perturbation direction and pristine parameter values from all editing methods. Only the experiment harness may access that ground truth for *post hoc* recovery metrics.
4. Repair using the same labeled adaptation budget allowed to every method; evaluate recovered target performance, collateral damage, support overlap and functional similarity on held-out examples.

Potential failure design: select a few columns influential for a prespecified digit subset, perturb them to degrade accuracy on those digits while retaining substantial performance on the remaining digits. If such localized selective damage is hard to produce, document it and use a less selective but objectively measurable corruption; do not hand-tune injections against the test set.

**Why this task exists:** Known injected damage is the oracle positive control for the weight-edit pipeline, the optimizer and the budget, and it validates the causality diagnostics. It runs in stage P2, before any Task B search. **It does not prove usefulness for naturally occurring model failures.**

### Task B: natural corruption-shift repair (primary scientific result)

1. **Hold out 10,000 MNIST-train IDs before base training.** Train the clean MNIST model on the remaining 50,000 IDs only and freeze it. Otherwise the preservation examples are memorized, and KL, CE and the gate understate collateral damage.
2. Choose **one non-geometric** MNIST-C corruption family by the preregistered headroom pilot (§8.3), *before* any method is run. Translation is a poor choice (a geometric shift that misaligns first-layer templates and is very unlikely to be fixable at W2); shot noise likely leaves too little headroom. Use paired clean/corrupted images from the held-out IDs for calibration and search. MNIST-C contains a single severity per corruption; it should not be described as having multiple severities unless additional transformations are generated separately [R11].
3. Split the 10,000 held-out IDs, by original image ID, into: `calibration` (localization/basis discovery), `edit_search` (objective evaluations), and `edit_selection` (early stopping, hyperparameters, top-K rescoring). Keep the official 10,000-image test split sealed until the methods are fixed.
4. Use a small labeled adaptation budget to force sample-efficient repair, e.g. 256 corrupted examples and 512 clean preservation examples in search; reserve larger disjoint pools for calibration and selection. Keep class balance or stratify explicitly.
5. Evaluate on official corrupted test images, clean MNIST test images, and one unoptimized corruption family, fixed in `study_v1.yaml`, as a transfer/interference check.
6. **Framing.** MNIST-C's authors note that fine-tuning on a single corruption trivially recovers accuracy and that training on benchmark corruptions defeats the benchmark's out-of-distribution purpose [R11]. Task B is therefore a **few-shot adaptation study using MNIST-C images**, not an MNIST-C robustness evaluation, and every gain is reported as a fraction of the FULL-FT ceiling.

**Pairing audit.** MNIST-C does not document that its files are index-aligned with original MNIST. Take clean views from MNIST-C's own `identity/` folder; assert elementwise label equality across `identity/`, the chosen corruption and original MNIST, and pixel equality between `identity/` and raw MNIST; for the chosen non-geometric corruption, run a per-index pixel-correlation check against `identity/`. If any check fails, regenerate the corruption with the MNIST-C code under a fixed seed.

**Labeled-data policy.** S2 and B2 use labeled calibration data that data-free methods never see. Every method that does not use the calibration pool (RAND, SPEC and their null draws) appends the calibration pool's examples to its search objective, so all arms see the same labeled examples; S1 needs unlabeled pairs only. Report "labeled examples used" as its own column. Never reuse search examples for localization or basis construction.

**Dataset licensing.** MNIST-C data (Zenodo record 3239543, v2) is CC BY 4.0, which requires attribution; the generation code is Apache-2.0, and its GitHub repository has been archived (read-only) since April 2026. Pin the download to the Zenodo zip by SHA-256. MNIST's license is commonly cited as CC BY-SA 3.0, but this is not stated at the original source; do not redistribute images. Check any employment-contract restrictions separately. Prefer scripts that download data from the original source instead of committing images into the repository.

### 8.3 Mandatory pilots before the freeze

Both pilots use base-model metrics and calibration data only; neither may run a candidate method, and both are shared design work charged to no method.

1. **Headroom pilot.** Evaluate every base seed on all 15 MNIST-C corruptions using calibration IDs. Choose the non-geometric corruption with a moderate drop (roughly 5–30 pp) that is stable across seeds, freeze it, and report per-seed headroom next to \(G_T\). Do not rely on published CNN headroom numbers for this MLP.
2. **Edit-site ceiling pilot.** On the chosen corruption, run matched-mask Adam at \(W_{2[:,S]}\), \(b_1\), \(W_1\) rows of \(k\) first-layer neurons (incoming weights; supports differ in size, so report them), and \(W_3\). Surgical fine-tuning predicts that early layers are best for input-level corruptions [R19], and an additive shift can be cancelled by a \(b_1\) or \(W_1\)-row edit but not by a \(W_2\) column edit. Keep \(W_2\) as the factorial site and report its ceiling as a fraction of the best site's; if the W2 ceiling is near zero, record it and treat the study as a test of a deliberately hard site.

### Experimental split map

| Split | Used for | Forbidden use |
|---|---|---|
| Base-model training (50k MNIST-train IDs) | Learn original \(\theta_0\) | Search-method performance claims; any overlap with the pools below |
| Calibration | Support localization and basis construction | Final method ranking |
| Edit search | Candidate objective queries | Architecture / experiment redesign after test |
| Edit selection | Choose penalty, radius and safe candidate | Repeated trial-and-error on sealed test |
| Sealed test | Final performance, calibration and robustness | Any optimization, direction selection or stopping |

Record all dataset-index hashes. If data are paired across clean and corrupted variants, split by **original image ID**, not by transformed view, to prevent leakage.

---

## 9. Baselines, controls and ablations

### 9.1 Minimum valid comparison table

**Zeroth-order factorial** (primary optimizer O2 for all; random components use 200 draws per seed):

| Code | Support | Directions | Purpose |
|---|---|---|---|
| BASE | None | None | Pre-edit performance |
| RAND | Random S0 | Random B0 | Null distribution |
| RAND-U | Random S0 | Structure-matched B0-u | Structure null on random supports |
| SPEC | Random S0 | Spectral B1 | Weights-only basis |
| BEHAV | Random S0 | Behavior B2 | H1a secondary stratum |
| CONTRAST-RAND | S1 | Random B0 | Cheap noncausal support |
| CAUS-RAND | Causal S2 | Random B0 | H1a/H2 null on the causal support |
| CAUS-U | Causal S2 | Structure-matched B0-u | H1b control |
| CAUS-SPEC | Causal S2 | Spectral B1 | Localization + spectrum |
| NEUROSCULPT | Causal S2 | Behavior B2 | Proposed combined method |
| FULL-SUPPORT | S0 draws and S2 | All 512 support coordinates (sep-CMA-ES) | What prior search-based repair does [R14–R18] |

**Required baselines and ceilings** (cost reported in the same query unit):

| Code | What it does | Search queries | Role |
|---|---|---|---|
| LSQ-EDIT | Closed-form ridge least squares on \(W_{2[:,S]}\) mapping \(h_{1,S}(\tilde x)\) to the paired clean \(a_2(x)\), with a clean-covariance penalty | 0 | Forward-only, ROME-style; uses pairing (class ii) |
| MEAN-SHIFT | \(b_2 \mathrel{+}= W_2(\mathbb E[h_1(x)]-\mathbb E[h_1(\tilde x)])\) | 0 | Closed-form activation alignment |
| LOGIT-BIAS | Primary optimizer over the 10-dim \(b_3\) | ≤240 FBE | H4 class-prior control |
| LASTLAYER | Closed-form refit of \(W_3\) on the search examples with preservation | 0 | Last-layer retraining baseline [R22] |
| ADAM-MASK | Adam on the free masked coordinates, every support | — | Gradient reference; basis-free H2 test |
| FULL-FT | Adam on all parameters | — | Recoverable-gap ceiling |
| ADAM-BASIS | Adam/L-BFGS on \(z\) in each basis | — | Harness-only subspace oracle (H1c) |

For every selected edit report the mean-shift fraction \(\|\Delta W\,\bar h_S\|^2/\mathbb E\|\Delta W\,h_{1,S}\|^2\), because \(h_1\ge0\) gives every \(W_2\) edit a bias-like component. Frame Tier 2 against the best of these baselines, not only against BASE and the random null.

If the full grid is too expensive, keep at least `BASE`, `RAND`, `CAUS-RAND`, `CAUS-U`, `NEUROSCULPT`, `BEHAV`, `FULL-SUPPORT`, the zero-query baselines, `LOGIT-BIAS`, `ADAM-MASK`, `FULL-FT` and `ADAM-BASIS`, and state which comparisons are now unanswerable. Every conclusion must rest on the methods actually run.

### 9.2 Required negative and robustness controls

- **Zero-edit test:** \(z=0\) must exactly reproduce baseline metrics.
- **Norm match:** compare the Frobenius norm of candidate edits, not only the number of edited entries.
- **Support/dimension match:** methods must use identical \(k\) and \(d\) for primary claims.
- **Random-seed repetition:** ≥6 base seeds, 200 random draws per random component per seed, and 5 optimizer seeds, all preregistered.
- **Label or pair shuffling:** label shuffling of the correct/incorrect grouping is the information control for B2; pair shuffling is the control for B2-pair (it is a no-op for B2, whose group means do not use pairing). The advantage should deteriorate when the informative signal is broken.
- **Structure match:** B0-u has B2's exact rank-1 form; B2's information claim is judged against it (H1b).
- **Clean-accuracy protection:** no high-target-accuracy result counts as successful if unrelated accuracy collapses.
- **Class-wise outcomes:** inspect per-class gains and regressions to catch prediction-frequency shortcuts.
- **Query accounting:** activation-restoration, basis-discovery, radius/λ-sweep and selection-rescoring forwards count toward reported total cost.
- **H3 arm:** run the primary method with \(\lambda_{\mathrm{CE}}=\lambda_{\mathrm{KL}}=0\), once with the selection gate kept and once with it removed, at the radius where the unpenalized search measurably loses selection accuracy.

### 9.3 Limited ablations after first success

Vary \(d\in\{4,8,16\}\), \(k\in\{4,8,16\}\), optimization budget, preservation weight, and corruption type **one factor at a time**. Avoid a full factorial cross product. B2 forces \(d=k\), so the d-ablation uses B2-pair; B1's dominant-vs-trailing comparison runs at \(d=4\). If d exceeds simple SVD rank, predefine cross-mode construction. Consider a second model architecture only once the first model's conclusions are stable.

---

## 10. Evaluation metrics and statistical protocol

### 10.1 Primary outcome and preservation

For target test accuracy \(A_T\), define the edit gain in **percentage points**:

\[
G_T=100\,[A_T(\theta_{\mathrm{edit}})-A_T(\theta_0)].
\]

For preservation accuracy, define the loss:

\[
D_P=100\,[A_P(\theta_0)-A_P(\theta_{\mathrm{edit}})].
\]

Report \(G_T\), \(D_P\), target and preserve cross-entropies, preservation KL, and the frequency of `NO_IMPROVING_EDIT`. A candidate with \(G_T>0\) but severe \(D_P>0\) is not automatically better.

### 10.2 Structural and efficiency metrics

- Fraction of *nonzero changed* weights \(\|\Delta\theta\|_0/P\) under an explicit numerical tolerance; also report editable support fraction, because these differ.
- Edit size \(\|\Delta W\|_F/\|W_{0[:,S]}\|_F\) and cosine/norm overlap with known planted damage (Task A only).
- Effective edit rank, singular spectrum of \(\Delta W\), and input-output effects on held-out examples.
- Candidate evaluations, calibration/diagnostic evaluations, model-example forwards, total wall time, peak RAM/VRAM, and optimizer evaluations until a specified gain is reached.
- Pareto curves: target gain versus preservation loss; performance versus computational budget; performance versus edit norm.

### 10.3 Controlled-damage metrics

For target loss, define a repair fraction (when the denominator is meaningfully positive):

\[
R_{\mathrm{loss}}=\frac{L_T(\theta_{\mathrm{damaged}})-L_T(\theta_{\mathrm{repaired}})}
{L_T(\theta_{\mathrm{damaged}})-L_T(\theta_{\mathrm{pristine}})}.
\]

A value of 1 indicates restoration of the pristine target loss; >1 means exceeding it; <0 means worsening. Treat true injected support as **ground truth only for final diagnosis**. Include support precision/recall, but do not equate support overlap with functional recovery.

### 10.4 Statistical analysis

- Train at least **6 independent base-model seeds** (10 preferred); an MNIST MLP trains in seconds. Do not treat repeated stochastic optimization runs on one model as independent trained networks.
- For each fixed trained model, compare methods using the **same test examples**. Use paired bootstrap resampling over test examples to estimate uncertainty in accuracy differences *conditional on that model*.
- For each seed, place the structured method in the **null distribution** of its random-component counterpart (200 draws) and report its percentile. The seed-level statistic is the gain over the null median.
- Primary inference is the seed-level one-sided Wilcoxon signed-rank test and a hierarchical bootstrap (seeds, then test examples) 95% CI, with the smallest effect of interest and TOST margin of §3 (0.5 pp). Report mean, median and standard deviation of gains across seeds.
- The pre-declared contrasts are those of §3; `NEUROSCULPT` versus `RAND` is descriptive. Apply Holm correction across the H1a, H1b and H2 tests; everything else is exploratory.
- Do not select the best seed, corruption family or method using sealed-test numbers.

### 10.5 Proposed acceptance criteria (engineering targets, not guaranteed claims)

**Tier 0 — validity gate (must pass):** deterministic evaluation, correct support masking, no base-checkpoint mutation, sealed test partition, complete logs and reproducibility.

**Tier 1 — scientific completion (must pass even if hypothesis is false):** run all preregistered primary comparisons and required baselines, at least six base-model seeds, comparable budgets, uncertainty estimates and a candid result report.

**Tier 2 — positive empirical finding (aspirational):** on natural shift, `NEUROSCULPT` improves target test accuracy by at least ~1 absolute percentage point relative to the unedited model **without more than ~0.5 points clean-test accuracy degradation**; H1a passes (gain over the B0 null ≥ 0.5 pp, Wilcoxon significant after Holm correction); H1b passes against B0-u; and `NEUROSCULPT` is not beaten by the best zero-query baseline (LSQ-EDIT, MEAN-SHIFT, LASTLAYER) or by LOGIT-BIAS. These numerical cutoffs are design targets, not evidence that success is likely; revise only on a validation-based, predeclared rationale.

**Tier 3 — stronger research result (optional):** replicate improvements across a second corruption or model class, show improved Pareto frontiers and diagnose why structure helps. Do not make Tier 3 a condition for shipping the initial project.

---

## 11. Implementation architecture and interfaces

### 11.1 Suggested repository

```text
neurosculpt/
├── README.md
├── pyproject.toml
├── configs/
│   ├── study_v1.yaml
│   └── smoke.yaml
├── src/neurosculpt/
│   ├── data.py           # indices, paired views, leakage checks
│   ├── models.py         # small model + checkpoint utilities
│   ├── edits.py          # support masks, basis matrices, functional patch
│   ├── localization.py   # random/contrast/restoration supports
│   ├── directions.py     # random/spectral/behavior bases
│   ├── objectives.py     # target, preservation, KL, safe selection
│   ├── search.py         # random/coordinate/CMA-ES
│   ├── experiment.py     # fixed budgets and full provenance
│   ├── metrics.py        # performance, structure, uncertainty
│   └── plotting.py       # spectrum, Pareto, cost, seed summaries
├── scripts/
│   ├── train_base.py
│   ├── make_splits.py
│   ├── run_smoke.py
│   ├── run_study.py
│   └── summarize.py
├── tests/
│   ├── test_edit_mask.py
│   ├── test_zero_edit.py
│   ├── test_basis_orthonormal.py
│   ├── test_no_mutation.py
│   ├── test_data_leakage.py
│   ├── test_loss_and_selection.py
│   └── test_query_counter.py
├── outputs/              # ignored, machine-generated
└── docs/
    ├── research_decisions.md
    ├── experiment_protocol.md
    └── findings.md
```

### 11.2 Core interfaces

```python
@dataclass(frozen=True)
class EditSpec:
    parameter_name: str
    support_columns: tuple[int, ...]
    basis: Tensor        # [d, m, n], zero outside support, normalized
    radius: float

# Pure, no checkpoint mutation:
def make_parameter_patch(base_params, spec: EditSpec, z: Tensor): ...
def evaluate_edit(model, base_params, spec, z, batches, objective): ...
def localize_support(model, calibration_pairs, method, k, budget): ...
def construct_basis(model, calibration_pairs, support, method, d): ...
def optimize_edit(candidate_fn, algorithm, query_cap, seed): ...
def choose_safe_candidate(history, selection_batches, preserve_gate): ...
```

Keep `model.eval()` fixed and avoid dropout/batchnorm variability. Use tensors with explicit device and dtype; test CPU execution first. Prefer a functional parameter override over in-place editing. Persist a sparse patch or coefficient/basis representation, not an entire modified model unless needed for demonstration.

### 11.3 Output schema per run

Write a structured file (JSON/Parquet/CSV) containing: `study_id`, immutable config hash, base checkpoint hash, dataset-split hashes, seed, corruption, support method, basis method, information class, null-draw index, optimizer, \(d\), \(k\), labeled examples used, losses, selected candidate coefficients, target/preservation metrics, edit norm, query counts, elapsed time, device, software versions, and failure/stop reason. Store one row per candidate separately from final summaries so curves can be recomputed without rerunning searches.

### 11.4 Minimal reporting graphics

1. Target gain versus preservation drop, with methods and seeds distinguished.
2. Target test accuracy versus total model-query budget (including localization).
3. Side-by-side random/spectral/behavior basis performance on **identical supports**.
4. Per-class confusion/accuracy change and patch-norm histogram.
5. Task A: injected-versus-recovered support and target-loss restoration.

Do not substitute an attractive dashboard for a well-grounded experiment.

---

## 12. Compute, Claude Code and execution budget

### 12.1 Keep two budgets separate

**Coding-agent budget:** Claude Code tokens are consumed by planning, code generation, debugging, reading logs, and iteration. Reduce expenditure with strict acceptance tests, short scripts, modular interfaces, and a fixed experiment matrix.

**Experimental compute budget:** Optimization consumes data passes and hardware time. An inexpensive codebase can still generate enormous query workloads if diagnostics and candidate evaluations are not capped. Track both explicitly.

### 12.2 Suggested per-method cap

- End-to-end cap: **240 full-batch-equivalent evaluations** per method/seed, including localization and basis-construction forwards. One full-batch equivalent is the number of model-example forward calls needed to score the fixed target-plus-preservation adaptation batch once. Log the actual raw model-example count as the authoritative cost; full-batch equivalents are only a convenient normalized summary.
- Localization: pre-screen at most 32 columns in the initial run, on a calibration pool of 1,024 original IDs (2,048 clean+corrupt images). Deduct all activation-restoration forwards from the end-to-end allowance (S1 ≈ 2.7 FBE; S2 restoration ≈ 32 × 1,024 MEQ ≈ 42.7 FBE, so NEUROSCULPT keeps ≈ 194 FBE for search). Search-only comparisons may additionally report a separate 240-candidate limit as a diagnostic, but must not present it as equal total cost.
- Selection: top-K rescoring (K = 5) on the selection split and the shared radius/λ sweep are charged and reported; Task A localization uses the same pre-screen and is charged the same way.
- Unit: one model-example query (MEQ) counts as 1 whether it starts at the input or at \(h_1\); 1 FBE = 768 MEQ. Log analytic MACs as a secondary unit. The 240 cap is a sample-efficiency choice: 240 FBE ≈ 40 GFLOP ≈ 1 s on one CPU core uncached, and the whole study is CPU-minutes.
- Bases: construct once per run; log construction time and any forward evaluations used.
- Training: cache each clean base model once and share its checkpoint between all methods for that seed.
- A fast smoke configuration: 1 seed, 32 candidates, 64 target + 64 preservation examples, 4 columns and 4 dimensions.

The precise total full study cost is the product of methods, seeds, candidate evaluations and examples per evaluation. Have the implementation print a dry-run cost estimate before launching. Do not hardcode a GPU requirement or promise a specific wall time without a local benchmark.

### 12.3 Token-safe staged workflow

| Stage | Build | Exit gate | Explicitly avoid |
|---|---|---|---|
| P0 | Frozen models on 50k IDs, held-out pools, clean/corrupt pairing audit, target and preserve metrics, **headroom and edit-site pilots** | Base metrics reproduce; pairing audit passes; corruption and site frozen | Any search algorithm |
| P1 | Sparse patch mechanism, cached fast path, objective, zero-edit and invariance tests, cost ledger | All core unit tests green | UI |
| P2 | Random bases (B0, B0-u) + CMA-ES + top-K selection; **Task A oracle positive control**; zero-query baselines | Task A recovers planted damage on ≥1 damage family; one complete reproducible Task B run | Large parameter sweeps |
| P3 | Spectral and behavior bases, S1/S2 localization, FULL-SUPPORT, ADAM-MASK, FULL-FT, subspace oracle | Full primary comparison runs on 1 seed | Additional architectures |
| P4 | All seeds with nulls, final test, uncertainty and ablations; **[secondary]** items only after this | Tier 0 and Tier 1 pass | New algorithm families |
| P5 (optional) | Streamlit demo, second task/model, novel descent | Only if P4 supports it and budget remains | Scope creep |

**Stop rule:** If a step repeatedly fails, isolate it with a synthetic fixture and simplify the research design. Do not consume the budget chasing a fragile speculative extension. A sound negative result is preferable to a visually impressive but invalid demonstration.

---

## 13. Risks, confounders and mitigation

| Risk | Why it matters | Required mitigation |
|---|---|---|
| Natural corruptions cannot be repaired with a sparse \(W_2\) edit | Limited subspace expressivity; input-level shifts favor early layers [R19] | Mandatory edit-site ceiling pilot (§8.3); report W2 ceiling as a fraction of the best site |
| B2 wins only because it is rank 1 | Structure confounded with information | Structure-matched B0-u control (H1b) |
| Weak random null | One draw per seed is a coin flip | 200 draws per seed, percentile, seed-level test |
| Structured arms see more labeled data | Data, not structure, explains the gain | Labeled-data policy (§8); report labeled examples used |
| A cheap baseline matches the method | Gain has no practical relevance | LSQ-EDIT, MEAN-SHIFT, LASTLAYER, LOGIT-BIAS required |
| Search overfits 256 adaptation examples | Small-data optimization | Sealed test, independent selection split, preservation checks |
| Causal restoration selects important but non-editable neurons | Localization/editability gap | Explicit S0/S2 comparison; cite Hase et al. [R7] |
| Behavior contrast reflects label mix, not mechanism | Spurious direction construction | Class matching, pair shuffling, reproducible controls |
| Spectral basis wins due to norm differences | Unfair comparison | Orthonormalize, norm/radius match, same initial point |
| Hidden extra search budget | Structured methods use diagnostics | Report total forwards, latency, and diagnostic budget |
| Gradient baseline appears unfair | One backward ≠ one forward | Report quality versus wall time, examples, approximate FLOPs |
| ReLU/permutation symmetries confuse interpretation | Equivalent functions have different weights | Compare functional effects and aligned motifs, not raw coordinates |
| Selection bias through repeated test inspection | Inflated performance | Immutable test protocol and results audit trail |
| Low original error leaves little headroom | Ceiling effect | Report baseline headroom and select failure regime on calibration data |
| Dataset attribution and redistribution obligations | MNIST-C is CC BY 4.0; MNIST's license is unstated at source | Attribute MNIST-C; pin the Zenodo zip by SHA-256; do not redistribute images; check employment-contract terms separately |
| One good stochastic run is presented as a theorem | Overclaiming | 3+ seeds, confidence intervals, negative results, clear qualifiers |

---

## 14. Instructions to the reviewing and implementing frontier agent

**Your first job is to improve this specification, not to start coding immediately.** Work through the following sequence and record changes in `docs/research_decisions.md`:

1. **Literature and novelty audit.** Verify references, search for more recent direct competitors, and replace any overambitious novelty claim with a narrower falsifiable one. In particular, compare with the two separate 2026 Shapley-guided repair papers (NSRepair [R9] and SHARPEN [R17]), RepTran [R18], forward-only test-time adaptation [R21] and low-rank or projected model editing [R23, R25]. The v1.1 audit is recorded in `docs/spec_review.md`.
2. **Mathematical audit.** Check all dimensions, equation assumptions, group sparsity accounting, basis orthogonality, ReLU/nonlinearity caveats and query-budget definitions. Distinguish proven equalities from heuristics.
3. **Experimental audit.** Identify leakage paths; confirm adaptation versus selection versus test splits; ensure that causal localization never receives privileged oracle information in Task A; verify that the main comparison isolates support and basis effects.
4. **Resource audit.** Build a dry-run cost estimator. Trim tasks or variants if the estimated agent/compute budget is too high. Do **not** silently drop necessary random, no-edit, or preservation baselines.
5. **Revise the design.** Produce a short decision record with `KEEP / CHANGE / DROP`, a rationale for each material revision, and a concrete v1 experiment protocol. Be conservative with scope.
6. **Implement stage by stage.** Complete tests and stop gates before launching expensive studies. Log failures rather than hiding them.
7. **Evaluate without cherry-picking.** Freeze choices before final test. Quantify uncertainty, compare against random and gradient-based alternatives, and report null results.
8. **Write findings.** Separate **PROVEN** mathematical facts, **OBSERVED** empirical results, **HYPOTHESIZED** mechanisms, and **UNTESTED** extensions. Include limitations and replication commands.

### Non-negotiable scientific integrity rules

- All data labels, model gradients, pristine repair weights, and test performance must be used only where the declared protocol permits.
- No frontier model may act as the edit-search controller in the proposed method. Claude Code is permitted to **write and debug code**, not to inspect every candidate and hand-pick weights from test feedback.
- Do not introduce standard backpropagation into the proposed search path; use it in separately labeled comparator methods only.
- Do not claim exact causal identification from correlation or a single activation-restoration score.
- No result may be called state of the art without a fair benchmark against relevant established methods.
- Include seeds, train/test split hashes, configurations and all commands needed to reproduce tables and plots.

### Questions the agent must answer before implementing

1. Is the main failure sufficiently large and localized to make restricted edits meaningful, or should the task be altered?
2. Is \(W_2\) the right fixed edit location, or is a cheap pilot on one alternative layer justified?
3. Does behavior-informed basis construction use any labels or paired signals unavailable to controls?
4. Are loss, accuracy, KL, edit norm and preservation gates scaled and computed correctly?
5. Does the optimizer actually find a solution superior to \(z=0\) on **selection**, not only on search examples?
6. Which comparison falsifies the central hypothesis if NeuroSculpt fails?
7. What precise claim will remain valid if no method improves held-out performance?

---

## 15. Final deliverable checklist

- [ ] `README.md` with clear goal, scientific scope and honest limitations.
- [ ] Dependency-locked environment; one-command smoke run.
- [ ] Deterministic splits and audited clean/corrupt pairing.
- [ ] Frozen pretrained checkpoints and hashes.
- [ ] Zero-edit, no-mutation, mask, orthogonality, budget and leakage unit tests.
- [ ] Random nulls (B0, B0-u, S0; 200 draws per seed), structured support/basis search methods, FULL-SUPPORT, zero-query baselines, LOGIT-BIAS, and no-edit and gradient comparators (ADAM-MASK, FULL-FT, harness-only ADAM-BASIS).
- [ ] Headroom and edit-site pilot reports, and the pairing audit.
- [ ] Two documented tasks: planted-damage sanity and natural-shift main result.
- [ ] Prespecified experiment configuration, multiple independent seeds and sealed test.
- [ ] A complete table of target accuracy, preservation accuracy, KL, edit size, total query cost and runtime.
- [ ] Pareto/cost and class-wise plots, including unsuccessful runs.
- [ ] Reproducible patch artifact and a script showing before/after inference.
- [ ] `docs/findings.md` differentiating observation, hypothesis, and mathematical reasoning.
- [ ] Clear answer to H1 and H2, including the possibility of a negative answer.

### Suggested final report title

**NeuroSculpt: Do Behavior-Informed Weight Subspaces Improve Gradient-Free Neural Network Repair?**

A suitable research outcome is not merely 'our method improved accuracy.' It is a controlled answer to **whether—and under what conditions—structural information helps search a frozen model's weight space more efficiently than random directions**.

---

## References and implementation resources

**[R1]** R. Takehi, J. Teoh, A. Zhang, O. Khattab. *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights*. November 2, 2026. https://rikiyat.github.io/blog/weight-space-programming/

**[R2]** C. Li, H. Farkhoor, R. Liu, J. Yosinski. *Measuring the Intrinsic Dimension of Objective Landscapes*. ICLR 2018. https://arxiv.org/abs/1804.08838

**[R3]** S. Malladi et al. *Fine-Tuning Language Models with Just Forward Passes* (MeZO). NeurIPS 2023. https://papers.nips.cc/paper_files/paper/2023/hash/a627810151be4d13f907ac898ff7e948-Abstract-Conference.html

**[R4]** N. Hansen, A. Ostermeier. *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evolutionary Computation 9(2), 2001; and N. Hansen, *The CMA Evolution Strategy: A Tutorial*, arXiv:1604.00772. Implementation: *pycma* documentation, https://cma-es.github.io/apidocs-pycma/index.html

**[R5]** K. Meng et al. *Locating and Editing Factual Associations in GPT* (ROME). NeurIPS 2022. https://proceedings.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html

**[R6]** G. Ilharco et al. *Editing Models with Task Arithmetic*. ICLR 2023. https://mlanthology.org/iclr/2023/ilharco2023iclr-editing/

**[R7]** P. Hase et al. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models*. NeurIPS 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html

**[R8]** F. Zhang, N. Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html

**[R9]** X. Du, Z. Zhang, X. Wang, L. Liu, J. Qian. *Neural Network Repair With Shapley-Guided Search* (NSRepair). CAAI Transactions on Intelligence Technology 11(3):816–834, 2026. doi:10.1049/cit2.70147. https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70147

**[R10]** PyTorch documentation. `torch.func.functional_call`. https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html

**[R11]** N. Mu, J. Gilmer. *MNIST-C: A Robustness Benchmark for Computer Vision*. ICML 2019 Workshop on Uncertainty and Robustness in Deep Learning, arXiv:1906.02337. Data: Zenodo record 3239543 (CC BY 4.0). Code (Apache-2.0, archived): https://github.com/google-research/mnist-c

**[R12]** Y. Xiong, X. Xie. *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting During Parameter-Efficient Fine-Tuning*. AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/40703

**[R13]** A. Gupta, A. Rao, G. Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting*. Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.902/

**[R14]** B. Sun, J. Sun, L. H. Pham, J. Shi. *Causality-Based Neural Network Repair* (CARE). ICSE 2022.

**[R15]** J. Sohn, S. Kang, S. Yoo. *Arachne: Search-Based Repair of Deep Neural Networks*. ACM TOSEM 32(4), 2023.

**[R16]** M. Sotoudeh, A. V. Thakur. *Provable Repair of Deep Neural Networks* (PRDNN). PLDI 2021. arXiv:2104.04413

**[R17]** Sun et al. *Shapley-Guided Neural Repair Approach via Derivative-Free Optimization* (SHARPEN). arXiv:2604.00422, April 2026. *(Full author list to be completed.)*

**[R18]** *RepTran*. ISSRE 2026. arXiv:2607.11193. *(Full title and authors to be completed.)*

**[R19]** Y. Lee, A. S. Chen, F. Tajwar, A. Kumar, H. Yao, P. Liang, C. Finn. *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts*. ICLR 2023. arXiv:2210.11466

**[R20]** C. Eastwood et al. *Source-Free Adaptation to Measurement Shift via Bottom-Up Feature Restoration* (BUFR). ICLR 2022. arXiv:2107.05446

**[R21]** S. Niu et al. *Test-Time Model Adaptation with Only Forward Passes* (FOA). ICML 2024. arXiv:2404.01650

**[R22]** P. Kirichenko, P. Izmailov, A. G. Wilson. *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations* (DFR). ICLR 2023.

**[R23]** S. Santurkar et al. *Editing a Classifier by Rewriting Its Prediction Rules*. NeurIPS 2021.

**[R24]** T. Sun et al. *Black-Box Tuning for Language-Model-as-a-Service* (BBT). ICML 2022; and *BBTv2*, EMNLP 2022.

**[R25]** J. Fang et al. *AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models*. ICLR 2025.

**Reference caveat:** These sources establish context and methodological precedents; they do not collectively establish that NeuroSculpt's proposed method will work. Check current links, software compatibility, and relevant licensing before implementation.
