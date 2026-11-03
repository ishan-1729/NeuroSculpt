# NeuroSculpt

**Structured, Gradient-Free Weight-Space Search for Targeted Neural Network Repair**  
**Research and Implementation Specification — v1.0**  
**Prepared:** 3 November 2026  
**Status:** Proposed research design; not an implemented or experimentally validated system  
**Audience:** Research scientist / frontier coding agent reviewing, revising, and implementing the project

> **Mission.** Test whether behavior-informed, low-dimensional edits to a frozen neural network can repair a specific failure more efficiently than matched random edits, while preserving unrelated behavior. Discover when this works, when it fails, and which structural assumptions explain the outcome.

> **Interpretation rule.** Every numerical budget and acceptance threshold in this document is a *proposed experimental-design choice*, not a reported result. The implementing agent may revise choices with a written rationale; it must not present hypotheses as findings.

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
3. Several fixed-support / fixed-dimension search bases, a simple zeroth-order optimizer and a matched random control.
4. Preservation-aware selection and an honest held-out evaluation with uncertainty.
5. Reproducible scripts, unit tests, compact plots, results tables, an editable weight patch, and a technical report.

### 1.4 Default scope (adjustable only with justification)

| Item | Default | Why |
|---|---|---|
| Model | MLP: 784 → 128 → 64 → 10, ReLU, no batch normalization | Fast, interpretable, supports exact layer interventions |
| Base data | MNIST | Small, standardized, easy to diagnose |
| Shift task | One MNIST-C corruption, with train/validation adaptation and sealed official test | Genuine non-planted adaptation rather than artificial recovery |
| Edited matrix | Second-layer weight matrix \(W_2 \in \mathbb R^{64\times128}\) | Edits the mapping from first-layer features into second-layer features |
| Editable support | \(k=8\) selected input-neuron columns of \(W_2\) | Exactly 512 candidate matrix entries, before low-dimensional restriction |
| Search dimension | \(d=8\) coefficients | Tractable zeroth-order optimization |
| Main optimizers | Coordinate search or CMA-ES, plus random-search control | Small implementation and understandable comparisons |
| Main budget | Up to ~240 full-batch-equivalent evaluations per method/seed, including diagnostic calls in end-to-end comparisons | Bounded search; predefine before experiments |
| Seeds | 3 independent base-model seeds for MVP; 5 if feasible | Avoid reporting a lucky single run |
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
- **Neural network repair:** Shapley-guided fault localization plus multi-objective repair has been explored in 2026 [R9]. NeuroSculpt must be positioned as a focused comparative study of **search subspaces and edit efficiency**, not as the first locate-then-repair framework.

### 2.3 Precise candidate contribution

The intended result is an **empirical method or finding** about the utility of structured weight-edit subspaces: under identical data, support, dimensionality and evaluation budgets, do spectral and behavior-guided bases yield better target/preservation trade-offs than random bases? Separately, does causal localization of activations yield better editable supports than random or activation-contrast selection? The project should also identify failure regimes, not only successes.

A rigorous negative result is acceptable. For example: 'Causal localization significantly predicts output sensitivity but does not outperform randomly selected editable columns' would be scientifically informative if demonstrated carefully.

---

## 3. Research questions and falsifiable hypotheses

| ID | Question / hypothesis | What would falsify it? |
|---|---|---|
| H1 (primary) | At fixed support, parameter count, search dimension and query budget, behavior-informed directions outperform an orthonormal random-direction baseline on held-out repair gain. | Similar or worse test gain across seeds and tasks. |
| H2 | At fixed basis type and query budget, columns chosen through paired causal activation-restoration outperform random columns as editable supports. | No statistically or practically meaningful advantage. |
| H3 | Preservation penalties and constraints meaningfully improve the target-versus-retention Pareto frontier. | Edited models show equivalent retention without the preservation terms. |
| H4 | Improvements on a natural shift cannot be explained solely by overfitting the adaptation samples or by changing predicted-class frequencies. | Improvement disappears on sealed test and independent corruption samples. |
| H5 (optional) | Useful low-dimensional directions exhibit cross-seed stability after appropriate alignment or functional comparison. | Directions fail to transfer or have no stable functional signature. |

**Primary hypothesis for the MVP:** H1. H2 and H3 are secondary. H4 is a required validity check. H5 is an extension; it must not delay the initial result.

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

All terms should be scaled to comparable magnitudes using the baseline scores; tune their coefficients on the **selection** split, never the final test. An alternative implementation is an augmented penalty for exceeding an accuracy-loss tolerance, but avoid making noisy, discontinuous accuracy the sole search signal.

**Selection rule:** Among search candidates satisfying a prespecified preservation-accuracy gate on the selection split, select the one with the lowest target cross-entropy (secondary tie-breaker: smallest \(\|\Delta W\|_F\)). If none satisfies the gate, retain the original model and record `NO_SAFE_EDIT`. Do not hide the failure by cherry-picking a candidate on test data.

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

Choose 8 columns uniformly without replacement, with a recorded seed. This must use the same matrix, cardinality, and downstream optimizer as the structured methods.

### S1. Activation-contrast support (cheap, noncausal)

On *paired clean and corrupted inputs* \((x,\tilde x)\), compute per-neuron differences in the first hidden layer:

\[
c_j=\mathbb E_{(x,\tilde x)}\big|h_{1,j}(x)-h_{1,j}(\tilde x)\big|.
\]

Rank columns by \(c_j\), optionally normalized by activation scale. This identifies sensitivity to corruption but does **not** establish a neuron causes failure. Compare against random S0.

### S2. Causal activation-restoration support (main structured method)

For a corrupted sample \(\tilde x\), run the model normally and then re-run while replacing one first-layer activation \(h_{1,j}(\tilde x)\) with \(h_{1,j}(x)\) from its paired clean input. Define the estimated repair effect:

\[
r_j=\mathbb E\left[
\ell_{\mathrm{CE}}(\theta_0;\tilde x,y)
-\ell_{\mathrm{CE}}(\theta_0;\tilde x,y\mid h_{1,j}\leftarrow h_{1,j}(x))
\right].
\]

Positive \(r_j\) means this activation restoration helps *under this intervention and test distribution*; it does not imply editing the corresponding outgoing column will help. Pre-screen the top ~32 activation-contrast candidates for expensive activation-restoration tests, then select 8 columns by mean \(r_j\) on a **separate calibration split**. Include wrong-pair or random-activation restoration controls if budget permits. Charge all these diagnostic forwards to the total experiment cost.

**For controlled planted damage:** Paired clean/corrupt activation restoration is not defined in the same way because the fault lies in the *weights*, not the input. Use a declared **S2-damage ablation variant** instead: for labeled samples, intervene on first-layer neuron \(j\) by setting \(h_{1,j}\leftarrow0\) in the damaged model, and rank candidate columns by the resulting improvement in target cross-entropy minus any measured preservation degradation. This uses only the damaged model and permitted labeled calibration data; the pristine weights and true injected support remain hidden. Label this as a different localization intervention, and do not directly pool its scores with Task B restoration scores.

### Localization validity warnings

- Restoring an activation is not the same operation as changing its outgoing weights; report both intervention scores and actual edit success.
- Strong source/target activation differences can reflect benign covariate shift rather than a failure circuit.
- Rank support candidates on calibration data only; do not optimize or rank on final-test examples.
- Do not use a layer-scan that accidentally grants structured methods far more model queries than random controls without reporting the difference.

---

## 6. Candidate edit directions: how may the selected weights change?

The project should compare at least the following three bases **on the same supports**. This factorial structure separates 'where to edit' from 'which way to edit.'

### B0. Isotropic random orthonormal basis (mandatory)

Draw \(d\) Gaussian vectors in the \(mk\)-dimensional supported coordinate space, apply QR, embed the resulting unit vectors into \(W_2\). Fix the basis at the start of a run; do not resample it on every candidate evaluation. Use the same \(d\), support and maximum Frobenius step radius as other methods.

### B1. Spectral basis (weight-space geometry)

Compute \(W_{2[:,S]}=U\Sigma V^\top\) on the supported submatrix. Construct rank-one directions \(u_i v_i^\top\), embedded back into the full matrix. For \(d\le\operatorname{rank}(W_{2[:,S]})\), use the first \(d\) singular pairs. If exploring \(d\) above the available rank, use prespecified orthogonal cross-products \(u_i v_j^\top\) and describe the rule explicitly. Compare dominant and trailing singular modes in an ablation: high singular value does not imply high editability.

### B2. Behavior-informed basis (proposed)

Estimate an output-space direction at the edited layer from paired examples, using the *second hidden layer* \(h_2\). One simple no-gradient contrast is:

\[
u=\operatorname{normalize}\left(
\mathbb E[h_2(\tilde x)\mid\text{incorrect}]
-\mathbb E[h_2(\tilde x)\mid\text{correct, class-matched}]
\right).
\]

For each chosen first-layer input column \(s_j\), define \(\widetilde B_j=u\,e_{s_j}^\top\). Orthogonalize and normalize the directions. Class matching and enough examples are essential; otherwise apparent signal may be caused by class balance. If the contrast is unstable or degenerate, log this and fall back to a declared control, **not** to an unreported hand-picked basis. A second version may use clean-versus-corrupt hidden-state differences rather than correct-versus-incorrect grouping.

**Important:** For a ReLU layer, an edit in preactivation space is not guaranteed to produce the same shift as the measured post-ReLU contrast. This is an empirical hypothesis, not an algebraic identity.

### Suggested primary comparisons (compact 2 × 3 factorial)

| Support \ Basis | Random B0 | Spectral B1 | Behavior B2 |
|---|---:|---:|---:|
| Random S0 | Required reference | Required | Required |
| Causal S2 | Required | Required | **Primary candidate** |

Activation-contrast S1 is an inexpensive additional support baseline if resources allow. Avoid building all possible support/basis/optimizer combinations before obtaining the first interpretable result.

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

### 7.2 Optimizers

**O0 — Random search:** sample \(z\) in a bounded \(d\)-ball, including \(z=0\). Mandatory sanity control.  
**O1 — Coordinate / pattern search:** test \(z\pm\eta e_i\), accept the better candidate, reduce \(\eta\) on stagnation. Easy to audit and strong at \(d=8\).  
**O2 — CMA-ES:** use an established implementation with seeded sampling, a bounded search radius and the same candidate-evaluation cap. CMA-ES adapts a distribution over edit coefficients; it requires **objective evaluations**, not gradients [R4].

Do not make the main experiment depend on perfectly tuning CMA-ES. Choose O1 as the default if O2 is unstable or overly expensive. Report actual forward-equivalent cost, not just iteration count.

### 7.3 Search bounds and candidate selection

Initialize \(z=0\). Bound \(\|\Delta W(z)\|_F\le\rho\), where \(\rho\) is a fraction of \(\|W_{2[:,S]}\|_F\) and is selected on the validation/selection split. A reasonable *starting sweep* is \(\rho/\|W_{2[:,S]}\|_F\in\{0.01,0.03,0.10\}\); the values must be treated as tunable engineering guesses, not established optimums.

All methods receive the same radius candidates. For **search-only** comparisons, match optimization-query limits and report diagnostic overhead separately. For **end-to-end efficiency** claims, count diagnostic forwards and grant every method the same total model-example-forward budget, reducing a structured method's search allowance accordingly. Candidate selection uses the preservation gate and target-validation score; final held-out tests are scored **once after all decisions are frozen**.

### 7.4 Gradient-based reference baseline

Include a *matched-mask* Adam baseline that updates the same eligible \(W_2\) columns while freezing all other parameters. Optionally include an Adam-in-the-same-basis control, which optimizes \(z\) using backprop; this isolates whether the basis is good even when gradients are available.

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

1. Train three small MLPs independently on MNIST clean-training data. Record clean validation/test performance and all seeds.
2. Duplicate each model and inject a fixed, localized perturbation into a small set of \(W_2\) columns. Tune perturbation strength **only on calibration data** to produce a measurable, well-defined failure without collapsing all predictions.
3. Hide injected support, perturbation direction and pristine parameter values from all editing methods. Only the experiment harness may access that ground truth for *post hoc* recovery metrics.
4. Repair using the same labeled adaptation budget allowed to every method; evaluate recovered target performance, collateral damage, support overlap and functional similarity on held-out examples.

Potential failure design: select a few columns influential for a prespecified digit subset, perturb them to degrade accuracy on those digits while retaining substantial performance on the remaining digits. If such localized selective damage is hard to produce, document it and use a less selective but objectively measurable corruption; do not hand-tune injections against the test set.

**Why this task exists:** Known injected damage lets us validate the weight-edit pipeline and causality diagnostics. **It does not prove usefulness for naturally occurring model failures.**

### Task B: natural corruption-shift repair (primary scientific result)

1. Train the clean MNIST model and freeze it.
2. Choose **one** MNIST-C corruption family (e.g. translation or shot noise) *before* inspecting final results. Use paired clean/corrupted images from the allowed training partition for calibration and search. MNIST-C contains a fixed corruption set; it should not be described as having multiple severities unless additional transformations are generated separately [R11].
3. Separate disjoint training-origin indices into: `calibration` (localization/basis discovery), `edit_search` (objective evaluations), and `edit_selection` (early stopping and hyperparameters). Keep the official 10,000-image test split sealed until the methods are fixed.
4. Use a small labeled adaptation budget to force sample-efficient repair, e.g. 256 corrupted examples and 512 clean preservation examples in search; reserve larger disjoint pools for calibration and selection. Keep class balance or stratify explicitly.
5. Evaluate on official corrupted test images, clean MNIST test images, and at least one unoptimized corruption family as a transfer/interference check.

**Dataset licensing:** verify the actual MNIST/MNIST-C source and license terms before using or redistributing data in compensated work. Prefer scripts that download data from its original source instead of committing images into the repository.

### Experimental split map

| Split | Used for | Forbidden use |
|---|---|---|
| Base-model training | Learn original \(\theta_0\) | Search-method performance claims |
| Calibration | Support localization and basis construction | Final method ranking |
| Edit search | Candidate objective queries | Architecture / experiment redesign after test |
| Edit selection | Choose penalty, radius and safe candidate | Repeated trial-and-error on sealed test |
| Sealed test | Final performance, calibration and robustness | Any optimization, direction selection or stopping |

Record all dataset-index hashes. If data are paired across clean and corrupted variants, split by **original image ID**, not by transformed view, to prevent leakage.

---

## 9. Baselines, controls and ablations

### 9.1 Minimum valid comparison table

| Code | Support | Directions | Optimizer | Purpose |
|---|---|---|---|---|
| BASE | None | None | None | Pre-edit performance |
| RAND | Random S0 | Random B0 | O1 or O2 | True gradient-free reference |
| SPEC | Random S0 | Spectral B1 | Same as RAND | Isolate basis geometry |
| BEHAV | Random S0 | Behavior B2 | Same as RAND | Isolate behavior direction |
| CAUS-RAND | Causal S2 | Random B0 | Same as RAND | Isolate localization |
| CAUS-SPEC | Causal S2 | Spectral B1 | Same as RAND | Combine localization + spectrum |
| NEUROSCULPT | Causal S2 | Behavior B2 | Same as RAND | Proposed combined method |
| ADAM-MASK | Same mask as chosen method | Free masked coordinates | Adam | Gradient-based quality/cost reference |

If the full 2 × 3 grid is too expensive, prioritize `BASE`, `RAND`, `BEHAV`, `CAUS-RAND`, `NEUROSCULPT`, `ADAM-MASK`; state which comparisons are now unanswerable. Every conclusion must rest on the methods actually run.

### 9.2 Required negative and robustness controls

- **Zero-edit test:** \(z=0\) must exactly reproduce baseline metrics.
- **Norm match:** compare the Frobenius norm of candidate edits, not only the number of edited entries.
- **Support/dimension match:** methods must use identical \(k\) and \(d\) for primary claims.
- **Random-seed repetition:** vary base-model initialization, support generation and optimizer seed in a preregistered way.
- **Label or pair shuffling:** behavior-guided advantage should deteriorate when informative pairings are broken, if it truly uses the intended information.
- **Clean-accuracy protection:** no high-target-accuracy result counts as successful if unrelated accuracy collapses.
- **Class-wise outcomes:** inspect per-class gains and regressions to catch prediction-frequency shortcuts.
- **Query accounting:** activation-restoration and basis-discovery forwards count toward reported total cost.

### 9.3 Limited ablations after first success

Vary \(d\in\{4,8,16\}\), \(k\in\{4,8,16\}\), optimization budget, preservation weight, and corruption type **one factor at a time**. Avoid a full factorial cross product. If d exceeds simple SVD rank, predefine cross-mode construction. Consider a second model architecture only once the first model's conclusions are stable.

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

Report \(G_T\), \(D_P\), target and preserve cross-entropies, preservation KL, and the frequency of `NO_SAFE_EDIT`. A candidate with \(G_T>0\) but severe \(D_P>0\) is not automatically better.

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

- Train at least **3 independent base-model seeds** for the MVP. Prefer 5 for stronger claims; do not treat repeated stochastic optimization runs on one model as independent trained networks.
- For each fixed trained model, compare methods using the **same test examples**. Use paired bootstrap resampling over test examples to estimate uncertainty in accuracy differences *conditional on that model*.
- Separately report mean, median and standard deviation of gains across independent model seeds. With only three seeds, uncertainty is large: label the result exploratory and avoid overstated significance claims.
- If running a larger study, use a hierarchical bootstrap or mixed-effects analysis that respects grouping by base-model seed and test example. Predeclare the primary pairwise comparison `NEUROSCULPT` versus `RAND`.
- Do not select the best seed, corruption family or method using sealed-test numbers. Multiple benchmark comparisons are exploratory unless multiplicity is addressed.

### 10.5 Proposed acceptance criteria (engineering targets, not guaranteed claims)

**Tier 0 — validity gate (must pass):** deterministic evaluation, correct support masking, no base-checkpoint mutation, sealed test partition, complete logs and reproducibility.

**Tier 1 — scientific completion (must pass even if hypothesis is false):** run all preregistered primary comparisons, at least three base-model seeds, comparable budgets, uncertainty estimates and a candid result report.

**Tier 2 — positive empirical finding (aspirational):** on natural shift, `NEUROSCULPT` improves target test accuracy by at least ~1 absolute percentage point relative to the unedited model **without more than ~0.5 points clean-test accuracy degradation**, and consistently improves over `RAND` at the same edit/query budget on most seeds. These numerical cutoffs are design targets, not evidence that success is likely; revise only on a validation-based, predeclared rationale.

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

Write a structured file (JSON/Parquet/CSV) containing: `study_id`, immutable config hash, base checkpoint hash, dataset-split hashes, seed, corruption, support method, basis method, optimizer, \(d\), \(k\), losses, selected candidate coefficients, target/preservation metrics, edit norm, query counts, elapsed time, device, software versions, and failure/stop reason. Store one row per candidate separately from final summaries so curves can be recomputed without rerunning searches.

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
- Localization: pre-screen at most 32 columns in the initial run. Deduct all activation-restoration forwards from the end-to-end allowance. Search-only comparisons may additionally report a separate 240-candidate limit as a diagnostic, but must not present it as equal total cost.
- Bases: construct once per run; log construction time and any forward evaluations used.
- Training: cache each clean base model once and share its checkpoint between all methods for that seed.
- A fast smoke configuration: 1 seed, 32 candidates, 64 target + 64 preservation examples, 4 columns and 4 dimensions.

The precise total full study cost is the product of methods, seeds, candidate evaluations and examples per evaluation. Have the implementation print a dry-run cost estimate before launching. Do not hardcode a GPU requirement or promise a specific wall time without a local benchmark.

### 12.3 Token-safe staged workflow

| Stage | Build | Exit gate | Explicitly avoid |
|---|---|---|---|
| P0 | Frozen model, clean/corrupt data splits, target and preserve metrics | Reproduce base metrics, clean/test separation | Any search algorithm |
| P1 | Sparse patch mechanism, objective, zero-edit and invariance tests | All core unit tests green | UI, CMA-ES |
| P2 | Random basis + coordinate search + preserved selection | At least one complete reproducible run | Large parameter sweeps |
| P3 | Spectral and behavior bases, causal restoration | Full primary comparison runs on 1 seed | Additional architectures |
| P4 | 3 seeds, final test, uncertainty and ablations | Tier 0 and Tier 1 pass | New algorithm families |
| P5 (optional) | Streamlit demo, second task/model, novel descent | Only if P4 supports it and budget remains | Scope creep |

**Stop rule:** If a step repeatedly fails, isolate it with a synthetic fixture and simplify the research design. Do not consume the budget chasing a fragile speculative extension. A sound negative result is preferable to a visually impressive but invalid demonstration.

---

## 13. Risks, confounders and mitigation

| Risk | Why it matters | Required mitigation |
|---|---|---|
| Natural corruptions cannot be repaired with a sparse \(W_2\) edit | Limited subspace expressivity | Treat as result; optionally run a predeclared layer-swap diagnostic |
| Search overfits 256 adaptation examples | Small-data optimization | Sealed test, independent selection split, preservation checks |
| Causal restoration selects important but non-editable neurons | Localization/editability gap | Explicit S0/S2 comparison; cite Hase et al. [R7] |
| Behavior contrast reflects label mix, not mechanism | Spurious direction construction | Class matching, pair shuffling, reproducible controls |
| Spectral basis wins due to norm differences | Unfair comparison | Orthonormalize, norm/radius match, same initial point |
| Hidden extra search budget | Structured methods use diagnostics | Report total forwards, latency, and diagnostic budget |
| Gradient baseline appears unfair | One backward ≠ one forward | Report quality versus wall time, examples, approximate FLOPs |
| ReLU/permutation symmetries confuse interpretation | Equivalent functions have different weights | Compare functional effects and aligned motifs, not raw coordinates |
| Selection bias through repeated test inspection | Inflated performance | Immutable test protocol and results audit trail |
| Low original error leaves little headroom | Ceiling effect | Report baseline headroom and select failure regime on calibration data |
| Data terms permit research but not compensated work | Licensing / compliance risk | Validate license and employment contract; avoid redistributing data |
| One good stochastic run is presented as a theorem | Overclaiming | 3+ seeds, confidence intervals, negative results, clear qualifiers |

---

## 14. Instructions to the reviewing and implementing frontier agent

**Your first job is to improve this specification, not to start coding immediately.** Work through the following sequence and record changes in `docs/research_decisions.md`:

1. **Literature and novelty audit.** Verify references, search for more recent direct competitors, and replace any overambitious novelty claim with a narrower falsifiable one. In particular, compare with 2026 Shapley-guided neural repair, zero-order adaptation and low-rank model editing.
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
- [ ] At least a random and a structured support/basis search method plus no-edit and gradient comparators.
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

**[R4]** N. Hansen et al. *pycma: CMA-ES Python implementation*. Documentation. https://cma-es.github.io/apidocs-pycma/index.html

**[R5]** K. Meng et al. *Locating and Editing Factual Associations in GPT* (ROME). NeurIPS 2022. https://proceedings.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html

**[R6]** G. Ilharco et al. *Editing Models with Task Arithmetic*. ICLR 2023. https://mlanthology.org/iclr/2023/ilharco2023iclr-editing/

**[R7]** P. Hase et al. *Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models*. NeurIPS 2023. https://papers.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html

**[R8]** F. Zhang, N. Nanda. *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. ICLR 2024. https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html

**[R9]** *Neural Network Repair With Shapley-Guided Search* (NSRepair). CAAI Transactions on Intelligence Technology, 2026. https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.70147

**[R10]** PyTorch documentation. `torch.func.functional_call`. https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html

**[R11]** Google Research. *MNIST-C: Corrupted MNIST benchmark*. Dataset and generation code. https://github.com/google-research/mnist-c

**[R12]** Y. Xiong, X. Xie. *OPLoRA: Orthogonal Projection LoRA Prevents Catastrophic Forgetting During Parameter-Efficient Fine-Tuning*. AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/40703

**[R13]** A. Gupta, A. Rao, G. Anumanchipalli. *Model Editing at Scale leads to Gradual and Catastrophic Forgetting*. Findings of ACL 2024. https://aclanthology.org/2024.findings-acl.902/

**Reference caveat:** These sources establish context and methodological precedents; they do not collectively establish that NeuroSculpt's proposed method will work. Check current links, software compatibility, and relevant licensing before implementation.
