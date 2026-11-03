# Research decisions

Decision record required by spec §13. Every material change to the design, and every deviation
from `configs/study.yaml` after it is frozen, gets an entry here.

**Statuses:** KEEP, CHANGE, DROP. **Phase:** pre-registration (before freeze) or deviation (after).

| # | Date | Item (spec §) | Status | Decision | Rationale | Phase |
|---|---|---|---|---|---|---|
| 1 | 2026-11-03 | Repository scaffold (§9.1) | KEEP | Package layout with stubs; only design-independent pieces implemented (functional patches, query counter, metrics, reparameterizations) | No logic is committed before the design is reviewed | pre-registration |
| 2 | 2026-11-03 | Project objective (§1) | CHANGE | Replace the earlier draft's basis-search question with a ground-truth, budget-matched test of LLM weight-space repair (H1–H8) | The earlier question was narrow and close to SHARPEN, RepTran, CARE and Arachne ([spec_review.md](spec_review.md)); the [judged brainstorm](research_direction_brainstorm.md) ranked this direction first (21/30) | pre-registration |
| 3 | 2026-11-03 | Primary target model (§1.4) | CHANGE | Readable 49-24-16-10 MLP on 7×7 MNIST; the 784-128-64-10 MLP becomes a P4 sandbox replication (superseded by row 15: now an UNTESTED extension) | Full weights must fit in an LLM context as text | pre-registration |
| 4 | 2026-11-03 | LLM edit controller (§1.2, §12) | CHANGE | LLMs act as edit controllers only in the preregistered LLM arms; Claude Code never steers sessions or picks edits | The LLM is the object of study | pre-registration |
| 5 | 2026-11-03 | Reused methodology (§4–§7) | KEEP | From the reviewed earlier draft: query accounting, 200-draw nulls, held-out pools and pairing audit, top-K selection with a one-sided bootstrap gate, `NO_IMPROVING_EDIT`, cheap baselines and gradient ceilings, float64 numerics, sealed test, Wilcoxon/TOST/Holm template | These fixes address confounds the review found; they still apply | pre-registration |
| 6 | 2026-11-03 | Discrete repairs (§4.2, §5.1, §6.2) | CHANGE | Disclose the family library to every arm; add TEMPLATE-SEARCH as a comparator-eligible arm; 25 pp severity target, moved within 15–40 pp only to make Q* identifiable | No black-box arm could propose "swap back" or "flip back" ([spec_review_2.md](spec_review_2.md) design/F01, critical) | pre-registration |
| 7 | 2026-11-03 | Stealth defect (§4.4, §5.1, §6.3) | CHANGE | Narrow the PROVEN claim to W2-only statistics; add BALANCE-MATCH; sample F4 swaps among balance-matched units; gate BALANCE-MATCH ≤ 20% | A cross-layer norm-balance heuristic finds unmatched swaps (reproduced by a reviewer; critical) | pre-registration |
| 8 | 2026-11-03 | Presentation controls (§4.3, §6.1) | CHANGE | Fresh random permutation of both hidden layers for every LLM-W session; drop LLM-W-perm; rename renorm to canonical form; uniform non-contiguous damage sites | Permutation is mean-zero under exchangeability; rescale-then-renormalize equals renormalize; contiguity is a text-only cue | pre-registration |
| 9 | 2026-11-03 | Hints and free compute (§4.2) | CHANGE | `per_class_acc` for every arm plus CMA-PCA; synthetic forwards capped at 1e5 and isolated by LLM-W-nosynth; H1 explicitly includes free weight-only compute | Unequal exploitation of shared hints; unmetered proxy-data channel | pre-registration |
| 10 | 2026-11-03 | Exchange rate (§4.5) | CHANGE | Per-budget envelope comparator; per-cell Q* with PAVA, log2 interpolation, censoring and KM median; grid from q = 1; metered-data value only, cost reported separately | Single-comparator, curve-of-medians estimator was fragile; query matching is not cost matching | pre-registration |
| 11 | 2026-11-03 | Selection and gate (§4.6) | CHANGE | Separate identical selection ledger outside b; pooled unaffected-class gate; Δ* and CMA-ORACLE must pass in ≥ 95% of calibration cells; selection in raw accuracy; failures score R = 0 | b = 0 arms were ill-defined; the per-class gate rejected exact repairs; avoid survivorship bias | pre-registration |
| 12 | 2026-11-03 | Hypotheses and statistics (§3, §7.2) | CHANGE | Seed as unit, 10 analysis + 3 calibration seeds; H2 threshold aligned at 128; H5 as stripped W vs stripped noW; H6 bands with fresh 64 FBE; H7 descriptive; H8 canonicalized and gain-matched; Holm family defined | Pseudo-replication over cells; misaligned or undefined falsifiers | pre-registration |
| 13 | 2026-11-03 | Repair metrics (§4.1, §4.4, §7.1) | CHANGE | Canonicalize edits over the h2 symmetry orbit and flat directions; add h2 agreement; affected-class rule and denominator floor | Δ* is one of a continuum of exact repairs | pre-registration |
| 14 | 2026-11-03 | Controls (§5.3, §6.1, §6.2) | CHANGE | Decoy for wrongW defined; add LLM-noW-strip; deterministic template extractor; labeled arms ineligible as comparator | Undefined or analyst-dependent controls | pre-registration |
| 15 | 2026-11-03 | Harness and cost (§6.1, §8, §10) | CHANGE | Pin effort/thinking instead of temperature; billed-token accounting with caching; identical caps; sandbox spec; infrastructure-failure policy; every template piloted; session table of 1,380; ~6-week timeline; 784-MLP replication moved to UNTESTED | Temperature is rejected by current APIs; unique-token estimates were 10x low; P0 was overloaded | pre-registration |
| 16 | 2026-11-03 | References (§2, refs) | CHANGE | Replace R8 with real symmetry sources; correct R1, R3, R11 descriptions; add R20–R26; complete R2, R4–R7, R13 metadata | R8 was unrelated; R1/R3/R11 were misdescribed | pre-registration |
| 17 | 2026-11-03 | Consistency-check fixes (§3–§6, config, code) | CHANGE | Index-only operators for every arm; pure black-box vs grey-box arms (only pure arms comparator-eligible); operator library matches the disclosed families; TEMPLATE-SEARCH as budgeted random-order enumeration; sandbox CPU-time budget and no code tool for nosynth; pairwise F4 eligibility with ≥ 20 sets and ELIGIBLE-ENUM; H4b at matched budget; one p-value per Holm member (m = 6), H5 gatekept, H8 estimation-only; pooled KM median for Q*; exactly 3 candidates per arm; split sizes; canonical layer order; per-entry significant digits | Template patches needed θ0 values; the compute cap was unenforceable; eligibility leaked the swap site; H2/H8 lacked p-values; the free best-of-3 favoured the LLM ([spec_review_2.md](spec_review_2.md), consistency check) | pre-registration |

## Open items

- Every † reference in spec §2.2 still needs hand verification before any write-up; RepTran's metadata is incomplete.
- Rows 6–16 come from the adversarial review of this spec on 2026-11-03 (57 of 58 findings survived verification); row 17
  from the follow-up consistency check. Both are recorded in [spec_review_2.md](spec_review_2.md).

## Pre-implementation questions (spec §13)

1. Does the readable MLP reach ≥ 90% clean accuracy with area averaging, and how many tokens do its weights take?
2. Does the P1 headroom gate hold within the severity band, including the lower bounds that make \(Q^*\) identifiable?
3. Can balance-matched F4 swap sets be found for every seed at the preregistered threshold?
4. Do \(\Delta^*\) and CMA-ORACLE pass the gate in ≥ 95% of calibration cells?
5. What billed cost per session does P2 measure, and is the cap affordable?
6. Which comparison falsifies H1, and what precise claim remains valid if it fails?
