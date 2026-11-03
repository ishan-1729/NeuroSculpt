# Experiment protocol

**STATUS: DRAFT.** It becomes binding when `configs/study.yaml` is frozen and posted at the end of P2.
Source: spec §3–§8 and §10.

## 1. Splits

| Split | Used for | Forbidden use |
|---|---|---|
| `base_train` (50k MNIST-train IDs) | Learn the clean model | Any overlap with the pools below |
| `calibration` (4,000 held-out IDs, two halves) | A: damage severity, affected classes. B: \(\lambda\), labeled arms | Final method ranking |
| `adaptation` (1,000 held-out IDs, behind the API) | All `evaluate` calls | Direct access by any arm |
| `selection` (5,000 held-out IDs, ≥ 500 per class) | Selection ledger and gate | Trial-and-error on the sealed test |
| `sealed_test` (official MNIST test) | Final metrics, scored once | Any optimization, selection or stopping |

- Split by **original image ID**. Record SHA-256 hashes of every index list in each run's output.
- Calibration seeds (100–102) absorb all pilots. Analysis seeds (0–9) are untouched until the P2 freeze.

## 2. Tasks

- **Task A, planted damage in \(W_2\):** families F1–F4 at uniformly sampled sites (any contiguity); F4 swaps only
  balance-matched units (≥ 20 eligible sets per seed). Severity targets a 25 pp affected-class drop on calibration seeds
  and moves only to make \(Q^*\) identifiable. Ground truth is visible
  only to the harness. Of n ≥ 2 affected classes, ceil(n/2) are in the adaptation pool and the rest are held out for H8.
- **Task B, exogenous shift (descriptive):** one non-geometric MNIST-C corruption chosen by the headroom rule on
  calibration data; LLM-W and LLM-noW at 64 FBE only.

## 3. Information parity and accounting

- Every arm gets the same disclosure (architecture, layer hint, family library but not the family, per-class accuracy)
  and the same API.
- One `evaluate` call = 1 FBE. Localization and hybrid searches count in \(b\). Selection rescoring uses a separate
  ledger, identical for every arm.
- Every arm may use harness-applied index-only operators on \(W_2\) (column-set permutation, row-set × column-set sign
  flip, column-set zero). Only pure black-box arms (no weight access) can be the primary comparator.
- LLM-W-family arms get a sandbox code tool with a 600 CPU-second budget; LLM-W-nosynth has no code tool.
- Billed tokens, turns, wall-clock and dollars are logged per session and reported as Pareto axes.

## 4. Selection and outcomes

- Each arm submits exactly 3 final candidates (padded with zero patches), rescored on the selection split.
- Gate: Task A, pooled unaffected-class accuracy drop ≤ 1 pp; Task B, clean drop ≤ 0.5 pp. Both use a one-sided 95%
  paired-bootstrap bound.
- Take the highest raw selection accuracy on in-pool affected classes among passing candidates. If none beats
  \(\Delta=0\) by the margin, record `NO_IMPROVING_EDIT`.
- Failed, refused, truncated and infrastructure-failed sessions score \(R=0\) and stay in every analysis.
- **Primary contrast:** LLM-W (M1) minus the primary comparator at 64 FBE on Task A, as sealed-test \(R\) on in-pool
  affected classes.
- **Statistics:** the seed is the unit; exact one-sided Wilcoxon over 10 seeds; family-stratified seed-level BCa CI;
  SESOI 10 pp; Holm across H2, H3a, H3b, H3c, H4b, H5 (one p-value each, H5 gatekept on H3a); H4a, H6, H7 and H8
  are estimation-only.

## 5. Freeze checklist

- [ ] LLM settings (model IDs, effort/thinking, max_tokens, API version, dates), number format and prompts pinned
- [ ] Damage severities, F4 balance threshold, Task B corruption, comparator and envelope fixed from calibration seeds
- [ ] \(\Delta^*\) and CMA-ORACLE pass the gate in ≥ 95% of calibration cells
- [ ] Per-template billed cost measured; spending cap set
- [ ] Power check done; seed count final
- [ ] Literature search re-run
- [ ] `study.yaml` set `frozen: true`, the commit hash recorded here, and the preregistration posted publicly
