# NeuroSculpt

**Structured, gradient-free weight-space search for targeted neural network repair.**

> **Status: pre-implementation.** The research design is in
> [NeuroSculpt_Research_and_Implementation_Spec.md](NeuroSculpt_Research_and_Implementation_Spec.md)
> and is under review. No method here has been implemented or validated yet, and nothing in this
> repository is a reported result. All budgets and thresholds are proposed design choices.

## Question

Given a frozen trained model and a specific failure, can a small, structured edit
\(\Delta W(z) = \sum_j z_j B_j\) found **without loss gradients** repair the failure better than an
equally sized *random* edit, under the same query budget, while leaving unrelated behaviour intact?

The contribution being tested is **subspace construction**, meaning where to edit (support) and
which way (basis). It is not zeroth-order optimisation, model editing or locate-then-repair as such,
since all three are established areas. A carefully run negative result counts as a valid outcome.

| ID | Hypothesis | Role |
|---|---|---|
| H1 | At fixed support, dimension and budget, behaviour-informed bases beat random orthonormal bases on held-out repair gain | Primary |
| H2 | Causally localised columns beat random columns as editable supports | Secondary |
| H3 | Preservation terms improve the target-vs-retention Pareto frontier | Secondary |
| H4 | Gains survive the sealed test and are not class-frequency shifts | Required validity check |
| H5 | Useful directions are stable across seeds | Optional extension |

## Default setting

- **Model:** MLP 784 → 128 → 64 → 10, ReLU, no batch norm, trained on MNIST.
- **Edit site:** second-layer weight matrix \(W_2 \in \mathbb R^{64\times128}\), \(k=8\) columns, \(d=8\) coefficients.
- **Task A (sanity):** planted, hidden damage to a few \(W_2\) columns.
- **Task B (main):** one MNIST-C corruption, with the official test split kept sealed.
- **Methods:** a 2 × 3 grid of supports (random S0, causal S2) × bases (random B0, spectral B1, behaviour B2),
  coordinate search / CMA-ES with a random-search control, and masked Adam as a gradient-based reference.

## Repository layout

```text
configs/            study_v1.yaml (pre-registration, DRAFT until frozen), smoke.yaml
src/neurosculpt/    data, models, edits, localization, directions, objectives,
                    search, experiment, metrics, plotting
scripts/            train_base, make_splits, run_smoke, run_study, summarize
tests/              zero-edit, no-mutation, mask, orthonormality, leakage, selection, query counter
docs/               research_decisions.md, experiment_protocol.md, findings.md
outputs/  data/     machine-generated, git-ignored
```

## Setup

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -e ".[dev]"
```

```bash
pytest
```

Planned entry points. They are stubs until their stage lands:

```bash
python scripts/make_splits.py --config configs/smoke.yaml
```

```bash
python scripts/run_smoke.py --config configs/smoke.yaml
```

## Staged plan

| Stage | Build | Exit gate |
|---|---|---|
| P0 | Frozen model, clean/corrupt splits, target and preserve metrics | Base metrics reproduce; test split separated |
| P1 | Sparse patch, objective, zero-edit and invariance tests | Core unit tests green |
| P2 | Random basis + coordinate search + safe selection | One complete reproducible run |
| P3 | Spectral and behaviour bases, causal restoration | Full primary comparison on 1 seed |
| P4 | All seeds, final test, uncertainty, ablations | Tier 0 and Tier 1 pass |
| P5 | Optional demo, second task or model | Only if P4 supports it |

## Integrity rules

- The proposed search path never uses autograd. Gradient methods are separately labelled baselines.
- The sealed test is scored **once**, after every choice in `configs/study_v1.yaml` is frozen.
- In Task A, the injected support and pristine weights are visible only to the evaluation harness.
- No language model chooses or tunes edits from test feedback. Claude Code writes and debugs code only.
- Every deviation from the pre-registration is recorded in [docs/research_decisions.md](docs/research_decisions.md).
- Write-ups label each statement as **PROVEN**, **OBSERVED**, **HYPOTHESIZED** or **UNTESTED**.

## Data

MNIST and MNIST-C are downloaded by script from their original sources and never committed.
Check their license terms before any redistribution or compensated use.
