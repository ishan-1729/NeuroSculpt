# NeuroSculpt

**What is reading the weights worth? A ground-truth, budget-matched test of LLM weight-space repair.**

> **Status: pre-implementation.** The research design is in
> [NeuroSculpt_Research_and_Implementation_Spec.md](NeuroSculpt_Research_and_Implementation_Spec.md).
> No method here has been implemented or validated yet, and nothing in this repository is a reported
> result. All budgets and thresholds are proposed design choices.

## Question

Recent work reports that large language models can read raw neural-network weights and write edits
that repair the network. NeuroSculpt tests that claim under conditions where it can fail cleanly:

- **Small "readable" target networks.** A 49 → 24 → 16 → 10 MLP on 7×7 MNIST, small enough that its
  full weights fit in an LLM's context as text.
- **Planted defects with an exactly known repair** \(\Delta^*\), including a "stealth" defect that
  leaves every W2-only weight statistic unchanged, with swaps balance-matched so that a cross-layer
  norm heuristic barely helps.
- **One metered data channel.** Every method, LLM or search, reaches data only through a counted
  `evaluate(patch)` API.

We measure whether a pinned LLM with white-box weight access beats the strongest black-box arm at
the same query budget, and how many queries that access is worth (the exchange rate \(Q^*\), in
metered-data terms only; LLM sessions cost far more compute, and both costs are reported). The
black-box arms never read the weights; they include CMA-ES variants and a discrete template search
that uses index-only operators (swap, sign-flip, zero) available to every arm. Then we test where any
advantage comes from:

- withholding the weights, or substituting a decoy network's weights;
- free compute on synthetic inputs, and cheap weight statistics, including a cross-layer
  norm-balance heuristic;
- random rescaling and a canonical form of the weights. Every session also sees freshly permuted
  hidden units;
- removing semantic priors;
- handing the LLM's chosen support or extracted edit template to search.

A careful negative result is a full result.

| ID | Hypothesis | Role |
|---|---|---|
| H1 | At 64 FBE, LLM-with-weights closes ≥ 10 pp more of the recoverable gap than the best black-box arm | Primary |
| H2 | Weight access is worth more than 128 queries at a 64-query budget (\(Q^*(64) > 128\)) | Confirmatory |
| H3 | The gain needs the true weights, beats data-free heuristics, and survives without free synthetic compute | Confirmatory |
| H4 | It is robust to rescaling, and beats trial swaps and heuristics on a balance-matched stealth defect | 4a estimate, 4b confirmatory |
| H5 | Weights still help after semantic stripping | Confirmatory |
| H6 | How much of the gain search recovers from the LLM's support or template | Estimate |
| H7 | Weight-defect vs input-shift advantage | Descriptive |
| H8 | At matched gain, LLM edits are closer to the true repair (canonicalized) | Estimate |

## Background

An earlier draft asked whether structured low-dimensional edit bases beat random bases for
gradient-free repair. No code was written for it. Its [review](docs/spec_review.md) found the question
narrow, and a [judged brainstorm](docs/research_direction_brainstorm.md) chose the current question.
The draft is kept for reference in [docs/archive/earlier_draft_spec.md](docs/archive/earlier_draft_spec.md),
and its reviewed methodology is reused here.

## Repository layout

```text
configs/            study.yaml (pre-registration, DRAFT until frozen), smoke.yaml, archive/
src/neurosculpt/    models, data, edits, damage, reparam, meter, llm_harness, templates, search, sandbox/,
                    localization, exchange, objectives, experiment, metrics, plotting
prompts/            versioned LLM prompts per arm
scripts/            train_base, make_splits, run_smoke, run_study, summarize
tests/              zero-edit, no-mutation, mask, reparameterization exactness, leakage, selection, meter
docs/               research_decisions.md, experiment_protocol.md, findings.md, spec_review_2.md (this spec),
                    spec_review.md (earlier draft), research_direction_brainstorm.md, archive/
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

Planned smoke run with a mock LLM and no API calls. The script is a stub until stage P0 lands:

```bash
python scripts/run_smoke.py --config configs/smoke.yaml
```

LLM API keys are read from environment variables and are never committed or logged.

## Staged plan

| Stage | Build | Exit gate |
|---|---|---|
| P0a (week 1) | Readable MLP, splits, damage generator, reparameterization, meter, data-free heuristics | Exactness tests pass; clean accuracy ≥ 90% |
| P0b (week 2) | Sandbox, LLM harness, template extractor, mock-LLM smoke run | Adversarial leakage test passes |
| P1 (week 3) | Black-box curves to 4,096 FBE, baselines and ceilings on calibration seeds | Headroom gate holds; comparator and envelope frozen |
| P2 (week 3) | LLM pilot of every prompt template on calibration seeds | Parse rate ≥ 90% per template; cap set; preregistration frozen and posted |
| P3 (weeks 4–5) | ~1,380 sessions in randomized order; sealed test opened once | Primary analysis exactly as preregistered |
| P4 (week 6) | Decomposition, canonicalized repair metrics, write-up | Every hypothesis reported, regardless of sign |

## Integrity rules

- Every arm reaches data only through the metered API. Gradient methods are labelled ceilings only.
- LLM arms run as automated, preregistered sessions. No human or coding agent steers sessions or picks
  edits from evaluation feedback. Claude Code writes and debugs code only.
- Ground truth (\(\Delta^*\), clean weights) is visible only to the evaluation harness.
- The sealed test is scored **once**, after `configs/study.yaml` is frozen. Deviations are recorded
  in [docs/research_decisions.md](docs/research_decisions.md).
- Write-ups label each statement as **PROVEN**, **OBSERVED**, **HYPOTHESIZED** or **UNTESTED**.

## Data

MNIST and MNIST-C are downloaded by script from their original sources and never committed.
MNIST-C data is CC BY 4.0 (attribution required); its generation code is Apache-2.0.

## License

Code and documentation in this repository are released under the [MIT License](LICENSE).
Datasets keep their own licenses.
