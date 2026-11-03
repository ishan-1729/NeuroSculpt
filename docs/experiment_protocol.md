# Experiment protocol (v1)

**STATUS: DRAFT.** It becomes binding when `configs/study_v1.yaml` is frozen. Source: spec §3, §8–§10, §12.

## 1. Splits

| Split | Used for | Forbidden use |
|---|---|---|
| Base-model training | Learn \(\theta_0\) | Search-method performance claims |
| Calibration | Support localization, basis construction | Final method ranking |
| Edit search | Candidate objective queries | Redesign after test |
| Edit selection | Penalties, radius, safe-candidate choice | Trial-and-error on sealed test |
| Sealed test | Final metrics, scored once | Any optimization, selection or stopping |

- Split by **original image ID** so clean and corrupted views of one digit stay together.
- Record SHA-256 hashes of every index list in each run's output.

## 2. Tasks

- **Task A, planted damage:** the injected columns, direction and pristine weights are visible only to the harness.
- **Task B, natural shift:** one MNIST-C corruption is fixed before any results are seen. Evaluate on corrupted test,
  clean test and one corruption that was not optimized.

## 3. Budget accounting

- Raw model-example forwards are the authoritative cost. Full-batch equivalents are only a summary.
- Localization and basis-construction forwards are charged against the end-to-end budget.
- Print a dry-run cost estimate before every study launch.

## 4. Selection and outcomes

- Safe selection: among candidates that pass the preservation gate on edit selection, take the lowest target CE,
  with smallest \(\|\Delta W\|_F\) breaking ties. If no candidate qualifies, record `NO_SAFE_EDIT`.
- Primary outcome: \(G_T\) (target gain, pp) with \(D_P\) (preservation drop, pp). The predeclared primary
  comparison is `NEUROSCULPT` vs `RAND`.
- Uncertainty: paired bootstrap over test examples within each model, then summary across base seeds.

## 5. Freeze checklist

- [ ] Corruption family, seeds, splits and hashes fixed
- [ ] Method list, budgets and radius grid fixed
- [ ] Metric definitions and primary comparison fixed
- [ ] `study_v1.yaml` set `frozen: true` and the commit hash recorded here
