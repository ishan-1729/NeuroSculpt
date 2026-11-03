# Research decisions

Decision record required by spec §14. Every material change to the design, and every deviation
from `configs/study_v1.yaml` after it is frozen, gets an entry here.

**Statuses:** KEEP, CHANGE, DROP. **Phase:** pre-registration (before freeze) or deviation (after).

| # | Date | Item (spec §) | Status | Decision | Rationale | Phase |
|---|---|---|---|---|---|---|
| 1 | 2026-11-03 | Repository scaffold (§11.1) | KEEP | Layout follows the spec; all modules are stubs | Design review pending, so no logic is committed before it | pre-registration |

## Open review items

The multi-lens review of the spec (literature, math, design, mechanism, feasibility) is in
progress. Its accepted findings go into the table above as CHANGE/DROP entries, each with a rationale.

## Pre-implementation questions (spec §14)

1. Is the main failure large and localized enough to make restricted edits meaningful?
2. Is \(W_2\) the right fixed edit location, or is a pilot on one alternative layer justified?
3. Does behaviour-informed basis construction use labels or pairings unavailable to controls?
4. Are loss, accuracy, KL, edit norm and preservation gates scaled and computed correctly?
5. Does the optimizer beat \(z=0\) on **selection**, not only on search examples?
6. Which comparison falsifies the central hypothesis if NeuroSculpt fails?
7. What precise claim remains valid if no method improves held-out performance?
