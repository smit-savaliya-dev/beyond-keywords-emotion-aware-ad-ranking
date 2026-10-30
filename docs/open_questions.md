# Open questions for the student

Each item lists the conservative option that was chosen so that work could continue. Change it if you disagree.

| ID | Date | Question | Chosen (conservative) option | Alternatives |
|---|---|---|---|---|
| Q01 | 2026-10-26 | Should the shuffled-user control (`*_userspecific`, D13) replace `*_personal` as the headline congruence measure? | Report both; headline the stricter `*_userspecific`, because it does not overstate personalisation | Headline the pre-declared `*_personal` only |
| Q02 | 2026-10-26 | Should M5 also be run with ALL 19 item candidates as a robustness variant (in case univariate day-to-day instability under-states value inside a tree model)? | Not part of v1; if added, it must be pre-declared before M5 runs and reported as a variant | Run v1 only |
| Q03 | 2026-10-26 | go_pct_* percentiles are ranked over all 65,238 items, including dev-only items (label-free, transductive) | Kept: no labels or clicks are involved, so it is not leakage; noted as a limitation | Rank over train-split items only |
| Q04 | 2026-10-30 | At t = 0.05, 75.0% of items labelled emotional have a classifier neutral score above 0.5, and 'joy' is 51% approval, 'anger' 60% disapproval. Is this acceptable as a reporting label? | Kept as pre-declared, used for reporting and validation only (model features stay soft). The hand labels decide: spec v2 reverts to the old labels if its kappa CI lies entirely below them | Raise t; move approval / disapproval out of joy / anger (needs a new mapping and a new pre-declaration) |
| Q05 | 2026-10-30 | go_intensity flips sign between fit and valid. Should M5 still try it? | Not in the M5 list; the column is kept | Pre-declare it as a separate M5 variant before M5 runs |
| Q06 | 2026-10-30 | emotion_examples_v2.csv shows model labels. Reading it before hand labelling could colour your judgement, even though it excludes your 180 items | Recommended order: finish the 180 labels first, then read the examples | Read the examples now |
