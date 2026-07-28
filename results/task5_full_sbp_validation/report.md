# Task 5 full-SBP validation

## Key outcomes
- `full-raw` and `full-orth` remained time-history equivalent to roundoff on all tested small multi-step cases.
- Mass drift stayed at roundoff level for both projected and full-SBP variants in the `tf=1` study.
- In the `tf=1` convergence study, projected cases had smaller final relative L2 errors than full-orth for all 12 scheme combinations tested here.
- Product-rule residuals were smaller for `full-orth` than `projected`, but both variants showed approximately fifth-order decay in this implementation.
- Long-time energy behavior depended more on form and flux than on projected/full choice: central conservative showed mild growth, while upwind/LF were mildly dissipative.

## Main artifacts
- `summary.csv`
- `summary.json`
- `raw_orth_equivalence/equivalence.csv`
- `mass/mass.csv`
- `stability/stability.csv`
- `product_rule/projected_product_rule.csv`
- `product_rule/full-orth_product_rule.csv`
- `stability/energy_ratio_conservative_ndiv4.png`
- `stability/energy_ratio_split_ndiv4.png`

## Convergence tables
- `convergence/table1_full-orth_conservative_central_convergence.csv`
- `convergence/table1_full-orth_conservative_lf_convergence.csv`
- `convergence/table1_full-orth_conservative_upwind_convergence.csv`
- `convergence/table1_full-orth_split_central_convergence.csv`
- `convergence/table1_full-orth_split_lf_convergence.csv`
- `convergence/table1_full-orth_split_upwind_convergence.csv`
- `convergence/table1_projected_conservative_central_convergence.csv`
- `convergence/table1_projected_conservative_lf_convergence.csv`
- `convergence/table1_projected_conservative_upwind_convergence.csv`
- `convergence/table1_projected_split_central_convergence.csv`
- `convergence/table1_projected_split_lf_convergence.csv`
- `convergence/table1_projected_split_upwind_convergence.csv`

## Scope statement
The nonzero constant-state RHS was not investigated or modified in this task, as requested.
