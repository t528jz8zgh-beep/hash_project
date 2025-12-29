# Report snippets (auto-generated)
## What is being compared
- We report both **insertion cost** (insert probes/time) and **successful-query probe complexity** (hit probes).
- For Elastic hashing, successful-query probe complexity is reported as the paper-defined index **φ(i,j)**; for Uniform/Funnel, hit probes correspond to the number of probes performed in the search procedure.
## Results: alpha sweep (load sensitivity)
- Highest load in this dataset: **alpha=0.98**.
- Figure references:
  - Fig.1: `fig1_alpha_hit_probes_p99.png`
  - Fig.2: `fig2_alpha_insert_probes_p99.png`
  - Fig.3: `fig3_alpha_insert_ns_per_op.png`
  - Diagnostic: `diag_alpha_inserted_frac.png` (must be close to 1.0 for valid same-load comparison)

**Important:** Elastic did not reach the target load in at least the highest-alpha setting (see `diag_alpha_inserted_frac.png` and `insert_fail_rate`). In that case, Elastic-vs-(Uniform/Funnel) comparisons at the same (n, delta) are not valid; rerun after fixing Elastic.

## Results: n sweep (scalability)
- Largest n in this dataset: **n=1048576** (delta fixed).
- Figure references:
  - Fig.4: `fig4_n_hit_probes_p99.png`
  - Fig.5: `fig5_n_insert_ns_per_op.png`
  - Diagnostic: `diag_n_inserted_frac.png`

## Discussion pointers (template)
- Discuss **hit_probes_p99** as the main metric aligned with the paper’s probe-complexity definition for present keys.
- Discuss **insert_probes_p99** and **insert_ns/op** as the engineering cost trade-off.
- Use diagnostics to confirm each algorithm actually achieved the intended load.
