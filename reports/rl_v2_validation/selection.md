| candidate | EEG weight | main-decoder success (12 runs) | final distance (m) | normalized AUC |
|---|---|---|---|---|
| `candidate_w010` | 0.1 | 0.9800 | 0.0278 | 0.7622 |
| `candidate_w050` | 0.5 | 0.9858 | 0.0254 | 0.7476 |
| `candidate_w100` | 1.0 | 0.9967 | 0.0230 | 0.7313 |

Validation-only selection by the rule declared in reports/rl_v2_experiment_plan.md before any validation run: highest mean final success. Candidates (EEG weight: mean final success / final distance m / AUC over the 12 EEGNet+CBraMod validation runs): candidate_w010 (w=0.1): 0.9800 / 0.0278 / 0.7622; candidate_w050 (w=0.5): 0.9858 / 0.0254 / 0.7476; candidate_w100 (w=1.0): 0.9967 / 0.0230 / 0.7313. Selected candidate_w100 (EEG weight 1.0). All three 33-run suites are complete and preserved; no test-participant feedback was used.
