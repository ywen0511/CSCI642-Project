| candidate | EEG weight | selection metric (12 runs) | final success | final distance (m) | normalized AUC | preferred pass |
|---|---|---|---|---|---|---|
| `v3_validation_w050` | 0.5 | 0.7551 | 0.9858 | 0.0276 | 0.7551 | n/a |
| `v3_validation_w100` | 1.0 | 0.7130 | 0.9242 | 0.0288 | 0.7130 | n/a |
| `v3_validation_w200` | 2.0 | 0.6422 | 0.9683 | 0.0229 | 0.6422 | n/a |

Validation-only selection by the rule declared in FINAL_V3_HANDOFF.md Section 6.3 (protocol v3) before any validation run of this suite family: highest mean normalized success AUC. Candidates (EEG weight: selection_auc / final success over the 12 EEGNet+CBraMod validation runs): v3_validation_w050 (w=0.5): 0.7551 / 0.9858; v3_validation_w100 (w=1.0): 0.7130 / 0.9242; v3_validation_w200 (w=2.0): 0.6422 / 0.9683. Selected v3_validation_w050 (EEG weight 0.5). All candidate suites are complete and preserved; no test-participant feedback was used.
