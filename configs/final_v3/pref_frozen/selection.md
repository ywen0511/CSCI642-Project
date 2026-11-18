| candidate | EEG weight | selection metric (12 runs) | final success | final distance (m) | normalized AUC | preferred pass |
|---|---|---|---|---|---|---|
| `pref_validation_w100` | 1.0 | 0.0517 | 0.9792 | 0.0235 | 0.5052 | 0.0517 |
| `pref_validation_w200` | 2.0 | 0.0533 | 0.9867 | 0.0228 | 0.4371 | 0.0533 |
| `pref_validation_w400` | 4.0 | 0.0483 | 0.8325 | 0.0536 | 0.2569 | 0.0483 |

Validation-only selection by the rule declared in reports/final_v3/preference_plan.md (route preference) before any validation run of this suite family: highest mean final preferred-pass rate. Candidates (EEG weight: selection_preferred_pass / final success over the 12 EEGNet+CBraMod validation runs): pref_validation_w100 (w=1.0): 0.0517 / 0.9792; pref_validation_w200 (w=2.0): 0.0533 / 0.9867; pref_validation_w400 (w=4.0): 0.0483 / 0.8325. Selected pref_validation_w200 (EEG weight 2.0). All candidate suites are complete and preserved; no test-participant feedback was used.
