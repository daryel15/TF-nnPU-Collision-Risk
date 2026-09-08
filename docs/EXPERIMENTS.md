# Experiment and result map

| Runnable entry | Reference files in results/ | Target / input | Reproduction status |
|---|---|---|---|
| Notebook 01 / `tf_nnpu evaluate` | main_tf_nnpu_metrics.csv, main_checkpoint_predictions.csv | s_t, lengths 2-15, supplied final model | Exact checkpoint evaluation with assertions |
| Notebook 02 / sensitivity | hyperparameter_sensitivity.csv | s_t; varying model/context/LR | Saved table extracted; opt-in isolated reruns generate new logs |
| Notebook 03 / cost_sensitive | cost_sensitive_symmetric_noise_f1.csv, statistical_significance.csv | s_t; symmetric corruption | Historical tables; new seeded training uses the supplied dataset explicitly |
| Notebook 04 / sulstm | su_lstm_selected_results.csv, su_lstm_full.csv | Train on Y; evaluate on s_t and separately Y; lengths 4-15 | Historical tables; reruns export scaler/checkpoint/predictions |
| Notebook 05 / gat | gat_p_to_u_results.csv, gat_clean.csv, gat_all_noise_f1.csv, gat_all_conditions.csv | s_t; lengths 4-15 | Clean and both noise modes available for viewing and rerunning |
| Notebook 06 / cmpa | cmpa_p_to_u_results.csv, cmpa_clean.csv, cmpa_all_noise_f1.csv, cmpa_all_conditions.csv | s_t; lengths 4-15 | Clean and both noise modes available for viewing and rerunning |
| Historical calibration | calibration_by_prefix.csv, calibration_summary.csv | Derived scenario outcome Y, historical calibration model | Reference tables only; distinct model/protocol from main checkpoint |

The F1 values 0.916279 (released checkpoint), 0.9018 (cost-sensitive clean run), and 0.932677 (common-endpoint TF comparison) refer to different runs/settings and must not be combined as one checkpoint result. Check manuscript figure/table numbering against the final manuscript; it is not frozen in this release.

Original result CSVs are copied unchanged. Additional tables are recovered from saved HTML notebook outputs; their provenance is in `results/table_provenance.json`. These exports retain displayed precision. `*_all_conditions.csv` records noisy-condition F1/degradation; clean results are in `*_clean.csv`. Missing full-precision historical metrics are not filled with estimates. The clean notebooks view the supplied schemas directly instead of looking for unavailable CSV filenames.

## Statistical reproduction

Historical significance tables depend on matched validation predictions that were not supplied as files. Those predictions cannot be reconstructed uniquely from aggregate F1 scores. The reference table is included as reported, not newly verified.

New cost-sensitive training saves all keyed validation predictions and automatically computes the original paired-prefix bootstrap/permutation analysis. To recompute a comparison from two prediction files:

```sh
python -m tf_nnpu.statistics path/to/nnpu_predictions.csv path/to/baseline_predictions.csv
```

Keys and labels must match. The procedure uses 10,000 bootstrap and permutation draws, seeds 42 and 123, respectively. It quantifies paired variation at the prefix level, not uncertainty across independent scenario splits or independent training runs. Smoke mode uses 100 draws and is only an execution check.

## Historical sources

`archive/` retains the original code and commentary as Markdown. Invalid pasted table rows have been converted to comments. It is historical reference material: optional raw-input and exploratory sections are not invoked by the runnable notebooks. No unavailable pretrained-encoder aliases or physical data are fabricated.
