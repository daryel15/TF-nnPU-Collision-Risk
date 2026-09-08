# Local validation, 2026-09-08

Environment: Windows, Python 3.11.3, PyTorch 2.5.1+cu121. Direct package versions used for these checks are pinned in `requirements.txt`.

| Check | Result |
|---|---|
| Main checkpoint, CPU | PASS: all five metrics and confusion-matrix assertions |
| Main checkpoint, CUDA through a fresh notebook kernel | PASS: all expected values |
| All six release notebooks, fresh kernels with `notebooks/` as working directory | PASS |
| Active Python syntax | PASS |
| Dataset, both splits, and all 16 mask selections | PASS |
| File manifest | PASS after final regeneration |
| Stage-2 training from the released encoder, one epoch / reduced samples | PASS |
| Fresh pretraining and sensitivity fine-tuning, one configuration / one epoch | PASS |
| Cost-sensitive nnPU/WBCE/Focal training and paired statistics, smoke mode | PASS |
| Su-LSTM training, smoke mode | PASS |
| TF/GAT-BCE/GAT-nnPU training, smoke mode | PASS |
| TF/CMPA-BCE/CMPA-nnPU training with rasterized DSM inputs, smoke mode | PASS |
| Exported smoke-run prediction keys/probabilities | Unique keys, matching lengths, probabilities in [0,1] |

The main checkpoint yields accuracy 0.9606843829632327, AUROC 0.990302159109059, precision 0.9380952380952381, recall 0.8954545454545455, and F1 0.9162790697674419. Its predictions are included in `results/main_checkpoint_predictions.csv`. This is the verified released artifact, not a newly trained replacement.

Smoke tests run the training and output-saving code paths but do not validate the reported scientific performance of newly trained baselines. Full-length baseline/sensitivity retraining, physical evaluation, raw acquisition, and a fresh dependency installation were not performed. Historical baseline statistics remain identified as reference results. The pinned direct dependencies are a record of the working environment, not a fully resolved cross-platform lock file.

Default notebook checks took roughly 3-15 seconds per notebook in this environment. These observations are usability checks, not model-latency benchmarks. Full training durations depend on hardware and selected experiments; the original complete runs were not timed during this packaging task.

To repeat validation:

```sh
python -m tf_nnpu verify
python -m tf_nnpu evaluate --device cpu
python tools/check_release.py
python -m experiments.run stage2 --train --smoke
python -m experiments.run sensitivity --train --smoke
python -m experiments.run cost_sensitive --train --smoke
python -m experiments.run sulstm --train --smoke
python -m experiments.run gat --train --smoke
python -m experiments.run cmpa --train --smoke
```

Test outputs and caches are excluded from the distribution ZIP and clean upload folder. Future runs create new `outputs/` directories. The GitHub workflow repeats artifact verification, CPU evaluation, and default notebook execution; it has been provided but has not been executed on GitHub yet.
