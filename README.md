# TF-nnPU collision-risk reproducibility

Processed simulation data, released model weights, checkpoint evaluation, and opt-in training workflows for **Robust Ego-Centric Collision Risk Estimation Under Positive–Unlabeled Supervision**.

Version **0.1.0** reproduces the supplied checkpoint and provides runnable TF, cost-sensitive, Su-LSTM, GAT, and CMPA training workflows. See [method details](docs/METHOD.md) and [experiment provenance](docs/EXPERIMENTS.md) before comparing different runs.

## Install (Python 3.11)

Extract or clone the repository and open a terminal in this folder.

Windows PowerShell, CPU installation (activation is unnecessary):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name tf-nnpu --display-name "TF-nnPU (Python 3.11)"
.\.venv\Scripts\python.exe -m tf_nnpu verify
.\.venv\Scripts\python.exe -m tf_nnpu evaluate --device cpu
.\.venv\Scripts\python.exe -m jupyterlab
```

In JupyterLab, select the **TF-nnPU (Python 3.11)** kernel and open `notebooks/01_main_tf_nnpu_reproduction.ipynb`. The notebooks find the repository from either its root or the `notebooks/` directory.

For NVIDIA CUDA 12.1, use the `https://download.pytorch.org/whl/cu121` index instead of `/cpu` in the torch installation command. Select `--device cuda` or the default `auto`. See the [official PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/#v251). The local validation environment used Python 3.11.3 and PyTorch 2.5.1+cu121; see [validation records](docs/VALIDATION.md) for the actual checks performed.

On Linux, create the environment with `python3.11 -m venv .venv` and replace `.\.venv\Scripts\python.exe` in the commands with `.venv/bin/python`. macOS installation was not tested.

## Quick reproduction

With the environment's Python on your PATH, run from the repository root:

```sh
python -m tf_nnpu verify
python -m tf_nnpu evaluate --device cpu
```

The first command verifies file hashes, both splits, and all 16 corruption masks. The second checks the released checkpoint against these targets:

| Accuracy | AUROC | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0.960684 | 0.990302 | 0.938095 | 0.895455 | 0.916279 |

Confusion matrix `[TN FP; FN TP]`: `[[2048, 39], [69, 591]]`. The target is the supplied observed PU endpoint label, not an independently annotated latent frame-risk label.

Evaluation saves keyed predictions, metrics, and environment metadata in a new timestamped `outputs/` directory. Hash mismatches and unexpected metrics stop execution. Ordinary floating-point differences are allowed up to `1e-6` in reported scalar metrics.

## View results or train

Notebooks 02-06 display recorded results by default. They do not train models, fit scalers, or build caches unless `RUN_TRAINING=True`. Full reruns are explicit:

```sh
python -m experiments.run gat
python -m experiments.run stage2 --train
python -m experiments.run sensitivity --train
python -m experiments.run cost_sensitive --train
python -m experiments.run sulstm --train
python -m experiments.run gat --train
python -m experiments.run cmpa --train
```

For a brief execution check, append `--smoke`, for example `python -m experiments.run cmpa --train --smoke`. Smoke runs use one epoch and reduced samples; their scores are not paper results. Full training may take hours, depending on hardware. CMPA full training generates a roughly 228 MiB cache under its output directory; no simulator is required.

Every training run writes new checkpoints, predictions, settings, and result tables under `outputs/`. Generated files never overwrite the released checkpoints or `results/` references. The original cost-sensitive dataset filename and historical RNG/checkpoint provenance could not all be established: new reruns are labeled accordingly rather than promised to reproduce their historical scores exactly.

## Contents and scope

- `data/`: 981 scenarios, 15 frames each; [data dictionary](data/README.md).
- `checkpoints/`: original encoder, final model, expected metrics.
- `splits/`, `noise_masks/`: fixed sample membership and corruption selections, with format documentation.
- `tf_nnpu/`, `experiments/`: evaluation, models, training, artifact checks, and paired statistics.
- `notebooks/`: six portable entry points.
- `results/`: original reference CSVs, additional tables recovered from saved notebook outputs, and the newly verified main-checkpoint predictions.
- `archive/`: historical research code as Markdown reference documents. It includes experiments requiring additional inputs and should not be executed as a complete workflow.
- `docs/`: methodology, provenance, release changes, and validation.

Physical-QCar feature sequences, event/frame metadata, raw sensor recordings, YOLO weights, and QLabs scene/control assets were not supplied. This release does not claim to regenerate those experiments. See [remaining inputs](docs/MISSING_ASSETS.md).

## Citation, licensing, and contact

Use [CITATION.cff](CITATION.cff) to cite the software and associated manuscript. No DOI or journal acceptance information is invented. Original software is under the [MIT license](LICENSE); supplied data, weights, and numerical result artifacts are under [CC BY 4.0](DATA_LICENSE.md). Third-party software retains its own license.

Author contact: Daryel Israel Leon Cachott, `daryel.leon@hotmail.com`. After publication on GitHub, issues can be used for reproducibility questions.
