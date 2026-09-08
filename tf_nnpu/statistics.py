"""Original paired prefix resampling, usable with exported prediction CSVs."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
def paired_bootstrap_f1(
    y_true,
    y_pred_a,
    y_pred_b,
    n_boot=10000,
    seed=42,
):
    """
    Paired bootstrap for:
        Delta F1 = F1(method A) - F1(method B)

    The SAME validation indices are resampled for both methods,
    preserving the paired comparison and the notebook's original split.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred_a = np.asarray(y_pred_a).astype(int)
    y_pred_b = np.asarray(y_pred_b).astype(int)

    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("The paired prediction arrays must have identical length.")

    rng = np.random.default_rng(seed)
    n = len(y_true)

    f1_a = f1_score(y_true, y_pred_a, zero_division=0)
    f1_b = f1_score(y_true, y_pred_b, zero_division=0)
    observed_delta = f1_a - f1_b

    bootstrap_deltas = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)

        boot_f1_a = f1_score(
            y_true[idx],
            y_pred_a[idx],
            zero_division=0,
        )
        boot_f1_b = f1_score(
            y_true[idx],
            y_pred_b[idx],
            zero_division=0,
        )

        bootstrap_deltas[b] = boot_f1_a - boot_f1_b

    ci_low, ci_high = np.percentile(
        bootstrap_deltas,
        [2.5, 97.5],
    )

    return {
        "f1_a": float(f1_a),
        "f1_b": float(f1_b),
        "delta_f1": float(observed_delta),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_deltas": bootstrap_deltas,
    }

def paired_permutation_f1(
    y_true,
    y_pred_a,
    y_pred_b,
    n_perm=10000,
    seed=123,
):
    """
    Two-sided paired permutation test.

    Under the null hypothesis, the two methods are exchangeable
    on each validation example. Predictions from A and B are
    randomly swapped within each paired sample.
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred_a = np.asarray(y_pred_a).astype(int)
    y_pred_b = np.asarray(y_pred_b).astype(int)

    if not (len(y_true) == len(y_pred_a) == len(y_pred_b)):
        raise ValueError("The paired prediction arrays must have identical length.")

    rng = np.random.default_rng(seed)
    n = len(y_true)

    observed_delta = (
        f1_score(y_true, y_pred_a, zero_division=0)
        - f1_score(y_true, y_pred_b, zero_division=0)
    )

    null_deltas = np.empty(n_perm, dtype=float)

    for i in range(n_perm):
        swap = rng.random(n) < 0.5

        perm_a = y_pred_a.copy()
        perm_b = y_pred_b.copy()

        temp = perm_a[swap].copy()
        perm_a[swap] = perm_b[swap]
        perm_b[swap] = temp

        null_deltas[i] = (
            f1_score(y_true, perm_a, zero_division=0)
            - f1_score(y_true, perm_b, zero_division=0)
        )

    # +1 correction avoids p=0 from finite Monte-Carlo sampling.
    p_value = (
        np.sum(np.abs(null_deltas) >= abs(observed_delta)) + 1
    ) / (n_perm + 1)

    return {
        "observed_delta": float(observed_delta),
        "p_value": float(p_value),
        "null_deltas": null_deltas,
    }

def compare_methods(
    y_true,
    pred_nnpu,
    pred_baseline,
    baseline_name,
    condition,
    n_boot=10000,
    n_perm=10000,
):
    boot = paired_bootstrap_f1(
        y_true=y_true,
        y_pred_a=pred_nnpu,
        y_pred_b=pred_baseline,
        n_boot=n_boot,
        seed=42,
    )

    perm = paired_permutation_f1(
        y_true=y_true,
        y_pred_a=pred_nnpu,
        y_pred_b=pred_baseline,
        n_perm=n_perm,
        seed=123,
    )

    # Difference is statistically supported at the 95% CI level
    # when zero is outside the confidence interval.
    significant_ci = (
        boot["ci_low"] > 0
        or boot["ci_high"] < 0
    )

    return {
        "Condition": condition,
        "Comparison": f"TF-nnPU vs {baseline_name}",
        "F1_nnPU": boot["f1_a"],
        "F1_baseline": boot["f1_b"],
        "Delta_F1": boot["delta_f1"],
        "CI_low": boot["ci_low"],
        "CI_high": boot["ci_high"],
        "p_value": perm["p_value"],
        "Significant_95CI": bool(significant_ci),
    }


def compare_prediction_files(a,b,baseline_name='baseline',condition='provided predictions',n_resamples=10000):
    a=pd.read_csv(a); b=pd.read_csv(b); keys=['sequence_id','endpoint']
    if a.duplicated(keys).any() or b.duplicated(keys).any(): raise ValueError('Duplicate prediction keys')
    a=a.sort_values(keys).reset_index(drop=True); b=b.sort_values(keys).reset_index(drop=True)
    if not a[keys].equals(b[keys]) or not np.array_equal(a.y_true,b.y_true): raise ValueError('Prediction keys or labels differ')
    return compare_methods(a.y_true.to_numpy(),a.y_pred.to_numpy(),b.y_pred.to_numpy(),baseline_name,condition,n_resamples,n_resamples)

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser();p.add_argument('nnpu_predictions');p.add_argument('baseline_predictions');p.add_argument('--resamples',type=int,default=10000)
    args=p.parse_args();print(json.dumps(compare_prediction_files(args.nnpu_predictions,args.baseline_predictions,n_resamples=args.resamples),indent=2))
