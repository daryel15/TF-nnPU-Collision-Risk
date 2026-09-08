# Corruption artifacts

`index.json` specifies the group, corruption mode, rate and RNG seed for all 16 masks. Validation labels are unchanged. `python -m tf_nnpu verify` reconstructs every selection and checks exact equality.

| Group | Array meaning | Seed rule |
|---|---|---|
| common_current_risk/symmetric | N x 2 `(sequence_id, zero-based endpoint)` pairs | 1042, 2042, 3042, 4042 |
| common_current_risk/p_to_u | Same pairs, restricted to observed-positive training samples | 5042, 6042, 7042, 8042 |
| su_lstm_outcome/p_to_u | Same pairs, restricted to derived-positive-outcome training samples | 1042, 2042, 3042, 4042 |
| cost_sensitive_symmetric | 1-D GLOBAL indices into the main prefix dataset | PyTorch seed 123 |

Symmetric rates are fractions of all training samples. P-to-U rates are fractions of positive training labels. Common and Su-LSTM masks use NumPy default_rng permutations; cost-sensitive masks use torch.randperm.

The previous four common-current-risk P-to-U masks were reconstructed using the wrong condition index. This release regenerates them with the supplied GAT/CMPA source's nine-condition schedule (clean, four symmetric, four P-to-U). Model seeds in those comparisons remain 42+condition_idx for TF and 5042+condition_idx for the baseline. Saved source and logs support this schedule; no historical raw RNG-state capture was available. Existing result CSVs were not edited to imply a new training verification.
