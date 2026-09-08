# Split format

Each `.npz` is loaded with `allow_pickle=False`.

- `train_idx`, `val_idx`: global indices in the corresponding prefix dataset, in DataLoader subset order.
- `keys`: N x 2 integer array `(sequence_id, endpoint)`. Endpoint is zero-based; prefix length is endpoint+1.
- Main file: `labels` contains the last-frame observed PU label.
- Common file: `current_risk_labels` contains observed PU labels; `scenario_outcome_labels` contains max(frame labels) for the scenario.

The main dataset has 10,987 training and 2,747 validation prefixes. The common dataset has 9,417 training and 2,355 validation samples. Both preserve the PyTorch seed-42 random prefix split. `python -m tf_nnpu verify` checks the files against the supplied dataset and original generation order. Training/evaluation runners load the stored arrays.
