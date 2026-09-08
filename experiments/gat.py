"""TF-nnPU versus GAT. Model equations are retained from the supplied notebook."""
from pathlib import Path

def run(train=False, smoke=False, output=None, device_name='auto'):
    import pandas as pd
    from tf_nnpu.paths import ROOT
    if not train:
        table=pd.read_csv(ROOT/'results'/'gat_all_noise_f1.csv')
        print('Recorded reference results; no training or cache construction.'); print(table.to_string(index=False))
        return table

    from tf_nnpu.paths import ROOT, DATA
    from tf_nnpu.artifacts import verify_manifest, validate_data_and_splits, load_split, load_noise_map, new_run
    from experiments.support import save_training_artifact

    verify_manifest(); validate_data_and_splits()
    out=new_run('gat',output,{'smoke':smoke,'device':device_name,'historical_loss':True})
    # ============================================================
    # 1. Imports / configuration
    # ============================================================

    import os
    import math
    import copy
    import random
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, Subset
    from torch.nn.utils.rnn import pad_sequence

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        precision_score,
        recall_score,
        f1_score,
    )

    import matplotlib.pyplot as plt

    try:
        from scipy.stats import wilcoxon
        SCIPY_AVAILABLE = True
    except Exception:
        SCIPY_AVAILABLE = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    SEED = 42

    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Same endpoint set for every model:
    # frame_id 3 = 4 observed frames; frame_id 14 = 15 observed frames
    MIN_ENDPOINT = 3
    MAX_ENDPOINT = 14

    N_OUTER_FOLDS = 5
    INNER_VAL_FRACTION = 0.15

    # Exact Stage-1 hyperparameters from your uploaded notebook
    AE_EPOCHS = 1 if smoke else 20
    AE_BATCH_SIZE = 16
    AE_LR = 1e-3
    AE_WEIGHT_DECAY = 1e-4

    # Exact Stage-2 hyperparameters from your uploaded notebook
    CLF_EPOCHS = 1 if smoke else 200
    CLF_BATCH_SIZE = 32
    CLF_LR = 1e-4
    CLF_WEIGHT_DECAY = 1e-4
    EARLY_STOP_PATIENCE = 8
    EARLY_STOP_MIN_DELTA = 1e-3

    POS_PRIOR = 0.10

    NOISE_CONDITIONS = [
        ("none", 0.00),
        ("symmetric", 0.05),
        ("symmetric", 0.10),
        ("symmetric", 0.15),
        ("symmetric", 0.20),
        ("p_to_u", 0.05),
        ("p_to_u", 0.10),
        ("p_to_u", 0.15),
        ("p_to_u", 0.20),
    ]
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name)


    def load_mix_txt(path: str) -> np.ndarray:
        """Load TXT file with columns:
        0: time/frame index
        1: distance
        2: angle
        3: sin(yaw)
        4: cos(yaw)
        5: speed
        6: frame-level label (0 = safe, 1 = collision)
        7: sequence_id
        """
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[None, :]  # ensure 2D
        return data


    def group_sequences_with_frame_labels(data: np.ndarray):
        """Group flat data by sequence_id and sort by time.

        Returns:
            sequences: list of arrays, each of shape (T_i, 5) with features
                       [distance, angle, sin(yaw), cos(yaw), speed]
            frame_labels: list of arrays, each of shape (T_i,) with 0/1 labels
            seq_ids: list of sequence ids
        """
        seq_col = data[:, 7].astype(int)
        time_col = data[:, 0]
        unique_ids = np.unique(seq_col)

        sequences = []
        frame_labels = []
        seq_ids = []

        for sid in unique_ids:
            mask = (seq_col == sid)
            seq_data = data[mask]
            # sort by time
            order = np.argsort(seq_data[:, 0])
            seq_data = seq_data[order]

            # features: columns 1..5
            X = seq_data[:, 1:6].astype(np.float32)
            y = seq_data[:, 6].astype(np.float32)

            sequences.append(X)
            frame_labels.append(y)
            seq_ids.append(int(sid))

        return sequences, frame_labels, seq_ids

    def pad_collate_unsupervised(batch):
        """Collate function for UnsupervisedSeqDataset.

        Args:
            batch: list of tensors [T_i, F]

        Returns:
            padded: (B, T_max, F)
            pad_mask: (B, T_max) bool, True where padding
        """
        lengths = [x.shape[0] for x in batch]
        padded = pad_sequence(batch, batch_first=True)  # (B, T_max, F)
        T_max = padded.shape[1]

        # pad_mask: True where positions are padding
        idxs = torch.arange(T_max).unsqueeze(0).expand(len(lengths), -1)
        lengths_tensor = torch.tensor(lengths).unsqueeze(1)
        pad_mask = idxs >= lengths_tensor

        return padded, pad_mask


    def pad_collate_supervised(batch):
        """Collate function for supervised prefix dataset.

        Args:
            batch: list of (tensor[T_i, F], label)

        Returns:
            padded: (B, T_max, F)
            labels: (B,)
            pad_mask: (B, T_max) bool
        """
        seqs, labels = zip(*batch)
        lengths = [x.shape[0] for x in seqs]
        padded = pad_sequence(seqs, batch_first=True)  # (B, T_max, F)
        T_max = padded.shape[1]

        idxs = torch.arange(T_max).unsqueeze(0).expand(len(lengths), -1)
        lengths_tensor = torch.tensor(lengths).unsqueeze(1)
        pad_mask = idxs >= lengths_tensor

        labels = torch.stack(labels)  # (B,)
        return padded, labels, pad_mask

    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequences with batch_first=True.

        Expects input of shape (B, T, D).
        """
        def __init__(self, d_model: int, max_len: int = 500):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, D)
            T = x.size(1)
            x = x + self.pe[:, :T, :]
            return x


    class TemporalEncoder(nn.Module):
        """Shared temporal encoder: input_dim -> d_model via TransformerEncoder."""
        def __init__(
            self,
            input_dim: int = 5,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            max_len: int = 500,
        ):
            super().__init__()
            self.input_dim = input_dim
            self.d_model = d_model

            self.in_proj = nn.Linear(input_dim, d_model)
            self.pos = PositionalEncoding(d_model, max_len=max_len)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
            """Forward pass.

            Args:
                x: (B, T, input_dim)
                pad_mask: (B, T) bool, True where padding

            Returns:
                h: (B, T, d_model)
            """
            h = self.in_proj(x)
            h = self.pos(h)
            h = self.encoder(h, src_key_padding_mask=pad_mask)
            return h

    class TemporalAutoencoder(nn.Module):
        """Temporal autoencoder: encoder + linear decoder back to input_dim."""
        def __init__(self, encoder: TemporalEncoder):
            super().__init__()
            self.encoder = encoder
            self.out_proj = nn.Linear(encoder.d_model, encoder.input_dim)

        def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None) -> torch.Tensor:
            """Reconstruct the input sequence.

            Args:
                x: (B, T, input_dim)
                pad_mask: (B, T) bool

            Returns:
                x_hat: (B, T, input_dim)
            """
            h = self.encoder(x, pad_mask=pad_mask)
            x_hat = self.out_proj(h)
            return x_hat

    class TemporalRiskTransformer(nn.Module):
        """Transformer encoder + masked attention pooling -> sequence-level logit.

        Uses a pretrained TemporalEncoder and a small MLP head.
        """
        def __init__(self, encoder: TemporalEncoder, d_hidden: int = 64, dropout: float = 0.1):
            super().__init__()
            self.encoder = encoder

            d_model = encoder.d_model

            # Attention pooling over time
            self.pool = nn.Linear(d_model, 1)

            # Classification head
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, 1),
            )

        def forward(self, x: torch.Tensor, pad_mask: torch.Tensor = None):
            """Forward pass.

            Args:
                x: (B, T, input_dim)
                pad_mask: (B, T) bool

            Returns:
                logits: (B,) raw logits for collision probability
                z_seq: (B, d_model) pooled sequence representation
            """
            h = self.encoder(x, pad_mask=pad_mask)  # (B, T, d_model)

            # Attention logits over time
            att_logits = self.pool(h).squeeze(-1)  # (B, T)

            if pad_mask is not None:
                att_logits = att_logits.masked_fill(pad_mask, float("-inf"))

            att_weights = F.softmax(att_logits, dim=1)  # (B, T)

            # Weighted sum
            z_seq = torch.sum(h * att_weights.unsqueeze(-1), dim=1)  # (B, d_model)

            logits = self.head(z_seq).squeeze(-1)  # (B,)
            return logits, z_seq

    # === nnPU (Positive–Unlabeled) loss for collision (positive) vs safe (unlabeled) ===
    import torch.nn.functional as F

    def nnpu_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        prior: float,
        gamma: float = 1.0,
        beta: float = 0.0,
    ) -> torch.Tensor:
        # nnPU loss (non-negative PU) with logistic loss.
        #
        # logits: (B,)
        # targets: (B,) in {0, 1}, where:
        #     1 = positive (collision)
        #     0 = unlabeled (safe / mixture)
        #
        # prior: π_p, estimated positive prior (collision prior) in the training set.
        # gamma, beta: hyperparameters controlling how aggressively to handle negative risk.

        # Ensure float
        y = targets.float()
        logits = logits.view(-1)

        # Masks
        pos_mask = (y == 1)
        unl_mask = (y == 0)

        # Logistic loss for y=+1 and y=-1
        # l(f, +1) = log(1 + exp(-f)) = softplus(-f)
        # l(f, -1) = log(1 + exp(f))  = softplus(f)
        def loss_pos(z):
            return F.softplus(-z)

        def loss_neg(z):
            return F.softplus(z)

        device = logits.device

        if pos_mask.any():
            pos_logits = logits[pos_mask]
            L_p_pos = loss_pos(pos_logits).mean()          # E_p[l(f, +1)]
            L_p_neg = loss_neg(pos_logits).mean()          # E_p[l(f, -1)]
        else:
            # No positive examples in this batch; set to zero
            L_p_pos = torch.tensor(0.0, device=device)
            L_p_neg = torch.tensor(0.0, device=device)

        if unl_mask.any():
            unl_logits = logits[unl_mask]
            L_u_neg = loss_neg(unl_logits).mean()          # E_u[l(f, -1)]
        else:
            L_u_neg = torch.tensor(0.0, device=device)

        # Positive risk
        risk_positive = prior * L_p_pos

        # Negative risk estimate
        risk_negative = L_u_neg - prior * L_p_neg

        # Non-negative PU trick
        if risk_negative.item() < -beta:
            # In this regime, flip the sign of the negative risk and scale by gamma
            loss = risk_positive - gamma * risk_negative
        else:
            loss = risk_positive + risk_negative

        return loss

    def upu_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        prior: float,
        gamma: float = 1.0,
        beta: float = 0.0,
    ) -> torch.Tensor:
        """Unbiased PU (uPU) loss — same as nnPU but WITHOUT non-negative clipping."""

        y = targets.float()
        logits = logits.view(-1)

        pos_mask = (y == 1)
        unl_mask = (y == 0)

        def loss_pos(z):
            return F.softplus(-z)

        def loss_neg(z):
            return F.softplus(z)

        # Positive risk
        if pos_mask.sum() > 0:
            L_p_pos = loss_pos(logits[pos_mask]).mean()
            L_p_neg = loss_neg(logits[pos_mask]).mean()
        else:
            L_p_pos = torch.tensor(0.0, device=logits.device)
            L_p_neg = torch.tensor(0.0, device=logits.device)

        # Unlabeled risk
        if unl_mask.sum() > 0:
            L_u_neg = loss_neg(logits[unl_mask]).mean()
        else:
            L_u_neg = torch.tensor(0.0, device=logits.device)

        risk_positive = prior * L_p_pos
        risk_negative = L_u_neg - prior * L_p_neg  # <-- NO max(0, ...) here

        loss = risk_positive + risk_negative
        return loss

    import copy
    class EarlyStopping:
        def __init__(self, patience=8, min_delta=1e-4, mode="max"):
            """Simple early stopping with best-weight restore.
            mode="max": metric should increase (e.g., AUC)
            mode="min": metric should decrease (e.g., loss)
            """
            self.patience = patience
            self.min_delta = min_delta
            self.mode = mode
            self.best = -np.inf if mode == "max" else np.inf
            self.num_bad_epochs = 0
            self.best_state = None

        def step(self, metric, model):
            improved = (metric > self.best + self.min_delta) if self.mode == "max" else (metric < self.best - self.min_delta)
            if improved:
                self.best = metric
                self.num_bad_epochs = 0
                self.best_state = copy.deepcopy(model.state_dict())
                return False
            self.num_bad_epochs += 1
            return self.num_bad_epochs >= self.patience

        def restore_best_weights(self, model):
            if self.best_state is not None:
                model.load_state_dict(self.best_state)

    # ============================================================
    # 9. Verify the copied TF-nnPU architecture
    # ============================================================

    _exact_encoder = TemporalEncoder(
        input_dim=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_len=500,
    )
    _exact_tf = TemporalRiskTransformer(encoder=_exact_encoder)

    n_params = sum(p.numel() for p in _exact_tf.parameters() if p.requires_grad)

    print("TF-nnPU trainable parameters:", f"{n_params:,}")
    assert n_params == 71746, (
        f"Expected 71,746 parameters from your original model, got {n_params:,}"
    )
    print("Architecture verification PASSED: 71,746 trainable parameters.")


    TXT_PATH = DATA
    raw_data=load_mix_txt(TXT_PATH)
    sequences, frame_labels, seq_ids=group_sequences_with_frame_labels(raw_data)
    scene_to_X={int(s):x for s,x in zip(seq_ids,sequences)}
    scene_to_s={int(s):y.astype(np.float32) for s,y in zip(seq_ids,frame_labels)}
    scene_to_Y={int(s):int(y.max()) for s,y in zip(seq_ids,frame_labels)}
    ALL_SCENES=np.array(sorted(scene_to_X),dtype=int)
    split=load_split('common')
    COMMON_KEYS=[tuple(map(int,k)) for k in split['keys']]
    COMMON_CLEAN_S=split['current_risk_labels']
    COMMON_FULL_Y=split['scenario_outcome_labels']
    N_COMMON=len(COMMON_KEYS)
    TRAIN_IDX=split['train_idx'][:32] if smoke else split['train_idx']
    VAL_IDX=split['val_idx'][:32] if smoke else split['val_idx']
    TRAIN_KEYS=[COMMON_KEYS[i] for i in TRAIN_IDX]
    VAL_KEYS=[COMMON_KEYS[i] for i in VAL_IDX]
    train_scenes={sid for sid,end in TRAIN_KEYS}
    val_scenes={sid for sid,end in VAL_KEYS}


    def make_training_noise_map(noise_mode,noise_rate,seed):
        return load_noise_map('common_current_risk',noise_mode,noise_rate,seed)


    # ============================================================
    # Transformer datasets
    # ============================================================

    class AllTrajectoryDataset(Dataset):
        """All 15-frame trajectories for original-like Stage-1 pretraining."""
        def __init__(self):
            self.data = [
                torch.tensor(
                    scene_to_X[int(sid)],
                    dtype=torch.float32,
                )
                for sid in ALL_SCENES
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]


    class CommonTFDataset(Dataset):
        """All common prefixes in COMMON_KEYS order."""

        def __init__(self, noise_map=None):
            self.items = []

            for sid, end in COMMON_KEYS:
                X = scene_to_X[sid]
                s = scene_to_s[sid]

                clean_s = int(s[end])

                train_y = (
                    int(noise_map[(sid, end)])
                    if (
                        noise_map is not None
                        and (sid, end) in noise_map
                    )
                    else clean_s
                )

                prefix = X[
                    :end + 1
                ].astype(np.float32)

                self.items.append(
                    (
                        torch.tensor(
                            prefix,
                            dtype=torch.float32,
                        ),
                        float(train_y),
                        float(clean_s),
                        int(sid),
                        int(end),
                    )
                )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            X, train_y, eval_y, sid, end = self.items[idx]

            return (
                X,
                torch.tensor(train_y, dtype=torch.float32),
                torch.tensor(eval_y, dtype=torch.float32),
                torch.tensor(sid, dtype=torch.long),
                torch.tensor(end, dtype=torch.long),
            )


    def pad_collate_common_tf(batch):
        seqs, train_y, eval_y, scenes, ends = zip(*batch)

        lengths = [
            x.shape[0]
            for x in seqs
        ]

        padded = pad_sequence(
            seqs,
            batch_first=True,
        )

        T_max = padded.shape[1]

        idxs = torch.arange(
            T_max
        ).unsqueeze(0).expand(
            len(lengths),
            -1,
        )

        lengths_tensor = torch.tensor(
            lengths
        ).unsqueeze(1)

        pad_mask = (
            idxs >= lengths_tensor
        )

        return (
            padded,
            torch.stack(train_y),
            torch.stack(eval_y),
            pad_mask,
            torch.stack(scenes),
            torch.stack(ends),
        )

    # ============================================================
    # Stage-1 reconstruction pretraining on all scenes
    # ============================================================

    def pretrain_tf_all_scenes(seed=SEED):
        set_seed(seed)

        encoder = TemporalEncoder(
            input_dim=5,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
            max_len=500,
        ).to(device)

        ae_model = TemporalAutoencoder(
            encoder
        ).to(device)

        ds = AllTrajectoryDataset()
        if smoke: ds=Subset(ds,list(range(32)))

        loader = DataLoader(
            ds,
            batch_size=AE_BATCH_SIZE,
            shuffle=True,
            collate_fn=pad_collate_unsupervised,
        )

        optimizer = torch.optim.AdamW(
            ae_model.parameters(),
            lr=AE_LR,
            weight_decay=AE_WEIGHT_DECAY,
        )

        criterion = nn.MSELoss()

        for epoch in range(
            1,
            AE_EPOCHS + 1,
        ):
            ae_model.train()

            total_loss = 0.0

            for X, pad_mask in loader:
                X = X.to(device)
                pad_mask = pad_mask.to(device)

                X_hat = ae_model(
                    X,
                    pad_mask=pad_mask,
                )

                loss = criterion(
                    X_hat,
                    X,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (
                epoch == 1
                or epoch % 5 == 0
                or epoch == AE_EPOCHS
            ):
                print(
                    f"AE epoch {epoch:2d}/{AE_EPOCHS} | "
                    f"loss={total_loss/len(loader):.6f}"
                )

        return copy.deepcopy(
            ae_model.encoder.state_dict()
        )


    PRETRAINED_TF_STATE = pretrain_tf_all_scenes(
        seed=SEED
    )
    torch.save(PRETRAINED_TF_STATE,out/'stage1_encoder.pt')


    @torch.no_grad()
    def evaluate_tf_clean_s(model, loader, threshold=0.5):
        model.eval()
        all_probs = []
        all_labels = []
        for X, train_y, eval_y, pad_mask, scenes, ends in loader:
            X = X.to(device)
            pad_mask = pad_mask.to(device)
            logits, _ = model(X, pad_mask=pad_mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(eval_y.numpy())
        y_true = np.concatenate(all_labels).astype(int)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= threshold).astype(int)
        return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0), '_y_true': y_true, '_y_prob': y_prob}

    def train_tf_nnpu_prefix_split(noise_map, prior=POS_PRIOR, seed=SEED):
        set_seed(seed)
        full_ds = CommonTFDataset(noise_map=noise_map)
        train_ds = Subset(full_ds, TRAIN_IDX.tolist())
        val_ds = Subset(full_ds, VAL_IDX.tolist())
        train_loader = DataLoader(train_ds, batch_size=CLF_BATCH_SIZE, shuffle=True, collate_fn=pad_collate_common_tf)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=pad_collate_common_tf)
        encoder = TemporalEncoder(input_dim=5, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_len=500)
        encoder.load_state_dict(PRETRAINED_TF_STATE)
        model = TemporalRiskTransformer(encoder=encoder).to(device)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CLF_LR, weight_decay=CLF_WEIGHT_DECAY)
        early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE, min_delta=EARLY_STOP_MIN_DELTA, mode='max')
        best_epoch = None
        for epoch in range(1, CLF_EPOCHS + 1):
            model.train()
            for X, train_y, eval_y, pad_mask, scenes, ends in train_loader:
                X = X.to(device)
                train_y = train_y.to(device)
                pad_mask = pad_mask.to(device)
                logits, _ = model(X, pad_mask=pad_mask)
                loss = nnpu_loss(logits, train_y, prior=prior, gamma=1.0, beta=0.0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_metrics = evaluate_tf_clean_s(model, val_loader)
            stop = early_stopper.step(val_metrics['f1'], model)
            if val_metrics['f1'] == early_stopper.best:
                best_epoch = epoch
            if stop:
                break
        early_stopper.restore_best_weights(model)
        metrics = evaluate_tf_clean_s(model, val_loader)
        metrics['best_epoch'] = best_epoch
        metrics['_model'] = model
        return metrics


    # ============================================================
    # 15A. GAT feature construction
    # ============================================================

    GAT_WINDOW = 4
    D_CPA_THRESHOLD = 20.0
    T_CPA_THRESHOLD = 1.5

    GAT_LSTM_HIDDEN = 8
    GAT_HIDDEN = 16
    GAT_HEADS = 2
    GAT_DROPOUT = 0.1

    GAT_LR = 1e-3
    GAT_WEIGHT_DECAY = 1e-4
    GAT_BATCH_SIZE = 128
    GAT_MAX_EPOCHS = 1 if smoke else 80
    GAT_PATIENCE = 8
    GAT_MIN_DELTA = 1e-3


    def qcar_to_cartesian_state(X5):
        """
        X5 columns:
        distance, angle, sin_yaw, cos_yaw, speed
        """
        d = X5[:, 0]
        theta = X5[:, 1]
        sin_psi = X5[:, 2]
        cos_psi = X5[:, 3]
        speed = X5[:, 4]

        x = d * np.cos(theta)
        y = d * np.sin(theta)
        vx = speed * cos_psi
        vy = speed * sin_psi

        return np.stack([x, y, vx, vy], axis=-1).astype(np.float32)


    def closest_point_features(state):
        pos = state[:2]
        vel = state[2:]

        vv = float(np.dot(vel, vel))
        rv = float(np.dot(pos, vel))

        approaching = vv > 1e-8 and rv < 0.0

        if approaching:
            t_cpa = max(0.0, -rv / vv)
            d_cpa = float(np.linalg.norm(pos + vel * t_cpa))
        else:
            t_cpa = 0.0
            d_cpa = float(np.linalg.norm(pos))

        return d_cpa, t_cpa, approaching


    def fit_gat_scalers(scene_ids_subset):
        node_states = []
        edge_states = []

        for sid in scene_ids_subset:
            state = qcar_to_cartesian_state(scene_to_X[int(sid)])
            node_states.append(state)

            for s in state:
                d_cpa, t_cpa, _ = closest_point_features(s)
                edge_states.append([d_cpa, t_cpa])

        node_scaler = StandardScaler().fit(np.concatenate(node_states, axis=0))
        edge_scaler = StandardScaler().fit(
            np.asarray(edge_states, dtype=np.float32)
        )

        return node_scaler, edge_scaler

    # ============================================================
    # 15C. Dynamic GAT-LSTM architecture
    # ============================================================

    class EdgeAwareGATLayer(nn.Module):
        def __init__(
            self,
            in_dim,
            out_dim=GAT_HIDDEN,
            heads=GAT_HEADS,
            edge_dim=2,
            dropout=GAT_DROPOUT,
        ):
            super().__init__()

            if out_dim % heads != 0:
                raise ValueError("out_dim must be divisible by heads")

            self.heads = heads
            self.head_dim = out_dim // heads

            self.W = nn.Linear(in_dim, out_dim, bias=False)
            self.edge_proj = nn.Linear(edge_dim, heads, bias=False)

            self.a_src = nn.Parameter(
                torch.empty(heads, self.head_dim)
            )
            self.a_dst = nn.Parameter(
                torch.empty(heads, self.head_dim)
            )

            nn.init.xavier_uniform_(self.a_src)
            nn.init.xavier_uniform_(self.a_dst)

            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, h, adj, edge_attr):
            B, N, _ = h.shape

            z = self.W(h).view(
                B, N, self.heads, self.head_dim
            )

            src = (z * self.a_src).sum(dim=-1)
            dst = (z * self.a_dst).sum(dim=-1)

            scores = src.unsqueeze(2) + dst.unsqueeze(1)
            scores = scores + self.edge_proj(edge_attr)
            scores = F.leaky_relu(scores, negative_slope=0.2)

            mask = adj.unsqueeze(-1) > 0
            scores = scores.masked_fill(~mask, -1e9)

            alpha = torch.softmax(scores, dim=2)
            alpha = self.dropout(alpha)

            out = torch.einsum(
                "bijh,bjhd->bihd",
                alpha,
                z,
            )
            out = out.reshape(B, N, -1)

            return self.norm(F.elu(out))


    class AdaptedDynamicGATLSTM(nn.Module):
        def __init__(self):
            super().__init__()

            self.node_lstm = nn.LSTM(
                input_size=4,
                hidden_size=GAT_LSTM_HIDDEN,
                num_layers=1,
                batch_first=True,
            )

            self.node_type_embedding = nn.Embedding(2, 2)

            self.gat = EdgeAwareGATLayer(
                in_dim=GAT_LSTM_HIDDEN + 2,
                out_dim=GAT_HIDDEN,
                heads=GAT_HEADS,
                edge_dim=2,
                dropout=GAT_DROPOUT,
            )

            self.classifier = nn.Sequential(
                nn.Linear(GAT_HIDDEN, 16),
                nn.ReLU(),
                nn.Dropout(GAT_DROPOUT),
                nn.Linear(16, 1),
            )

        def forward(self, node_seq, node_type, adj, edge_attr):
            B, N, T, Fdim = node_seq.shape

            x = node_seq.reshape(B * N, T, Fdim)

            _, (h_n, _) = self.node_lstm(x)

            h = h_n[-1].reshape(B, N, -1)

            type_emb = self.node_type_embedding(node_type)

            h = torch.cat(
                [h, type_emb],
                dim=-1,
            )

            h = self.gat(
                h,
                adj,
                edge_attr,
            )

            graph_embedding = h.mean(dim=1)

            return self.classifier(
                graph_embedding
            ).squeeze(-1)

    # ============================================================
    # Common GAT dataset
    # ============================================================

    # Fit normalization on scenes represented in the training subset.
    # With prefix-level splitting this will normally include almost/all scenes.
    GAT_TRAIN_SCENES = np.array(
        sorted(train_scenes),
        dtype=int,
    )

    GAT_NODE_SCALER, GAT_EDGE_SCALER = fit_gat_scalers(
        GAT_TRAIN_SCENES
    )


    class CommonGATDataset(Dataset):
        """All GAT windows in the exact same COMMON_KEYS order."""

        def __init__(self, noise_map=None):
            self.items = []

            # Cache Cartesian states once per scene.
            state_cache = {
                sid: qcar_to_cartesian_state(
                    scene_to_X[sid]
                )
                for sid in ALL_SCENES
            }

            for sid, end in COMMON_KEYS:
                state_raw = state_cache[sid]
                clean_s = int(
                    scene_to_s[sid][end]
                )

                train_y = (
                    int(noise_map[(sid, end)])
                    if (
                        noise_map is not None
                        and (sid, end) in noise_map
                    )
                    else clean_s
                )

                target_raw = state_raw[
                    end-GAT_WINDOW+1:end+1
                ]

                d_cpa, t_cpa, approaching = (
                    closest_point_features(
                        state_raw[end]
                    )
                )

                cross_edge = (
                    approaching
                    and d_cpa < D_CPA_THRESHOLD
                    and t_cpa < T_CPA_THRESHOLD
                )

                target_seq = (
                    GAT_NODE_SCALER
                    .transform(target_raw)
                    .astype(np.float32)
                )

                ego_raw = np.zeros_like(
                    target_raw
                )

                ego_seq = (
                    GAT_NODE_SCALER
                    .transform(ego_raw)
                    .astype(np.float32)
                )

                node_seq = np.stack(
                    [ego_seq, target_seq],
                    axis=0,
                )

                node_type = np.asarray(
                    [0, 1],
                    dtype=np.int64,
                )

                adj = np.eye(
                    2,
                    dtype=np.float32,
                )

                if cross_edge:
                    adj[0, 1] = 1.0
                    adj[1, 0] = 1.0

                edge_attr = np.zeros(
                    (2, 2, 2),
                    dtype=np.float32,
                )

                e = (
                    GAT_EDGE_SCALER
                    .transform(
                        np.asarray(
                            [[d_cpa, t_cpa]],
                            dtype=np.float32,
                        )
                    )[0]
                    .astype(np.float32)
                )

                if cross_edge:
                    edge_attr[0, 1] = e
                    edge_attr[1, 0] = e

                self.items.append(
                    (
                        torch.tensor(
                            node_seq,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            node_type,
                            dtype=torch.long,
                        ),
                        torch.tensor(
                            adj,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            edge_attr,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            train_y,
                            dtype=torch.float32,
                        ),
                        torch.tensor(
                            clean_s,
                            dtype=torch.float32,
                        ),
                    )
                )

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]
    import joblib
    joblib.dump({"node":GAT_NODE_SCALER,"edge":GAT_EDGE_SCALER},out/"scalers.joblib")


    @torch.no_grad()
    def evaluate_gat_clean_s(model, loader, threshold=0.5):
        model.eval()
        all_probs = []
        all_labels = []
        for node_seq, node_type, adj, edge_attr, train_y, eval_y in loader:
            node_seq = node_seq.to(device)
            node_type = node_type.to(device)
            adj = adj.to(device)
            edge_attr = edge_attr.to(device)
            logits = model(node_seq, node_type, adj, edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(eval_y.numpy())
        y_true = np.concatenate(all_labels).astype(int)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= threshold).astype(int)
        return {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0), '_y_true': y_true, '_y_prob': y_prob}

    def train_gat_prefix_split(noise_map, loss_mode, prior=POS_PRIOR, seed=SEED):
        set_seed(seed)
        full_ds = CommonGATDataset(noise_map=noise_map)
        train_ds = Subset(full_ds, TRAIN_IDX.tolist())
        val_ds = Subset(full_ds, VAL_IDX.tolist())
        train_loader = DataLoader(train_ds, batch_size=GAT_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
        model = AdaptedDynamicGATLSTM().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=GAT_LR, weight_decay=GAT_WEIGHT_DECAY)
        early_stopper = EarlyStopping(patience=GAT_PATIENCE, min_delta=GAT_MIN_DELTA, mode='max')
        best_epoch = None
        for epoch in range(1, GAT_MAX_EPOCHS + 1):
            model.train()
            for node_seq, node_type, adj, edge_attr, train_y, eval_y in train_loader:
                node_seq = node_seq.to(device)
                node_type = node_type.to(device)
                adj = adj.to(device)
                edge_attr = edge_attr.to(device)
                train_y = train_y.to(device)
                logits = model(node_seq, node_type, adj, edge_attr)
                if loss_mode == 'bce':
                    loss = F.binary_cross_entropy_with_logits(logits, train_y)
                elif loss_mode == 'nnpu':
                    loss = nnpu_loss(logits, train_y, prior=prior, gamma=1.0, beta=0.0)
                else:
                    raise ValueError(loss_mode)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_metrics = evaluate_gat_clean_s(model, val_loader)
            stop = early_stopper.step(val_metrics['f1'], model)
            if val_metrics['f1'] == early_stopper.best:
                best_epoch = epoch
            if stop:
                break
        early_stopper.restore_best_weights(model)
        metrics = evaluate_gat_clean_s(model, val_loader)
        metrics['best_epoch'] = best_epoch
        metrics['_model'] = model
        return metrics

    rows=[]
    for condition_idx,(mode,rate) in enumerate(NOISE_CONDITIONS):
        if smoke and condition_idx>0: break
        corruption_seed=SEED+1000*condition_idx
        noise_map,stats=make_training_noise_map(mode,rate,corruption_seed)
        methods=[('TF-nnPU',None),('GAT-LSTM-BCE','bce'),('GAT-LSTM-nnPU','nnpu')]
        for method,loss_mode in methods:
            model_seed=SEED+condition_idx if loss_mode is None else SEED+5000+condition_idx
            if loss_mode is None:
                res=train_tf_nnpu_prefix_split(noise_map,prior=POS_PRIOR,seed=model_seed)
            else:
                res=train_gat_prefix_split(noise_map,loss_mode=loss_mode,prior=POS_PRIOR,seed=model_seed)
            metadata={'method':method,'noise_mode':mode,'noise_rate':rate,'corruption_seed':corruption_seed,
                      'model_seed':model_seed,'best_epoch':res['best_epoch'],'smoke':smoke,'target':'observed_PU_label_s_t'}
            save_training_artifact(out,f'{condition_idx:02d}_{method}',res.pop('_model'),VAL_KEYS,res.pop('_y_true'),res.pop('_y_prob'),metadata)
            rows.append({**metadata,**res,**stats})
            pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
            print(method,res)
    print('Outputs:',out)
    return pd.DataFrame(rows)
