# Historical source: 03_cost_sensitive_and_significance.ipynb


Reference only. The runnable entry point is in `notebooks/`.


# Unsupervised Pretraining + Robust Supervised Head for Temporal Collision Risk

This notebook implements the following pipeline:

1. **Data loading** from a `safe_col_mix.txt` file with per-frame features and labels.
2. **Unsupervised pretraining** of a temporal Transformer encoder as an autoencoder on trajectory sequences (ignoring labels).
3. **Supervised fine-tuning** of a classifier head on top of the pretrained encoder, using a **robust bootstrapped loss** to mitigate label flipping.
4. (Optional) Simple evaluation and inspection utilities.

You should adapt the path to your TXT file and adjust hyperparameters as needed for your data and GPU.


## Original cell 1 (zero-based)

```python
import os
import math
import numpy as np
from typing import List, Tuple
import torch
import joblib
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import lineStyles
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import time
from sklearn.manifold import TSNE

# Optional: for simple metrics
try:
    from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Path to your mixed safe/collision TXT file
TXT_PATH = "safe_col_mix_dth_9.txt"  # <-- change this to your actual path


```

## Data Loading Utilities

## Original cell 3 (zero-based)

```python
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


# Quick sanity check (only if file exists)
if os.path.exists(TXT_PATH):
    raw_data = load_mix_txt(TXT_PATH)
    sequences, frame_labels, seq_ids = group_sequences_with_frame_labels(raw_data)
    print(f"Loaded {len(sequences)} sequences from {TXT_PATH}")
else:
    print(f"WARNING: TXT_PATH '{TXT_PATH}' not found. Please update the path.")

```

## Datasets: Unsupervised and Supervised (Prefix-Based)

## Original cell 5 (zero-based)

```python
class UnsupervisedSeqDataset(Dataset):
    """Dataset for unsupervised pretraining.

    Returns full sequences (T_i, 5), ignoring labels.
    """
    def __init__(self, txt_path: str):
        data = load_mix_txt(txt_path)
        sequences, _, _ = group_sequences_with_frame_labels(data)
        self.seqs = [torch.tensor(s, dtype=torch.float32) for s in sequences]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def build_current_risk_prefixes(
    sequences: List[np.ndarray],
    frame_labels: List[np.ndarray],
    min_len: int = 2,
    max_len: int = None,
):
    """Build prefix sequences and labels = last-frame label for each prefix.

    For each sequence of length T:
        prefixes of length L in [min_len, ..., T] (or up to max_len if set).
        label of prefix = y_seq[L-1]
    """
    prefix_seqs = []
    prefix_labels = []

    for X, y in zip(sequences, frame_labels):
        T = X.shape[0]
        if T < min_len:
            continue
        L_max = T if max_len is None else min(max_len, T)
        for L in range(min_len, L_max + 1):
            prefix_seqs.append(X[:L].astype(np.float32))
            prefix_labels.append(float(y[L - 1]))

    return prefix_seqs, np.array(prefix_labels, dtype=np.float32)


class CurrentRiskPrefixDataset(Dataset):
    """Supervised dataset of variable-length prefixes with binary labels.

    Each sample:
        X: (L, 5) prefix of a trajectory
        y: scalar, last-frame label in {0,1}
    """
    def __init__(self, txt_path: str, min_len: int = 2, max_len: int = None):
        data = load_mix_txt(txt_path)
        sequences, frame_labels, _ = group_sequences_with_frame_labels(data)
        prefix_seqs, prefix_labels = build_current_risk_prefixes(
            sequences, frame_labels, min_len=min_len, max_len=max_len
        )
        self.seqs = [torch.tensor(s, dtype=torch.float32) for s in prefix_seqs]
        self.labels = torch.tensor(prefix_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

```

## Collate Functions (Padding Variable-Length Sequences)

## Original cell 7 (zero-based)

```python
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

```

## Temporal Encoder (Transformer) and Positional Encoding

## Original cell 9 (zero-based)

```python
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

```

## Unsupervised Autoencoder Model (Pretraining Stage)

## Original cell 11 (zero-based)

```python
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

```

## Supervised Classifier with Attention Pooling (Robust Head)

## Original cell 13 (zero-based)

```python
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

```

## Robust Loss: Bootstrapped BCE (Mitigating Label Flipping)

## Original cell 15 (zero-based)

```python
def bootstrapped_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 0.8,
) -> torch.Tensor:
    """Bootstrapped binary cross-entropy with logits.

    targets: (B,) in {0,1}
    We mix the noisy targets with the model's own predictions to reduce the
    influence of flipped labels.

    y_soft = beta * y + (1 - beta) * p.detach()
    loss = BCEWithLogits(logits, y_soft)
    """
    probs = torch.sigmoid(logits)
    y_soft = beta * targets + (1.0 - beta) * probs.detach()
    loss = F.binary_cross_entropy_with_logits(logits, y_soft)
    return loss

```

## Original cell 16 (zero-based)

```python

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

```

## Stage 1: Unsupervised Pretraining of the Encoder

## Original cell 18 (zero-based)

```python
# Hyperparameters for the encoder and AE
encoder = TemporalEncoder(
    input_dim=5,
    d_model=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    max_len=500,
)
encoder.to(device)

ae_model = TemporalAutoencoder(encoder).to(device)

ae_epochs = 20           # adjust as needed
ae_batch_size = 16       # adjust based on GPU memory
ae_lr = 1e-3
ae_weight_decay = 1e-4

ae_optimizer = torch.optim.AdamW(ae_model.parameters(), lr=ae_lr, weight_decay=ae_weight_decay)
ae_criterion = nn.MSELoss()

if os.path.exists(TXT_PATH):
    unsup_ds = UnsupervisedSeqDataset(TXT_PATH)
    unsup_loader = DataLoader(
        unsup_ds,
        batch_size=ae_batch_size,
        shuffle=True,
        collate_fn=pad_collate_unsupervised,
    )
    print(f"Unsupervised dataset size: {len(unsup_ds)} sequences")
else:
    unsup_ds = None
    unsup_loader = None
    print("Unsupervised dataset not created because TXT file is missing.")

if unsup_loader is not None:
    for epoch in range(1, ae_epochs + 1):
        ae_model.train()
        epoch_loss = 0.0
        for X, pad_mask in unsup_loader:
            X = X.to(device)
            pad_mask = pad_mask.to(device)

            X_hat = ae_model(X, pad_mask=pad_mask)
            loss = ae_criterion(X_hat, X)

            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()

            epoch_loss += loss.item()

        mean_loss = epoch_loss / len(unsup_loader)
        print(f"[AE] Epoch {epoch}/{ae_epochs} - MSE Loss: {mean_loss:.6f}")
else:
    print("Skip AE training (no data).")

```

## Original cell 19 (zero-based)

```python
# Save Stage 1 pretrained encoder weights
torch.save(ae_model.encoder.state_dict(), "pretrained_encoder.pt")
print("Stage 1 encoder saved.")
```

## Stage 2: Supervised Fine-Tuning with Robust Head

## Original cell 21 (zero-based)

```python
class LabelFlipWrapper(Dataset):
    """
    Wraps any dataset returning (x, y) with y in {0,1}.
    Flips y for a percentage of samples (symmetric flipping: 0<->1).
    """
    def __init__(self, base_ds, flip_pct: float, seed: int = 0):
        assert 0.0 <= flip_pct <= 1.0
        self.base_ds = base_ds
        self.flip_pct = flip_pct

        n = len(base_ds)
        k = int(round(flip_pct * n))

        g = torch.Generator()
        g.manual_seed(seed)

        perm = torch.randperm(n, generator=g)
        flip_local_idx = perm[:k]
        self.flip_mask = torch.zeros(n, dtype=torch.bool)
        self.flip_mask[flip_local_idx] = True

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, i):
        x, y = self.base_ds[i]
        # ensure scalar 0/1 float
        y = float(y)
        if self.flip_mask[i]:
            y = 1.0 - y
        return x, torch.tensor(y, dtype=torch.float32)
```

## Original cell 22 (zero-based)

```python
# Create a fresh encoder
encoder = TemporalEncoder(
    input_dim=5, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_len=500,)

# Load Stage 1 weights into it
encoder.load_state_dict(torch.load("pretrained_encoder.pt"))

# Create the classifier using this encoder
clf_model = TemporalRiskTransformer(encoder=encoder).to(device)

print("Fresh model with pretrained encoder loaded.")

# Now run Stage 2 with your chosen prior
pos_prior = 0.1# change this for each run: 0.05, 0.10, 0.15, 0.20, 0.25, 0.30
```

## Original cell 23 (zero-based)

```python
# Optionally: freeze some or all encoder parameters at the start
freeze_encoder = False  # set True to freeze encoder fully
if freeze_encoder:
    for p in clf_model.encoder.parameters():
        p.requires_grad = False

# PU-learning dataset (prefix-based: collision = positive, safe = unlabeled)
if os.path.exists(TXT_PATH):
    sup_ds = CurrentRiskPrefixDataset(TXT_PATH, min_len=2, max_len=None)
    print(f"Supervised prefix dataset size: {len(sup_ds)} samples")
    torch.manual_seed(42)
    # Train/validation split
    n_total = len(sup_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds, val_ds = random_split(sup_ds, [n_train, n_val])
    flip_pct = 0   # 10% label flipping
    flip_seed = 123

    #train_ds = LabelFlipWrapper(train_ds, flip_pct=flip_pct, seed=flip_seed)
    #train_ds = PositiveHideWrapper(
    #base_ds=train_ds,
    #hide_pct=0.7,          # hide 70% of positives → c = 0.3
    #unlabeled_value=0,
    #seed=42,)


    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=pad_collate_supervised,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=pad_collate_supervised,
    )
else:
    sup_ds = None
    train_loader = None
    val_loader = None
    print("Supervised dataset not created because TXT file is missing.")
```

## Original cell 24 (zero-based)

```python
# Training hyperparameters for supervised fine-tuning
clf_epochs = 200  # set high; early stopping will stop earlier
clf_lr = 1e-4  # smaller LR, since encoder is already pretrained
clf_weight_decay = 1e-4
boot_beta = 0.8

clf_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, clf_model.parameters()),
    lr=clf_lr,
    weight_decay=clf_weight_decay,
)

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


def evaluate_classifier(model, loader, threshold=0.5):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y, pad_mask in loader:
            X = X.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)

            logits, _ = model(X, pad_mask=pad_mask)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    preds = (all_probs >= threshold).astype(np.int32)

    acc = accuracy_score(all_labels, preds)
    auc = roc_auc_score(all_labels, all_probs)
    recall = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)

    return acc, auc, recall, f1

```

# COST-SENSITIVE LEARNING BASELINES

## Original cell 26 (zero-based)

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import copy

# ============================================================
# COST-SENSITIVE LOSS FUNCTIONS
# ============================================================

def weighted_bce_loss(logits, targets, pos_weight=3.0):
    """
    Weighted Binary Cross-Entropy.
    pos_weight > 1 penalizes false negatives more (costs more to miss a collision).
    """
    weight = torch.where(targets == 1,
                         torch.tensor(pos_weight, device=logits.device),
                         torch.tensor(1.0, device=logits.device))
    loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weight)
    return loss


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """
    Focal Loss (Lin et al., 2017).
    Reduces loss for well-classified examples, focusing on hard cases.
    alpha: weighting factor for positive class
    gamma: focusing parameter (higher = more focus on hard examples)
    """
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

    # p_t = prob if y=1, (1-prob) if y=0
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # alpha weighting
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    # focal modulation
    focal_weight = alpha_t * (1 - p_t) ** gamma

    loss = (focal_weight * bce).mean()
    return loss


# ============================================================
# EXPERIMENT: COST-SENSITIVE BASELINES
# ============================================================

baselines = [
    {
        "name": "TF-nnPU",
        "loss_fn": lambda logits, y: nnpu_loss(logits, y, prior=0.10),
        "description": "Proposed (nnPU)"
    },
    {
        "name": "TF-WBCE",
        "loss_fn": lambda logits, y: weighted_bce_loss(logits, y, pos_weight=3.0),
        "description": "Weighted BCE (w=3.0)"
    },
    {
        "name": "TF-Focal",
        "loss_fn": lambda logits, y: focal_loss(logits, y, alpha=0.25, gamma=2.0),
        "description": "Focal Loss (α=0.25, γ=2.0)"
    },
]

baseline_results = {}

for bl in baselines:
    print(f"\n{'='*60}")
    print(f"  Training: {bl['name']} — {bl['description']}")
    print(f"{'='*60}")

    # --- Fresh model with SAME pretrained encoder ---
    encoder_bl = TemporalEncoder(
        input_dim=5, d_model=64, nhead=4, num_layers=2,
        dim_feedforward=128, dropout=0.1, max_len=500
    )
    encoder_bl.load_state_dict(torch.load("pretrained_encoder.pt"))
    clf_bl = TemporalRiskTransformer(encoder_bl).to(device)

    optimizer_bl = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, clf_bl.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    early_stopper_bl = EarlyStopping(patience=8, min_delta=1e-3, mode="max")

    # --- Train ---
    print("  Training Stage 2...")
    for epoch in range(1, 201):
        clf_bl.train()
        epoch_loss = 0.0
        for X, y, pad_mask in train_loader:
            X, y, pad_mask = X.to(device), y.to(device), pad_mask.to(device)
            logits, _ = clf_bl(X, pad_mask=pad_mask)
            loss = bl["loss_fn"](logits, y)
            optimizer_bl.zero_grad()
            loss.backward()
            optimizer_bl.step()
            epoch_loss += loss.item()

        acc, auc, recall, f1 = evaluate_classifier(clf_bl, val_loader)
        if epoch % 10 == 0:
            print(f"    Epoch {epoch}: Loss={epoch_loss/len(train_loader):.4f}, Val F1={f1:.4f}")

        if early_stopper_bl.step(f1, clf_bl):
            print(f"    Early stopping at epoch {epoch}")
            break

    early_stopper_bl.restore_best_weights(clf_bl)

    # --- Evaluate ---
    clf_bl.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y, pad_mask in val_loader:
            X, pad_mask = X.to(device), pad_mask.to(device)
            logits, _ = clf_bl(X, pad_mask=pad_mask)
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(y.numpy())

    y_true = np.concatenate(all_labels).astype(int)
    y_prob = np.concatenate(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1_val = f1_score(y_true, y_pred, zero_division=0)

    baseline_results[bl["name"]] = {
        "precision": p,
        "recall": r,
        "f1": f1_val,
        "description": bl["description"],

        # NEW for Reviewer 1, Comment 3:
        # Store the exact matched validation outputs used to compute
        # the reported clean-supervision metrics. This does NOT
        # change the train/validation split or training procedure.
        "y_true": y_true.copy(),
        "y_prob": y_prob.copy(),
        "y_pred": y_pred.copy(),
    }

    print(f"\n  {bl['name']}: P={p:.4f}, R={r:.4f}, F1={f1_val:.4f}")


# ============================================================
# ALSO ADD THE EXISTING RESULTS (from Fig. 8)
# ============================================================
# Add uPU, SVM, LogReg results if you have them
# baseline_results["TF-uPU"] = {"precision": ..., "recall": ..., "f1": 0.88, "description": "Unbiased PU"}
# baseline_results["TF-SVM"] = {"precision": ..., "recall": ..., "f1": 0.81, "description": "SVM"}
# baseline_results["TF-LogReg"] = {"precision": ..., "recall": ..., "f1": 0.65, "description": "Logistic Regression"}


# ============================================================
# PRINT SUMMARY TABLE
# ============================================================
print(f"\n{'='*65}")
print(f"  COST-SENSITIVE BASELINE COMPARISON")
print(f"{'='*65}")
print(f"  {'Method':<16} {'Description':<28} {'Precision':>10} {'Recall':>8} {'F1':>6}")
print(f"  {'─'*60}")
for name in ["TF-WBCE", "TF-Focal", "TF-nnPU"]:
    r = baseline_results[name]
    marker = " ←" if name == "TF-nnPU" else ""
    print(f"  {name:<16} {r['description']:<28} {r['precision']:>10.4f} {r['recall']:>8.4f} {r['f1']:>6.4f}{marker}")


# ============================================================
# NOISE ROBUSTNESS: Run at 10% and 15% noise too
# ============================================================
print(f"\n{'='*60}")
print(f"  NOISE ROBUSTNESS COMPARISON")
print(f"{'='*60}")

noise_levels = [0.05, 0.10, 0.15, 0.2]
noise_results = {name: {} for name in ["TF-WBCE", "TF-Focal", "TF-nnPU"]}

# NEW for Reviewer 1, Comment 3:
# Save the matched validation predictions at every noise level so
# significance can be evaluated without changing the existing split.
noise_predictions = {
    name: {} for name in ["TF-WBCE", "TF-Focal", "TF-nnPU"]
}

for noise_pct in noise_levels:
    print(f"\n  --- Noise level: {noise_pct*100:.0f}% ---")

    # Create noisy training data
    torch.manual_seed(42)
    n_total = len(sup_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_ds_noise, val_ds_noise = random_split(sup_ds, [n_train, n_val])

    # Apply label flipping
    train_ds_noisy = LabelFlipWrapper(train_ds_noise, flip_pct=noise_pct, seed=123)

    train_loader_noise = DataLoader(train_ds_noisy, batch_size=32, shuffle=True,
                                     collate_fn=pad_collate_supervised)
    val_loader_noise = DataLoader(val_ds_noise, batch_size=64, shuffle=False,
                                  collate_fn=pad_collate_supervised)

    for bl in baselines:
        # Fresh model with pretrained encoder
        encoder_n = TemporalEncoder(
            input_dim=5, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=128, dropout=0.1, max_len=500
        )
        encoder_n.load_state_dict(torch.load("pretrained_encoder.pt"))
        clf_n = TemporalRiskTransformer(encoder_n).to(device)

        optimizer_n = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, clf_n.parameters()),
            lr=1e-4, weight_decay=1e-4
        )
        early_stopper_n = EarlyStopping(patience=8, min_delta=1e-3, mode="max")

        for epoch in range(1, 201):
            clf_n.train()
            for X, y, pad_mask in train_loader_noise:
                X, y, pad_mask = X.to(device), y.to(device), pad_mask.to(device)
                logits, _ = clf_n(X, pad_mask=pad_mask)
                loss = bl["loss_fn"](logits, y)
                optimizer_n.zero_grad()
                loss.backward()
                optimizer_n.step()

            acc, auc, recall, f1 = evaluate_classifier(clf_n, val_loader_noise)
            if early_stopper_n.step(f1, clf_n):
                break

        early_stopper_n.restore_best_weights(clf_n)

        clf_n.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X, y, pad_mask in val_loader_noise:
                X, pad_mask = X.to(device), pad_mask.to(device)
                logits, _ = clf_n(X, pad_mask=pad_mask)
                all_probs.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(y.numpy())

        y_true = np.concatenate(all_labels).astype(int)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)
        f1_n = f1_score(y_true, y_pred, zero_division=0)

        noise_results[bl["name"]][noise_pct] = f1_n

        # NEW for Reviewer 1, Comment 3:
        # Keep the exact validation outputs for paired resampling.
        noise_predictions[bl["name"]][noise_pct] = {
            "y_true": y_true.copy(),
            "y_prob": y_prob.copy(),
            "y_pred": y_pred.copy(),
        }

        print(f"    {bl['name']}: F1={f1_n:.4f}")

# Print noise comparison
print(f"\n{'='*55}")
print(f"  F1-SCORE UNDER LABEL NOISE")
print(f"{'='*55}")
print(f"  {'Method':<16} {'0%':>8} {'5%':>8} {'10%':>8} {'15%':>8} {'20%':>8}")
print(f"  {'─'*45}")
for name in ["TF-WBCE", "TF-Focal", "TF-nnPU"]:
    f1_0 = baseline_results[name]["f1"]
    vals = [f"{f1_0:.3f}"]
    for n in noise_levels:
        vals.append(f"{noise_results[name][n]:.3f}")
    print(f"  {name:<16} {'  '.join(vals)}")
```

# Reviewer 1 — Comment 3: Statistical significance

This section **keeps the original train/validation split and training protocol unchanged**. It only uses the already-generated matched validation predictions to quantify whether the observed F1 differences between TF-nnPU and the cost-sensitive baselines are larger than evaluation-set variation.

Two paired analyses are reported:

1. **Paired bootstrap 95% confidence interval** for $\Delta F1 = F1_{\mathrm{nnPU}} - F1_{\mathrm{baseline}}$.
2. **Paired permutation test** on the same validation predictions.

The analysis is performed for the clean condition and for each label-noise level already evaluated above.


## Original cell 28 (zero-based)

```python
# ============================================================
# REVIEWER 1 — COMMENT 3
# STATISTICAL SIGNIFICANCE OF F1 DIFFERENCES
#
# IMPORTANT:
# - Uses the SAME split already created above.
# - Does NOT retrain or resplit the dataset.
# - Uses the exact matched validation predictions produced above.
# ============================================================

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------

required_methods = ["TF-nnPU", "TF-Focal", "TF-WBCE"]

for method in required_methods:
    if method not in baseline_results:
        raise RuntimeError(
            f"{method} not found in baseline_results. "
            "Run the cost-sensitive baseline cell above first."
        )

    for key in ["y_true", "y_prob", "y_pred"]:
        if key not in baseline_results[method]:
            raise RuntimeError(
                f"baseline_results['{method}']['{key}'] is missing. "
                "Re-run the modified cost-sensitive baseline cell above."
            )

if "noise_predictions" not in globals():
    raise RuntimeError(
        "noise_predictions is not defined. "
        "Run the modified noise-robustness experiment above first."
    )


# ============================================================
# 1. PAIRED BOOTSTRAP 95% CI FOR DELTA F1
# ============================================================

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


# ============================================================
# 2. PAIRED PERMUTATION TEST FOR DELTA F1
# ============================================================

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


# ============================================================
# 3. WRAPPER FOR ONE PAIRED COMPARISON
# ============================================================

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


# ============================================================
# 4. CLEAN-SUPERVISION COMPARISONS
# ============================================================

clean_stats = []

y_true_clean = baseline_results["TF-nnPU"]["y_true"]
pred_nnpu_clean = baseline_results["TF-nnPU"]["y_pred"]

# Verify that all methods were evaluated on the same labels/order.
for baseline_name in ["TF-Focal", "TF-WBCE"]:
    if not np.array_equal(
        y_true_clean,
        baseline_results[baseline_name]["y_true"],
    ):
        raise RuntimeError(
            f"Validation labels/order differ between TF-nnPU and {baseline_name}."
        )

    clean_stats.append(
        compare_methods(
            y_true=y_true_clean,
            pred_nnpu=pred_nnpu_clean,
            pred_baseline=baseline_results[baseline_name]["y_pred"],
            baseline_name=baseline_name,
            condition="Clean",
        )
    )

clean_stats_df = pd.DataFrame(clean_stats)


# ============================================================
# 5. LABEL-NOISE COMPARISONS
# ============================================================

noise_stats = []

for noise_pct in noise_levels:

    y_true_noise = noise_predictions["TF-nnPU"][noise_pct]["y_true"]
    pred_nnpu_noise = noise_predictions["TF-nnPU"][noise_pct]["y_pred"]

    for baseline_name in ["TF-Focal", "TF-WBCE"]:

        if not np.array_equal(
            y_true_noise,
            noise_predictions[baseline_name][noise_pct]["y_true"],
        ):
            raise RuntimeError(
                f"Validation labels/order differ between TF-nnPU and "
                f"{baseline_name} at noise={noise_pct:.0%}."
            )

        noise_stats.append(
            compare_methods(
                y_true=y_true_noise,
                pred_nnpu=pred_nnpu_noise,
                pred_baseline=noise_predictions[baseline_name][noise_pct]["y_pred"],
                baseline_name=baseline_name,
                condition=f"{int(round(noise_pct * 100))}% noise",
            )
        )

noise_stats_df = pd.DataFrame(noise_stats)


# ============================================================
# 6. FINAL REVIEWER TABLE
# ============================================================

stats_df = pd.concat(
    [clean_stats_df, noise_stats_df],
    ignore_index=True,
)

stats_display = stats_df.copy()

stats_display["F1_nnPU"] = stats_display["F1_nnPU"].map(
    lambda x: f"{x:.4f}"
)
stats_display["F1_baseline"] = stats_display["F1_baseline"].map(
    lambda x: f"{x:.4f}"
)
stats_display["Delta_F1"] = stats_display["Delta_F1"].map(
    lambda x: f"{x:+.4f}"
)

stats_display["95% CI"] = stats_df.apply(
    lambda r: f"[{r['CI_low']:+.4f}, {r['CI_high']:+.4f}]",
    axis=1,
)

stats_display["p_value"] = stats_df["p_value"].map(
    lambda x: "<0.0001" if x < 1e-4 else f"{x:.4f}"
)

stats_display = stats_display[
    [
        "Condition",
        "Comparison",
        "F1_nnPU",
        "F1_baseline",
        "Delta_F1",
        "95% CI",
        "p_value",
        "Significant_95CI",
    ]
]

print("=" * 100)
print("REVIEWER 1 — COMMENT 3: PAIRED STATISTICAL SIGNIFICANCE")
print("=" * 100)
display(stats_display)

stats_display.to_csv(
    "reviewer1_comment3_statistical_significance.csv",
    index=False,
)

print("\nSaved: reviewer1_comment3_statistical_significance.csv")


# ============================================================
# 7. COMPACT INTERPRETATION HELPER
# ============================================================

print("\nINTERPRETATION")
print("-" * 100)

for _, row in stats_df.iterrows():
    ci_excludes_zero = (row["CI_low"] > 0) or (row["CI_high"] < 0)
    perm_sig = row["p_value"] < 0.05

    if row["Delta_F1"] > 0 and ci_excludes_zero and perm_sig:
        verdict = "TF-nnPU advantage statistically supported"
    elif row["Delta_F1"] > 0:
        verdict = "TF-nnPU higher, but difference not statistically conclusive"
    elif row["Delta_F1"] < 0 and ci_excludes_zero and perm_sig:
        verdict = "baseline advantage statistically supported"
    else:
        verdict = "no statistically conclusive difference"

    print(
        f"{row['Condition']:<10} | "
        f"{row['Comparison']:<24} | "
        f"ΔF1={row['Delta_F1']:+.4f} | "
        f"95% CI=[{row['CI_low']:+.4f}, {row['CI_high']:+.4f}] | "
        f"p={row['p_value']:.4g} | "
        f"{verdict}"
    )

```

## Original cell 29 (zero-based)

```python

```