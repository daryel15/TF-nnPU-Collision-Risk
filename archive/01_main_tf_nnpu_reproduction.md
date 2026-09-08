# Historical source: 01_main_tf_nnpu_reproduction.ipynb


Reference only. The runnable entry point is in `notebooks/`.


# TF-nnPU Level-1 Reproducibility

This notebook starts from the released processed feature dataset and preserves the original prefix construction and 80/20 random prefix split. It first verifies the released final checkpoint and then provides optional Stage-2 retraining from the canonical pretrained encoder.


## Original cell 1 (zero-based)

```python
from pathlib import Path
import os, math, random, hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "safe_col_mix.txt"
PRETRAINED_ENCODER_PATH = ROOT / "checkpoints" / "pretrained_encoder1.pt"
FINAL_MODEL_PATH = ROOT / "checkpoints" / "stage2_trained_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Python/PyTorch device:", device)
print("PyTorch:", torch.__version__)

```

## 1. Verify canonical file hashes

## Original cell 3 (zero-based)

```python
EXPECTED_SHA256 = {
    "data": "3fdb5067fd6af8003381c978b42000010fa4f09aaa9504a4f812389be235d963",
    "pretrained": "9416f9374bdea1aebd528073714047ef5c45f3e494eeb482d02a4daddad889d8",
    "stage2": "f20b3f070f72b14fdd7a5c2e180b8badf548e97aed18806e82f463bcf410f1f4",
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

for name, path in [
    ("data", DATA_PATH),
    ("pretrained", PRETRAINED_ENCODER_PATH),
    ("stage2", FINAL_MODEL_PATH),
]:
    digest = sha256(path)
    print(name, digest, "OK" if digest == EXPECTED_SHA256[name] else "MISMATCH")

```

## 2. Data loading and causal prefix construction

## Original cell 5 (zero-based)

```python
def load_mix_txt(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]
    return data

def group_sequences_with_frame_labels(data):
    seq_col = data[:, 7].astype(int)
    unique_ids = np.unique(seq_col)

    sequences, frame_labels, seq_ids = [], [], []
    for sid in unique_ids:
        seq_data = data[seq_col == sid]
        seq_data = seq_data[np.argsort(seq_data[:, 0])]  # preserve original sorting
        X = seq_data[:, 1:6].astype(np.float32)
        y = seq_data[:, 6].astype(np.float32)
        sequences.append(X)
        frame_labels.append(y)
        seq_ids.append(int(sid))
    return sequences, frame_labels, seq_ids

def build_current_risk_prefixes(sequences, frame_labels, min_len=2, max_len=None):
    prefix_seqs, prefix_labels = [], []
    for X, y in zip(sequences, frame_labels):
        T = X.shape[0]
        if T < min_len:
            continue
        L_max = T if max_len is None else min(max_len, T)
        for L in range(min_len, L_max + 1):
            prefix_seqs.append(X[:L].astype(np.float32))
            prefix_labels.append(float(y[L - 1]))
    return prefix_seqs, np.asarray(prefix_labels, dtype=np.float32)

class CurrentRiskPrefixDataset(Dataset):
    def __init__(self, txt_path, min_len=2, max_len=None):
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

def pad_collate_supervised(batch):
    seqs, labels = zip(*batch)
    lengths = [x.shape[0] for x in seqs]
    padded = pad_sequence(seqs, batch_first=True)
    T_max = padded.shape[1]
    idxs = torch.arange(T_max).unsqueeze(0).expand(len(lengths), -1)
    lengths_tensor = torch.tensor(lengths).unsqueeze(1)
    pad_mask = idxs >= lengths_tensor
    return padded, torch.stack(labels), pad_mask

```

## Original cell 6 (zero-based)

```python
raw = load_mix_txt(DATA_PATH)
sequences, frame_labels, seq_ids = group_sequences_with_frame_labels(raw)

print("Rows:", len(raw))
print("Sequences:", len(sequences))
print("Sequence lengths:", sorted(set(len(x) for x in sequences)))

sup_ds = CurrentRiskPrefixDataset(DATA_PATH, min_len=2, max_len=None)
print("Prefixes:", len(sup_ds))

assert len(raw) == 14715
assert len(sequences) == 981
assert len(sup_ds) == 13734
assert set(len(x) for x in sequences) == {15}

```

## 3. Preserve the original 80/20 random prefix split

## Original cell 8 (zero-based)

```python
# IMPORTANT: this is intentionally the same split protocol used in the research notebook.
torch.manual_seed(42)

n_total = len(sup_ds)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_ds, val_ds = random_split(sup_ds, [n_train, n_val])

train_positive = int(sum(float(train_ds[i][1]) for i in range(len(train_ds))))
val_positive = int(sum(float(val_ds[i][1]) for i in range(len(val_ds))))

print("Train:", len(train_ds), "positive:", train_positive)
print("Validation:", len(val_ds), "positive:", val_positive)

assert len(train_ds) == 10987
assert len(val_ds) == 2747
assert train_positive == 2717
assert val_positive == 660

train_loader = DataLoader(
    train_ds, batch_size=32, shuffle=True, collate_fn=pad_collate_supervised
)
val_loader = DataLoader(
    val_ds, batch_size=64, shuffle=False, collate_fn=pad_collate_supervised
)

```

## 4. Model definition

## Original cell 10 (zero-based)

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TemporalEncoder(nn.Module):
    def __init__(
        self, input_dim=5, d_model=64, nhead=4, num_layers=2,
        dim_feedforward=128, dropout=0.1, max_len=500
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x, pad_mask=None):
        h = self.in_proj(x)
        h = self.pos(h)
        return self.encoder(h, src_key_padding_mask=pad_mask)

class TemporalRiskTransformer(nn.Module):
    def __init__(self, encoder, d_hidden=64, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        d_model = encoder.d_model
        self.pool = nn.Linear(d_model, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x, pad_mask=None):
        h = self.encoder(x, pad_mask=pad_mask)
        att_logits = self.pool(h).squeeze(-1)
        if pad_mask is not None:
            att_logits = att_logits.masked_fill(pad_mask, float("-inf"))
        att_weights = F.softmax(att_logits, dim=1)
        z_seq = torch.sum(h * att_weights.unsqueeze(-1), dim=1)
        logits = self.head(z_seq).squeeze(-1)
        return logits, z_seq

```

## 5. Exact evaluation of the released Stage-2 checkpoint

## Original cell 12 (zero-based)

```python
def evaluate(model, loader, threshold=0.5):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for X, y, pad_mask in loader:
            X = X.to(device)
            pad_mask = pad_mask.to(device)
            logits, _ = model(X, pad_mask=pad_mask)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())

    y_prob = np.concatenate(probs)
    y_true = np.concatenate(labels).astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

eval_model = TemporalRiskTransformer(TemporalEncoder()).to(device)
eval_model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
metrics = evaluate(eval_model, val_loader, threshold=0.5)

for k, v in metrics.items():
    print(k, v)

```

## Original cell 13 (zero-based)

```python
# Sanity checks against the released canonical checkpoint
assert abs(metrics["accuracy"] - 0.9606843829632327) < 1e-6
assert abs(metrics["auc"] - 0.990302159109059) < 1e-6
assert abs(metrics["precision"] - 0.9380952380952381) < 1e-6
assert abs(metrics["recall"] - 0.8954545454545455) < 1e-6
assert abs(metrics["f1"] - 0.9162790697674419) < 1e-6
assert metrics["confusion_matrix"].tolist() == [[2048, 39], [69, 591]]
print("Checkpoint reproduction: PASS")

```

## 6. Optional Stage-2 retraining from the canonical pretrained encoder

This reproduces the **training procedure** from the released pretrained encoder while preserving the same dataset and split. The released `stage2_trained_model.pt` remains the canonical artifact for exact numerical reproduction of the historical trained model.


## Original cell 15 (zero-based)

```python
def nnpu_loss(logits, targets, prior=0.10, gamma=1.0, beta=0.0):
    y = targets.float()
    logits = logits.view(-1)

    pos_mask = (y == 1)
    unl_mask = (y == 0)

    def loss_pos(z):
        return F.softplus(-z)

    def loss_neg(z):
        return F.softplus(z)

    device_local = logits.device

    if pos_mask.any():
        pos_logits = logits[pos_mask]
        L_p_pos = loss_pos(pos_logits).mean()
        L_p_neg = loss_neg(pos_logits).mean()
    else:
        L_p_pos = torch.tensor(0.0, device=device_local)
        L_p_neg = torch.tensor(0.0, device=device_local)

    if unl_mask.any():
        L_u_neg = loss_neg(logits[unl_mask]).mean()
    else:
        L_u_neg = torch.tensor(0.0, device=device_local)

    risk_positive = prior * L_p_pos
    risk_negative = L_u_neg - prior * L_p_neg

    if risk_negative.item() < -beta:
        return risk_positive - gamma * risk_negative
    return risk_positive + risk_negative

class EarlyStopping:
    def __init__(self, patience=8, min_delta=1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.bad_epochs = 0
        self.best_state = None

    def step(self, metric, model):
        if metric > self.best + self.min_delta:
            self.best = metric
            self.bad_epochs = 0
            self.best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

```

## Original cell 16 (zero-based)

```python
RUN_STAGE2_RETRAINING = False

if RUN_STAGE2_RETRAINING:
    encoder = TemporalEncoder()
    encoder.load_state_dict(torch.load(PRETRAINED_ENCODER_PATH, map_location="cpu"))
    model = TemporalRiskTransformer(encoder).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4
    )
    stopper = EarlyStopping(patience=8, min_delta=1e-3)

    for epoch in range(1, 201):
        model.train()
        running_loss = 0.0

        for X, y, pad_mask in train_loader:
            X = X.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)

            logits, _ = model(X, pad_mask=pad_mask)
            loss = nnpu_loss(logits, y, prior=0.10, gamma=1.0, beta=0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        current = evaluate(model, val_loader, threshold=0.5)
        print(
            f"Epoch {epoch:03d} | loss={running_loss/len(train_loader):.4f} "
            f"| val_F1={current['f1']:.4f} | val_AUC={current['auc']:.4f}"
        )

        if stopper.step(current["f1"], model):
            print("Early stopping")
            break

    stopper.restore(model)
    retrained_metrics = evaluate(model, val_loader, threshold=0.5)
    print(retrained_metrics)

```