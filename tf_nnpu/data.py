from pathlib import Path
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
