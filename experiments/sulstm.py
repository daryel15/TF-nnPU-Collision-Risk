"""Matched supervised LSTM. Model equations are retained from the supplied notebook."""
from pathlib import Path

def run(train=False, smoke=False, output=None, device_name='auto'):
    import pandas as pd
    from tf_nnpu.paths import ROOT
    if not train:
        table=pd.read_csv(ROOT/'results'/'su_lstm_selected_results.csv')
        print('Recorded reference results; no training or cache construction.'); print(table.to_string(index=False))
        return table

    from tf_nnpu.paths import ROOT, DATA
    from tf_nnpu.artifacts import verify_manifest, validate_data_and_splits, load_split, load_noise_map, new_run
    from experiments.support import save_training_artifact

    verify_manifest(); validate_data_and_splits()
    out=new_run('sulstm',output,{'smoke':smoke,'device':device_name,'historical_loss':True})
    import copy
    import random
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, Subset, random_split
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score,
        recall_score, f1_score,
    )

    import matplotlib.pyplot as plt

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MIN_ENDPOINT = 3
    MAX_ENDPOINT = 14

    INPUT_DIM = 5
    HIDDEN_SIZE = 64
    NUM_LAYERS = 1
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3

    MAX_EPOCHS = 1 if smoke else 200
    EARLY_STOP_PATIENCE = 8
    EARLY_STOP_MIN_DELTA = 1e-3
    THRESHOLD = 0.5

    NOISE_CONDITIONS = [
        ("none", 0.00),
        ("p_to_u", 0.05),
        ("p_to_u", 0.10),
        ("p_to_u", 0.15),
        ("p_to_u", 0.20),
    ]

    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed(SEED)
    print("Device:", DEVICE)

    DEVICE=torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device_name=="auto" else device_name)


    def load_mix_txt(path):
        data = np.loadtxt(path)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[1] < 8:
            raise ValueError("Expected 8 columns.")
        return data

    def group_sequences_with_frame_labels(data):
        seq_col = data[:, 7].astype(int)
        sequences, frame_labels, seq_ids = [], [], []
        for sid in np.unique(seq_col):
            seq_data = data[seq_col == sid]
            seq_data = seq_data[np.argsort(seq_data[:, 0])]
            sequences.append(seq_data[:, 1:6].astype(np.float32))
            frame_labels.append(seq_data[:, 6].astype(np.float32))
            seq_ids.append(int(sid))
        return sequences, frame_labels, seq_ids

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


    # Feature standardization, following the supervised LSTM baseline practice.
    scaler = StandardScaler()

    train_frame_matrix = np.concatenate(
        [scene_to_X[sid] for sid in sorted(train_scenes)],
        axis=0,
    )

    scaler.fit(train_frame_matrix)

    scaled_scene_to_X = {
        int(sid): scaler.transform(scene_to_X[int(sid)]).astype(np.float32)
        for sid in ALL_SCENES
    }

    print("Scaler mean:", np.round(scaler.mean_, 6))
    print("Scaler scale:", np.round(scaler.scale_, 6))

    import joblib
    joblib.dump(scaler,out/"scaler.joblib")


    def make_sulstm_training_noise_map(noise_mode,noise_rate,seed):
        return load_noise_map('su_lstm_outcome',noise_mode,noise_rate,seed)


    class CommonSuLSTMDataset(Dataset):
        def __init__(self, noise_map=None):
            self.noise_map = noise_map

            # Cache all scaled prefixes once; avoids repeated sklearn transforms.
            self.items = []
            for sid, end in COMMON_KEYS:
                X = scaled_scene_to_X[sid][:end + 1]
                clean_train_y = int(scene_to_Y[sid])

                train_y = (
                    int(noise_map[(sid, end)])
                    if noise_map is not None and (sid, end) in noise_map
                    else clean_train_y
                )

                eval_current_s = int(scene_to_s[sid][end])
                eval_outcome_Y = int(scene_to_Y[sid])

                self.items.append((
                    torch.tensor(X, dtype=torch.float32),
                    torch.tensor(train_y, dtype=torch.float32),
                    torch.tensor(eval_current_s, dtype=torch.float32),
                    torch.tensor(eval_outcome_Y, dtype=torch.float32),
                ))

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]


    def collate_sulstm(batch):
        Xs, train_y, eval_s, eval_Y = zip(*batch)

        lengths = torch.tensor(
            [len(x) for x in Xs],
            dtype=torch.long,
        )

        X_padded = pad_sequence(
            Xs,
            batch_first=True,
        )

        return (
            X_padded,
            lengths,
            torch.stack(train_y),
            torch.stack(eval_s),
            torch.stack(eval_Y),
        )


    class SuLSTM(nn.Module):
        def __init__(self, input_dim=5, hidden_size=64):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.classifier = nn.Linear(hidden_size, 1)

        def forward(self, X_padded, lengths):
            packed = pack_padded_sequence(
                X_padded,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
            return self.classifier(h_n[-1]).squeeze(-1)

    model_check = SuLSTM()
    print(
        "Parameters:",
        sum(p.numel() for p in model_check.parameters() if p.requires_grad)
    )


    def safe_auc(y_true, y_prob):
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_prob)

    def binary_metrics(y_true, y_prob, threshold=0.5):
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob, dtype=float)
        y_pred = (y_prob >= threshold).astype(int)

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "auc": safe_auc(y_true, y_prob),
        }

    class EarlyStopping:
        def __init__(self, patience=8, min_delta=1e-3):
            self.patience = patience
            self.min_delta = min_delta
            self.best = -np.inf
            self.bad_epochs = 0
            self.best_state = None

        def step(self, metric, model):
            if metric > self.best + self.min_delta:
                self.best = float(metric)
                self.bad_epochs = 0
                self.best_state = copy.deepcopy(model.state_dict())
                return False
            self.bad_epochs += 1
            return self.bad_epochs >= self.patience

        def restore(self, model):
            if self.best_state is not None:
                model.load_state_dict(self.best_state)

    @torch.no_grad()
    def evaluate_sulstm(model, loader):
        model.eval()
        probs, current_s, outcome_Y = [], [], []

        for X, lengths, train_y, eval_s, eval_Y in loader:
            X = X.to(DEVICE)
            lengths = lengths.to(DEVICE)

            p = torch.sigmoid(model(X, lengths)).cpu().numpy()

            probs.append(p)
            current_s.append(eval_s.numpy())
            outcome_Y.append(eval_Y.numpy())

        y_prob = np.concatenate(probs)
        y_s = np.concatenate(current_s).astype(int)
        y_Y = np.concatenate(outcome_Y).astype(int)

        return {
            "_y_true": y_s, "_y_prob": y_prob,
            "current_risk": binary_metrics(y_s, y_prob, THRESHOLD),
            "scenario_outcome": binary_metrics(y_Y, y_prob, THRESHOLD),
        }


    def train_sulstm_prefix_split(noise_map, seed):
        set_seed(seed)

        full_ds = CommonSuLSTMDataset(noise_map=noise_map)

        train_ds = Subset(full_ds, TRAIN_IDX.tolist())
        val_ds = Subset(full_ds, VAL_IDX.tolist())

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_sulstm,
            num_workers=0,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=128,
            shuffle=False,
            collate_fn=collate_sulstm,
            num_workers=0,
        )

        model = SuLSTM(INPUT_DIM, HIDDEN_SIZE).to(DEVICE)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        criterion = nn.BCEWithLogitsLoss()

        stopper = EarlyStopping(
            patience=EARLY_STOP_PATIENCE,
            min_delta=EARLY_STOP_MIN_DELTA,
        )

        best_epoch = None

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()

            for X, lengths, train_y, eval_s, eval_Y in train_loader:
                X = X.to(DEVICE)
                lengths = lengths.to(DEVICE)
                train_y = train_y.to(DEVICE)

                optimizer.zero_grad()
                logits = model(X, lengths)
                loss = criterion(logits, train_y)
                loss.backward()
                optimizer.step()

            val_eval = evaluate_sulstm(model, val_loader)
            val_f1 = val_eval["current_risk"]["f1"]

            if val_f1 > stopper.best + stopper.min_delta:
                best_epoch = epoch

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"epoch={epoch:03d} | "
                    f"current-risk F1={val_f1:.4f} | "
                    f"legacy outcome F1={val_eval['scenario_outcome']['f1']:.4f}"
                )

            if stopper.step(val_f1, model):
                break

        stopper.restore(model)
        final_eval = evaluate_sulstm(model, val_loader)

        return {
            "_model": model, "_y_true": final_eval["_y_true"], "_y_prob": final_eval["_y_prob"],
            "best_epoch": best_epoch,
            "current_risk": final_eval["current_risk"],
            "scenario_outcome": final_eval["scenario_outcome"],
        }



    rows=[]
    for condition_idx,(mode,rate) in enumerate(NOISE_CONDITIONS):
        if smoke and condition_idx>0: break
        corruption_seed=SEED+1000*condition_idx
        noise_map,stats=make_sulstm_training_noise_map(mode,rate,corruption_seed)
        model_seed=SEED+5000+condition_idx
        res=train_sulstm_prefix_split(noise_map,seed=model_seed)
        meta={'method':'Su-LSTM','noise_mode':mode,'noise_rate':rate,'corruption_seed':corruption_seed,
              'model_seed':model_seed,'best_epoch':res['best_epoch'],'smoke':smoke,
              'training_target':'scenario_outcome','evaluation_target':'observed_PU_label_s_t'}
        save_training_artifact(out,f'{condition_idx:02d}_Su-LSTM',res['_model'],VAL_KEYS,res['_y_true'],res['_y_prob'],meta)
        rows.append({**meta,**res['current_risk'],'outcome_f1':res['scenario_outcome']['f1'],**stats})
        pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    print('Outputs:',out)
    return pd.DataFrame(rows)
