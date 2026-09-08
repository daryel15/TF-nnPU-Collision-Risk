from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'safe_col_mix.txt'
ENCODER = ROOT / 'checkpoints' / 'pretrained_encoder1.pt'
MODEL = ROOT / 'checkpoints' / 'stage2_trained_model.pt'
