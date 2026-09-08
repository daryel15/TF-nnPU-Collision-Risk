import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from .paths import ROOT, DATA, MODEL
from .models import TemporalEncoder, TemporalRiskTransformer
from .data import CurrentRiskPrefixDataset, pad_collate_supervised
from .artifacts import verify_manifest, validate_data_and_splits, load_split, new_run

def metric_dict(y,p):
    pred=(p>=.5).astype(int)
    return {'accuracy':float(accuracy_score(y,pred)), 'auc':float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
            'precision':float(precision_score(y,pred,zero_division=0)), 'recall':float(recall_score(y,pred,zero_division=0)),
            'f1':float(f1_score(y,pred,zero_division=0)), 'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist()}

@torch.no_grad()
def predict(model,loader,device):
    model.eval(); probs=[]; labels=[]
    for x,y,mask in loader:
        logits,_=model(x.to(device),pad_mask=mask.to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy()); labels.append(y.numpy())
    return np.concatenate(labels).astype(int),np.concatenate(probs)

def evaluate_checkpoint(device='auto', output=None):
    verify_manifest(); validate_data_and_splits()
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)
    model=TemporalRiskTransformer(TemporalEncoder()).to(device)
    model.load_state_dict(torch.load(MODEL,map_location=device,weights_only=True))
    split=load_split(); ds=CurrentRiskPrefixDataset(DATA)
    loader=DataLoader(Subset(ds,split['val_idx'].tolist()),batch_size=64,collate_fn=pad_collate_supervised)
    y,p=predict(model,loader,device); metrics=metric_dict(y,p)
    expected=json.loads((ROOT/'checkpoints/expected_metrics.json').read_text())
    for k in ['accuracy','auc','precision','recall','f1']:
        if abs(metrics[k]-expected[k])>1e-6: raise AssertionError(f'{k}: expected {expected[k]}, got {metrics[k]}')
    if metrics['confusion_matrix']!=expected['confusion_matrix']: raise AssertionError('Confusion matrix mismatch')
    out=new_run('checkpoint_evaluation',output,{'device':str(device),'threshold':.5,'target':'observed_PU_label_s_t'})
    keys=split['keys'][split['val_idx']]
    pd.DataFrame({'prefix_index':split['val_idx'],'sequence_id':keys[:,0],'endpoint':keys[:,1],
                  'prefix_length':keys[:,1]+1,'y_true':y,'y_prob':p,'y_pred':(p>=.5).astype(int)}).to_csv(out/'predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2)); print('Checkpoint reproduction: PASS'); print('Outputs:',out)
    return metrics,out
