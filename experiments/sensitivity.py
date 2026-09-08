"""Hyperparameter reruns on the released dataset. Model equations are retained from the supplied notebook."""
from pathlib import Path

def run(train=False, smoke=False, output=None, device_name='auto'):
    import pandas as pd
    from tf_nnpu.paths import ROOT
    if not train:
        table=pd.read_csv(ROOT/'results'/'hyperparameter_sensitivity.csv')
        print('Recorded reference results; no training or cache construction.'); print(table.to_string(index=False))
        return table

    from tf_nnpu.paths import ROOT, DATA
    from tf_nnpu.artifacts import verify_manifest, validate_data_and_splits, load_split, load_noise_map, new_run
    from experiments.support import save_training_artifact

    verify_manifest(); validate_data_and_splits()
    out=new_run('sensitivity',output,{'smoke':smoke,'device':device_name,'historical_loss':True})

    import json
    import numpy as np
    import torch
    from tf_nnpu.paths import ENCODER
    from tf_nnpu.models import TemporalEncoder
    from tf_nnpu.training import pretrain, fine_tune
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name)

    configs=[]
    for key,values in [('num_layers',[1,2,3]),('nhead',[1,2,4,8]),('context',[5,10,15]),('lr',[1e-5,5e-5,1e-4,5e-4,1e-3])]:
        for value in values:
            config={'num_layers':2,'nhead':4,'context':15,'lr':1e-4};config[key]=value
            configs.append((key,value,config))
    if smoke: configs=[('smoke_baseline',0,{'num_layers':2,'nhead':4,'context':15,'lr':1e-4})]
    rows=[]
    for key,value,config in configs:
        encoder,h1=pretrain(device,epochs=1 if smoke else 20,smoke=smoke,context=config['context'],num_layers=config['num_layers'],nhead=config['nhead'])
        label=f'{key}_{value}'
        torch.save(encoder.state_dict(),out/(label+'_encoder.pt'))
        model,metrics,y,p,keys,history,best=fine_tune(encoder,device,epochs=1 if smoke else 200,smoke=smoke,context=config['context'],lr=config['lr'])
        meta={'varied_parameter':key,'value':value,**config,'seed':42,'best_epoch':best,'smoke':smoke,'status':'new_seeded_rerun'}

        stem=label.replace(' ','_')
        save_training_artifact(out,stem,model,keys,y,p,meta)
        pd.DataFrame(h1).to_csv(out/(stem+'_stage1_history.csv'),index=False)
        pd.DataFrame(history).to_csv(out/(stem+'_stage2_history.csv'),index=False)
        row={**meta,**{k:v for k,v in metrics.items() if k!='confusion_matrix'}}
        rows.append(row);pd.DataFrame(rows).to_csv(out/'results.csv',index=False)

    print('Outputs:',out)
    return pd.DataFrame(rows)
