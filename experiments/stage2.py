"""Stage-2 rerun from the released encoder. Model equations are retained from the supplied notebook."""
from pathlib import Path

def run(train=False, smoke=False, output=None, device_name='auto'):
    import pandas as pd
    from tf_nnpu.paths import ROOT
    if not train:
        table=pd.read_csv(ROOT/'results'/'main_tf_nnpu_metrics.csv')
        print('Recorded reference results; no training or cache construction.'); print(table.to_string(index=False))
        return table

    from tf_nnpu.paths import ROOT, DATA
    from tf_nnpu.artifacts import verify_manifest, validate_data_and_splits, load_split, load_noise_map, new_run
    from experiments.support import save_training_artifact

    verify_manifest(); validate_data_and_splits()
    out=new_run('stage2',output,{'smoke':smoke,'device':device_name,'historical_loss':True})

    import json
    import numpy as np
    import torch
    from tf_nnpu.paths import ENCODER
    from tf_nnpu.models import TemporalEncoder
    from tf_nnpu.training import pretrain, fine_tune
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name)

    encoder=TemporalEncoder();encoder.load_state_dict(torch.load(ENCODER,map_location='cpu',weights_only=True))
    model,metrics,y,p,keys,history,best=fine_tune(encoder,device,epochs=1 if smoke else 200,smoke=smoke)
    meta={'seed':42,'best_epoch':best,'initialization':'released pretrained_encoder1.pt','smoke':smoke}
    save_training_artifact(out,'stage2',model,keys,y,p,meta)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print('Outputs:',out)
    return pd.DataFrame([{k:v for k,v in metrics.items() if k!='confusion_matrix'}])
