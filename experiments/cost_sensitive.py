"""Cost-sensitive reruns and paired statistics. Model equations are retained from the supplied notebook."""
from pathlib import Path

def run(train=False, smoke=False, output=None, device_name='auto'):
    import pandas as pd
    from tf_nnpu.paths import ROOT
    if not train:
        table=pd.read_csv(ROOT/'results'/'cost_sensitive_symmetric_noise_f1.csv')
        print('Recorded reference results; no training or cache construction.'); print(table.to_string(index=False))
        return table

    from tf_nnpu.paths import ROOT, DATA
    from tf_nnpu.artifacts import verify_manifest, validate_data_and_splits, load_split, load_noise_map, new_run
    from experiments.support import save_training_artifact

    verify_manifest(); validate_data_and_splits()
    out=new_run('cost_sensitive',output,{'smoke':smoke,'device':device_name,'historical_loss':True})

    import json
    import numpy as np
    import torch
    from tf_nnpu.paths import ENCODER
    from tf_nnpu.models import TemporalEncoder
    from tf_nnpu.training import pretrain, fine_tune
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device_name=='auto' else device_name)

    from tf_nnpu.artifacts import mask_path
    from tf_nnpu.statistics import compare_prediction_files
    encoder,h1=pretrain(device,epochs=1 if smoke else 20,smoke=smoke)
    torch.save(encoder.state_dict(),out/'stage1_encoder.pt')
    rows=[]; significance=[]
    rates=[0.] if smoke else [0.,.05,.10,.15,.20]
    for condition_idx,rate in enumerate(rates):
        flips=None if rate==0 else np.load(mask_path('cost_sensitive_symmetric','symmetric',rate,123),allow_pickle=False)
        for method_idx,(method,loss_mode) in enumerate([('TF-nnPU','nnpu'),('TF-WBCE','wbce'),('TF-Focal','focal')]):
            seed=42+100*condition_idx+method_idx
            model,metrics,y,p,keys,history,best=fine_tune(encoder,device,seed=seed,epochs=1 if smoke else 200,smoke=smoke,loss_mode=loss_mode,flip_indices=flips)
            label=f'{round(rate*100):02d}_{method}'
            meta={'method':method,'noise_rate':rate,'seed':seed,'corruption_seed':123,'best_epoch':best,'smoke':smoke,
                  'dataset':'released safe_col_mix.txt','status':'new_seeded_rerun; historical dth_9 file identity unverified'}

            stem=label.replace(' ','_')
            save_training_artifact(out,stem,model,keys,y,p,meta)
            pd.DataFrame(h1).to_csv(out/(stem+'_stage1_history.csv'),index=False)
            pd.DataFrame(history).to_csv(out/(stem+'_stage2_history.csv'),index=False)
            row={**meta,**{k:v for k,v in metrics.items() if k!='confusion_matrix'}}
            rows.append(row);pd.DataFrame(rows).to_csv(out/'results.csv',index=False)

        for method in ['TF-WBCE','TF-Focal']:
            stem=f'{round(rate*100):02d}'
            significance.append(compare_prediction_files(out/(stem+'_TF-nnPU_predictions.csv'),out/(stem+'_'+method+'_predictions.csv'),method,f'{rate:.0%} noise',100 if smoke else 10000))
        pd.DataFrame(significance).to_csv(out/'statistical_significance.csv',index=False)
    print('Outputs:',out)
    return pd.DataFrame(rows)
