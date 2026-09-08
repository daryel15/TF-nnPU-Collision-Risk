import json
import numpy as np
import pandas as pd
import torch

def save_training_artifact(out, stem, model, keys, labels, probabilities, metadata):
    keys=np.asarray(keys); labels=np.asarray(labels); probabilities=np.asarray(probabilities)
    if len(keys)!=len(labels) or len(keys)!=len(probabilities): raise ValueError('Prediction key count mismatch')
    frame=pd.DataFrame({'sequence_id':keys[:,0], 'endpoint':keys[:,1], 'prefix_length':keys[:,1]+1,
                        'y_true':labels,'y_prob':probabilities,'y_pred':(probabilities>=.5).astype(int)})
    frame.to_csv(out/(stem+'_predictions.csv'),index=False)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/(stem+'.pt'))
    (out/(stem+'_config.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8')
