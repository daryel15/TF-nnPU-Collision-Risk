import hashlib
import json
from datetime import datetime, timezone
import platform
from pathlib import Path
import numpy as np
import torch
from .paths import ROOT, DATA

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def verify_manifest():
    lines=(ROOT/'MANIFEST_SHA256.txt').read_text().splitlines()
    for line in lines:
        expected, name=line.split('  ',1)
        p=ROOT/name
        if not p.is_file() or digest(p)!=expected:
            raise ValueError(f'Artifact hash mismatch or missing file: {name}')
    return len(lines)

def load_split(name='main'):
    filename={'main':'main_prefix_split_seed42.npz', 'common':'common_t4_t15_split_seed42.npz'}[name]
    with np.load(ROOT/'splits'/filename,allow_pickle=False) as z:
        return {k:z[k] for k in z.files}

def mask_path(group, mode, rate, seed):
    return ROOT/'noise_masks'/group/f'{mode}_{round(rate*100):02d}_seed{seed}.npy'

def load_noise_map(group, mode, rate, seed):
    z=load_split('common'); keys=z['keys'][z['train_idx']]
    label='scenario_outcome_labels' if group=='su_lstm_outcome' else 'current_risk_labels'
    clean=z[label][z['train_idx']]; noisy=clean.copy()
    if mode!='none' and rate:
        selected=np.load(mask_path(group,mode,rate,seed),allow_pickle=False)
        lookup={tuple(k):i for i,k in enumerate(keys)}
        local=np.asarray([lookup[tuple(k)] for k in selected],dtype=int)
        if mode=='p_to_u' and not np.all(clean[local]==1): raise ValueError('P-to-U mask selects unlabeled samples')
        noisy[local]=1-noisy[local] if mode=='symmetric' else 0
    p_to_u=int(np.sum((clean==1)&(noisy==0))); u_to_p=int(np.sum((clean==0)&(noisy==1)))
    stats={'n_train':len(keys),'clean_positive':int(clean.sum()),'final_positive':int(noisy.sum()),
           'p_to_u':p_to_u,'u_to_p':u_to_p,'flipped_total':int(np.sum(clean!=noisy)),
           'realized_flip_rate':float(np.mean(clean!=noisy)),
           'fraction_of_positive_labels_removed':float(p_to_u/max(1,clean.sum()))}
    return {tuple(k):int(y) for k,y in zip(keys,noisy)}, stats

def validate_data_and_splits():
    from .data import load_mix_txt, group_sequences_with_frame_labels
    a=load_mix_txt(DATA)
    if a.shape!=(14715,8) or not np.isfinite(a).all(): raise ValueError('Unexpected dataset shape/values')
    if not np.isin(a[:,6],[0,1]).all(): raise ValueError('Labels must be 0/1')
    seqs,labels,ids=group_sequences_with_frame_labels(a)
    if len(ids)!=981 or any(len(s)!=15 for s in seqs): raise ValueError('Unexpected sequences')
    for name,start in [('main',1),('common',3)]:
        z=load_split(name); keys=np.array([(sid,end) for sid in ids for end in range(start,15)])
        y=np.array([int(v[end]) for v in labels for end in range(start,15)])
        label='labels' if name=='main' else 'current_risk_labels'
        if not np.array_equal(keys,z['keys']) or not np.array_equal(y,z[label]): raise ValueError('Split sample order mismatch')
        perm=torch.randperm(len(keys),generator=torch.Generator().manual_seed(42)).numpy()
        n=int(.8*len(keys))
        if not np.array_equal(z['train_idx'],perm[:n]) or not np.array_equal(z['val_idx'],perm[n:]): raise ValueError('Split indices mismatch')
        if name=='common':
            outcome=np.array([int(v.max()) for v in labels for _ in range(start,15)])
            if not np.array_equal(outcome,z['scenario_outcome_labels']): raise ValueError('Outcome label mismatch')
    count=0
    for entry in json.loads((ROOT/'noise_masks/index.json').read_text()):
        group,mode,rate,seed=[entry[k] for k in ['group','mode','rate','seed']]
        z=load_split('main' if group=='cost_sensitive_symmetric' else 'common')
        n=len(z['train_idx'])
        if group=='cost_sensitive_symmetric':
            idx=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).numpy()[:round(rate*n)]
            expected=z['train_idx'][idx]
        else:
            label='scenario_outcome_labels' if group=='su_lstm_outcome' else 'current_risk_labels'
            y=z[label][z['train_idx']]
            candidates=np.where(y==1)[0] if mode=='p_to_u' else np.arange(n)
            idx=np.random.default_rng(seed).permutation(candidates)[:round(rate*len(candidates))]
            expected=z['keys'][z['train_idx']][idx]
        actual=np.load(mask_path(group,mode,rate,seed),allow_pickle=False)
        if not np.array_equal(actual,expected): raise ValueError(f'Mask mismatch: {group}/{mode}/{seed}')
        count+=1
    return {'rows':len(a),'sequences':len(ids),'splits':2,'masks':count}

def new_run(name, output=None, config=None):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    p=Path(output).resolve() if output else ROOT/'outputs'/f'{name}_{stamp}'
    # Never write generated results over the released inputs or reference results.
    for d in ['data','checkpoints','splits','noise_masks','results','archive','docs','tf_nnpu','notebooks']:
        if p==ROOT/d or (ROOT/d) in p.parents: raise ValueError('Choose an output directory outside release inputs')
    p.mkdir(parents=True,exist_ok=False)
    info={'experiment':name,'utc':stamp,'python':platform.python_version(),'torch':torch.__version__,
          'cuda':torch.version.cuda,'dataset_sha256':digest(DATA),'config':config or {}}
    (p/'run.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return p
