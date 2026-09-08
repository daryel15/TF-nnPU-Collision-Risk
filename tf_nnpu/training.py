"""Seeded reruns on released inputs. These do not recreate missing historical RNG states."""
import copy
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from .paths import DATA
from .data import CurrentRiskPrefixDataset, UnsupervisedSeqDataset, pad_collate_supervised, pad_collate_unsupervised
from .models import TemporalEncoder, TemporalAutoencoder, TemporalRiskTransformer
from .losses import nnpu_loss
from .artifacts import load_split
from .evaluation import predict, metric_dict

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class ContextDataset(Dataset):
    def __init__(self,base,context): self.base,self.context=base,context
    def __len__(self): return len(self.base)
    def __getitem__(self,i):
        item=self.base[i]
        if isinstance(item,tuple): return item[0][-self.context:],item[1]
        return item[-self.context:]

def pretrain(device,seed=42,epochs=20,smoke=False,context=15,num_layers=2,nhead=4):
    seed_all(seed)
    model=TemporalAutoencoder(TemporalEncoder(num_layers=num_layers,nhead=nhead)).to(device)
    data=ContextDataset(UnsupervisedSeqDataset(DATA),context)
    if smoke: data=Subset(data,list(range(32)))
    loader=DataLoader(data,batch_size=16,shuffle=True,collate_fn=pad_collate_unsupervised)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    history=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for x,mask in loader:
            x,mask=x.to(device),mask.to(device)
            loss=F.mse_loss(model(x,mask),x)
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        history.append({'epoch':epoch,'reconstruction_mse':float(np.mean(losses))})
        print('Stage 1',history[-1])
    return model.encoder,history

def fine_tune(encoder,device,seed=42,epochs=200,smoke=False,context=15,lr=1e-4,loss_mode='nnpu',flip_indices=None):
    seed_all(seed)
    data=CurrentRiskPrefixDataset(DATA)
    if flip_indices is not None:
        data.labels=data.labels.clone(); data.labels[flip_indices]=1-data.labels[flip_indices]
    split=load_split(); ti=split['train_idx'][:64] if smoke else split['train_idx']; vi=split['val_idx'][:64] if smoke else split['val_idx']
    tr=DataLoader(ContextDataset(Subset(data,ti.tolist()),context),batch_size=32,shuffle=True,collate_fn=pad_collate_supervised)
    va=DataLoader(ContextDataset(Subset(data,vi.tolist()),context),batch_size=64,shuffle=False,collate_fn=pad_collate_supervised)
    model=TemporalRiskTransformer(copy.deepcopy(encoder)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    best=-np.inf; bad=0; state=None; best_epoch=None; history=[]
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for x,y,mask in tr:
            x,y,mask=x.to(device),y.to(device),mask.to(device)
            logits,_=model(x,mask)
            if loss_mode=='nnpu': loss=nnpu_loss(logits,y,prior=.1,gamma=1.,beta=0.)
            elif loss_mode=='wbce': loss=F.binary_cross_entropy_with_logits(logits,y,weight=torch.where(y==1,3.,1.))
            elif loss_mode=='focal':
                p=torch.sigmoid(logits); pt=p*y+(1-p)*(1-y); alpha=.25*y+.75*(1-y)
                loss=(alpha*(1-pt)**2*F.binary_cross_entropy_with_logits(logits,y,reduction='none')).mean()
            else: raise ValueError(loss_mode)
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        y,p=predict(model,va,device); metrics=metric_dict(y,p)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'validation_f1':metrics['f1']})
        if metrics['f1']>best+1e-3:
            best=metrics['f1'];bad=0;state=copy.deepcopy(model.state_dict());best_epoch=epoch
        else: bad+=1
        print('Stage 2',history[-1])
        if bad>=8: break
    model.load_state_dict(state)
    y,p=predict(model,va,device)
    return model,metric_dict(y,p),y,p,split['keys'][vi],history,best_epoch
