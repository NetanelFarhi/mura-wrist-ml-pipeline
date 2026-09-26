"""Bounded Phase 2B training on cached frozen embeddings (train/val only)."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader,TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score,average_precision_score
from src.models.multiview_classifier import MultiViewEmbeddingClassifier
from src.evaluation.phase2a import operating_metrics,select_sensitivity_constrained_threshold

def load(path):
 d=np.load(path);return tuple(torch.from_numpy(d[k]) for k in ('embeddings','mask','labels'))+(d['patient_ids'],d['study_ids'],d['num_views'])
def predict(model,x,m):
 model.eval();out=[];att=[]
 with torch.no_grad():
  for i in range(0,len(x),64):
   logits,w=model(x[i:i+64],m[i:i+64]);out.append(torch.sigmoid(logits));att.append(w.cpu() if w is not None else None)
 return torch.cat(out).numpy(),(torch.cat([z for z in att if z is not None]).numpy() if att[0] is not None else None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--train',default='models/phase2b/train_frozen_embeddings.npz');p.add_argument('--val',default='models/phase2b/val_frozen_embeddings.npz');p.add_argument('--output',default='models/phase2b');p.add_argument('--seed',type=int,default=23);a=p.parse_args();torch.manual_seed(a.seed);np.random.seed(a.seed)
 tx,tm,ty,*_=load(a.train);vx,vm,vy,vp,vs,vn=load(a.val);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);loader=DataLoader(TensorDataset(tx,tm,ty.float()),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(a.seed));pos=(ty==0).sum().item()/(ty==1).sum().item();rows=[]
 for pool in ('mean','max','attention'):
  model=MultiViewEmbeddingClassifier(pool).float();opt=AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);sched=ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3,min_lr=1e-6);lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos));best=-1;wait=0;history=[];started=time.perf_counter()
  for epoch in range(1,41):
   model.train();losses=[]
   for x,m,y in loader:
    opt.zero_grad();logits,_=model(x,m);loss=lossfn(logits,y);loss.backward();opt.step();losses.append(loss.item())
   prob,_=predict(model,vx.float(),vm.bool());auc=roc_auc_score(vy.numpy(),prob);sched.step(auc);history.append({'epoch':epoch,'loss':float(np.mean(losses)),'val_auroc':float(auc),'lr':opt.param_groups[0]['lr']})
   if auc>best:best=auc;wait=0;torch.save({'state':model.state_dict(),'pooling':pool,'best_epoch':epoch},out/f'{pool}_embedding.pt')
   else:wait+=1
   if wait>=7:break
  ck=torch.load(out/f'{pool}_embedding.pt',map_location='cpu');model.load_state_dict(ck['state']);prob,weights=predict(model,vx.float(),vm.bool());base={'model':pool,'auroc':float(roc_auc_score(vy.numpy(),prob)),'auprc':float(average_precision_score(vy.numpy(),prob)),'best_epoch':ck['best_epoch'],'training_seconds':time.perf_counter()-started,'total_parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
  for target in (.85,.90):base[f'specificity_at_{target}']=select_sensitivity_constrained_threshold(vy.numpy(),prob,target)['specificity']
  metrics=operating_metrics(vy.numpy(),prob,.5);base.update({f'at_0_5_{k}':v for k,v in metrics.items() if k in ('sensitivity','specificity','f1','false_negative_count','false_positive_count')});rows.append(base)
  frame=pd.DataFrame({'patient_id':vp,'study_id':vs,'label':vy.numpy(),'num_views':vn,'study_probability':prob});frame.to_csv(out/f'{pool}_validation_predictions.csv',index=False)
  if weights is not None:
   index=pd.read_csv(out/'val_embedding_view_index.csv'); weight_rows=[]
   for _,r in index.iterrows(): weight_rows.append({**r.to_dict(),'attention_weight':float(weights[int(r.study_index),int(r.image_view_id)-1])})
   pd.DataFrame(weight_rows).to_csv(out/'attention_weights_validation.csv',index=False)
  (out/f'{pool}_history.json').write_text(json.dumps(history,indent=2))
 pd.DataFrame(rows).to_csv(out/'model_metrics.csv',index=False)
if __name__=='__main__':main()
