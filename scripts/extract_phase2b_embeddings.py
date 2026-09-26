"""Cache frozen Phase-1 Model-B 512-D real-view embeddings for train/validation only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from src.data.transforms import build_eval_transform
from src.models.resnet18_classifier import build_resnet18_classifier

class ImageRows(Dataset):
    def __init__(self, rows, root): self.rows=rows.reset_index(drop=True); self.root=Path(root); self.transform=build_eval_transform( image_size=224)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows.iloc[i]
        with Image.open(self.root/row.image_path) as im: image=self.transform(im.copy())
        return image,i

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--manifest',default='data/manifests/wrist_manifest.csv');p.add_argument('--split-manifest',default='data/manifests/split_manifest.csv');p.add_argument('--checkpoint',default='models/resnet18/best_resnet18_study_level.pt');p.add_argument('--output-dir',default='models/phase2b');p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device',default='cpu');a=p.parse_args()
 manifest=pd.read_csv(a.manifest); split=pd.read_csv(a.split_manifest); rows=manifest.merge(split,on=['patient_id','study_id'],validate='many_to_one')
 if not set(rows.final_split.unique()).issubset({'train','val','test'}): raise ValueError('unexpected split')
 device=torch.device(a.device); model=build_resnet18_classifier(weights='imagenet',allow_weight_download=False).to(device); ck=torch.load(a.checkpoint,map_location=device);model.load_state_dict(ck['model_state_dict']);model.backbone.fc=torch.nn.Identity();model.eval()
 out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
 for split_name in ('train','val'):
  part=rows[rows.final_split.eq(split_name)].sort_values(['patient_id','study_id','image_path']).reset_index(drop=True)
  loader=DataLoader(ImageRows(part,a.data_root),batch_size=a.batch_size,shuffle=False,num_workers=0); features=np.empty((len(part),512),dtype=np.float32)
  with torch.inference_mode():
   for images,idx in loader: features[idx.numpy()]=model.backbone(images.to(device)).cpu().numpy()
  groups=list(part.groupby(['patient_id','study_id','label','num_views_in_study'],sort=True)); max_views=max(len(g) for _,g in groups); x=np.zeros((len(groups),max_views,512),np.float32); mask=np.zeros((len(groups),max_views),bool); patient=[];study=[];labels=[];views=[]; lookup=[]
  for i,((pid,sid,label,nviews),g) in enumerate(groups):
   inds=g.index.to_numpy(); x[i,:len(g)]=features[inds];mask[i,:len(g)]=True;patient.append(pid);study.append(sid);labels.append(label);views.append(nviews)
   for pos,(_,r) in enumerate(g.iterrows()): lookup.append({'study_index':i,'patient_id':pid,'study_id':sid,'study_label':label,'num_views':nviews,'image_view_id':pos+1,'image_path':r.image_path})
  if not np.all(mask.sum(1)==np.asarray(views)) or x.shape[0]!=len(set(study)): raise RuntimeError('study/view integrity failure')
  np.savez_compressed(out/f'{split_name}_frozen_embeddings.npz',embeddings=x,mask=mask,labels=np.asarray(labels),patient_ids=np.asarray(patient),study_ids=np.asarray(study),num_views=np.asarray(views))
  pd.DataFrame(lookup).to_csv(out/f'{split_name}_embedding_view_index.csv',index=False)
if __name__=='__main__':main()
