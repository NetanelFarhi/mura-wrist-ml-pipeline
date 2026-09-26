"""Export real-view probabilities from the frozen Phase 1 Model B checkpoint.

Only development splits are permitted. This script intentionally has no test
mode so the opened Phase 1 test set cannot be used by Phase 2A.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from src.data.transforms import build_eval_transform
from src.models.resnet18_classifier import build_resnet18_classifier
from src.evaluation.phase2a import validate_per_view_export

class Views(Dataset):
    def __init__(self, frame: pd.DataFrame, data_root: Path):
        self.frame=frame.reset_index(drop=True); self.data_root=data_root; self.transform=build_eval_transform(image_size=224)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index):
        row=self.frame.iloc[index]
        with Image.open(self.data_root / row.image_path) as im: image=self.transform(im.copy())
        return image, index

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--data-root',required=True); pa.add_argument('--manifest',default='data/manifests/wrist_manifest.csv'); pa.add_argument('--split-manifest',default='data/manifests/split_manifest.csv'); pa.add_argument('--checkpoint',default='models/resnet18/best_resnet18_study_level.pt'); pa.add_argument('--output-dir',default='models/resnet18'); pa.add_argument('--batch-size',type=int,default=32); pa.add_argument('--device',default='cpu'); args=pa.parse_args()
    device=torch.device(args.device)
    manifest=pd.read_csv(args.manifest); splits=pd.read_csv(args.split_manifest)
    df=manifest.merge(splits,on=['patient_id','study_id'],validate='many_to_one')
    if not set(df.final_split.unique()).issubset({'train','val','test'}): raise ValueError('Unexpected final split')
    model=build_resnet18_classifier(weights='imagenet',allow_weight_download=False).to(device)
    checkpoint=torch.load(args.checkpoint,map_location=device); model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    root=Path(args.data_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    for split,name in [('train','per_view_predictions_train.csv'),('val','per_view_predictions_validation.csv')]:
        part=df.loc[df.final_split.eq(split)].copy().sort_values(['patient_id','study_id','image_path']).reset_index(drop=True)
        loader=DataLoader(Views(part,root),batch_size=args.batch_size,shuffle=False,num_workers=0)
        probs=torch.empty(len(part),dtype=torch.float32)
        with torch.inference_mode():
            for images,indices in loader:
                probs[indices]=torch.sigmoid(model(images.to(device))).cpu()
        part['image_view_id']=part.groupby('study_id').cumcount()+1
        part['view_probability']=probs.numpy()
        part=part.rename(columns={'label':'study_label','num_views_in_study':'num_views'})
        result=part[['patient_id','study_id','image_path','image_view_id','study_label','view_probability','num_views']]
        # Defensive identity/mask invariants: no padding is ever materialized here.
        validate_per_view_export(result)
        result.to_csv(out/name,index=False)
        summary=result.groupby(['patient_id','study_id','study_label','num_views'],as_index=False).view_probability.agg(['mean','max','min','std']).reset_index()
        summary['std']=summary['std'].fillna(0.0); summary['range_view_probability']=summary['max']-summary['min']
        summary.rename(columns={'mean':'mean_view_probability','max':'max_view_probability','min':'min_view_probability','std':'std_view_probability'}).to_csv(out/f'{split}_view_probability_summary.csv',index=False)
if __name__=='__main__': main()
