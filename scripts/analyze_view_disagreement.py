"""Descriptive validation-only view-disagreement analysis for Phase 2A."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--view-summary',default='models/resnet18/val_view_probability_summary.csv'); p.add_argument('--validation-predictions',default='models/resnet18/validation_predictions.csv'); p.add_argument('--operating-points',default='reports/phase2a/operating_points.csv'); p.add_argument('--output',default='reports/phase2a/view_disagreement.md'); a=p.parse_args()
    if 'test' in str(a.view_summary).lower() or 'test' in str(a.validation_predictions).lower(): raise ValueError('Phase 2A disagreement analysis forbids test artifacts.')
    summary=pd.read_csv(a.view_summary); pred=pd.read_csv(a.validation_predictions); policies=pd.read_csv(a.operating_points)
    pred=pred.rename(columns={'label':'study_label'})
    df=pred.merge(summary,on=['patient_id','study_id','study_label','num_views'],validate='one_to_one')
    lines=['# Validation-only view-disagreement exploration','','`std_view_probability` and `range_view_probability` describe dispersion of Model B probabilities across real radiographic views in a study. They are not uncertainty estimates and do not establish causation.','','| Policy | Group | n | Median SD | Median range |','|---|---:|---:|---:|---:|']
    change=[]
    flags={}
    for _, policy in policies.iterrows():
        name=policy.policy; flag=df.abnormal_probability >= policy.threshold; flags[name]=flag
        correct=flag.eq(df.study_label.astype(bool))
        for group,mask in [('correct',correct),('incorrect',~correct),('false_negative',(df.study_label.eq(1)&~flag)),('true_positive',(df.study_label.eq(1)&flag))]:
            sub=df[mask]; lines.append(f'| {name} | {group} | {len(sub)} | {sub.std_view_probability.median():.6f} | {sub.range_view_probability.median():.6f} |')
    names=list(flags)
    lines+=['','## Error transitions between policies','','| From → to | FNs resolved | New FPs introduced |','|---|---:|---:|']
    for left,right in zip(names,names[1:]):
        fn_resolved=int(((df.study_label.eq(1)) & ~flags[left] & flags[right]).sum())
        new_fp=int(((df.study_label.eq(0)) & ~flags[left] & flags[right]).sum())
        lines.append(f'| {left} → {right} | {fn_resolved} | {new_fp} |')
    lines+=['','The same frozen image model and mean-pooling scores are used in every row. The analysis is restricted to the development validation set; it neither trains learned pooling nor uses the already-opened test set.']
    Path(a.output).write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__': main()
