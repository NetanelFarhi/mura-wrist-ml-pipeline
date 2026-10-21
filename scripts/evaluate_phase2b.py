"""Validation-only aggregate reports for bounded Phase 2B models.

Per-study joins are held in memory only. This script writes no patient/study
identifiers or per-study probabilities to the repository report directory.
"""
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
out=Path('reports/phase2b');out.mkdir(parents=True,exist_ok=True);root=Path('models/phase2b')
metrics=pd.read_csv(root/'model_metrics.csv');phase1=pd.read_csv('models/resnet18/validation_predictions.csv').rename(columns={'abnormal_probability':'phase1_mean_probability'})
preds={m:pd.read_csv(root/f'{m}_validation_predictions.csv').rename(columns={'study_probability':f'{m}_embedding_probability'}) for m in ['mean','max','attention']}
d=phase1
for x in preds.values(): d=d.merge(x[['patient_id','study_id',x.columns[-1]]],on=['patient_id','study_id'])
w=pd.read_csv(root/'attention_weights_validation.csv')
ent=w.groupby('study_id').attention_weight.apply(lambda x:float(-(x*np.log(np.maximum(x,1e-12))).sum())).rename('attention_entropy')
d=d.merge(ent,on='study_id');d=d.merge(pd.read_csv('models/resnet18/val_view_probability_summary.csv')[['study_id','range_view_probability']],on='study_id')
for m in ['mean','max','attention']: d[f'{m}_correct_0_5']=(d[f'{m}_embedding_probability']>=.5)==d.label
summary=[]
for m in ['mean','max','attention']:
    correct=d[f'{m}_correct_0_5']
    summary.append({
        'method':m,
        'studies':int(len(d)),
        'correct':int(correct.sum()),
        'incorrect':int((~correct).sum()),
        'false_negative':int(((d.label==1)&(~correct)).sum()),
        'false_positive':int(((d.label==0)&(~correct)).sum()),
    })
pd.DataFrame(summary).to_csv(out/'paired_error_summary.csv',index=False)
att=d.attention_embedding_probability;low,high=d.range_view_probability.quantile([1/3,2/3]);rows=[]
for name,mask in [('low',d.range_view_probability<=low),('medium',(d.range_view_probability>low)&(d.range_view_probability<=high)),('high',d.range_view_probability>high)]: rows.append({'group':name,'n':int(mask.sum()),'attention_auroc':roc_auc_score(d.label[mask],att[mask]),'mean_auroc':roc_auc_score(d.label[mask],d.mean_embedding_probability[mask])})
sub=pd.DataFrame(rows);sub.to_csv(out/'disagreement_subgroups.csv',index=False)
(out/'attention_analysis.md').write_text('# Attention analysis\n\nAttention is a pooling weight, not lesion localization. Padded views are excluded and have zero saved weight.\n\nMedian attention entropy: %.6f.\n\n%s' % (d.attention_entropy.median(),sub.to_markdown(index=False)))
(out/'disagreement_analysis.md').write_text(sub.to_markdown(index=False))
metrics.to_csv(out/'model_comparison.csv',index=False)
(out/'complexity.md').write_text(metrics[['model','total_parameters','trainable_parameters','training_seconds']].to_markdown(index=False)+'\n\nEncoder is frozen/shared; counts are pooling/head only. CPU training times exclude one-time embedding extraction.')
(out/'error_analysis.md').write_text(
    'Aggregate paired validation summary: `paired_error_summary.csv`. '
    'Attention errors at 0.5: %d; mean errors: %d. Individual study IDs and '
    'per-study probabilities are local-only. No medical explanations inferred.'
    % (int((~d.attention_correct_0_5).sum()),int((~d.mean_correct_0_5).sum()))
)
