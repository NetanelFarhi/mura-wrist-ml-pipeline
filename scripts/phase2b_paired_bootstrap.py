import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from src.evaluation.phase2a import select_sensitivity_constrained_threshold
rng=np.random.default_rng(23); root='models/phase2b/'; names=['mean','max','attention']; dfs={n:pd.read_csv(root+n+'_validation_predictions.csv') for n in names}; base=dfs['mean']; patients=base.patient_id.unique(); rows=[]
for a,b in [('mean','max'),('mean','attention'),('max','attention')]:
 vals=[]
 for _ in range(2000):
  sampled=rng.choice(patients,len(patients),True); ix=np.concatenate([np.where(base.patient_id.values==p)[0] for p in sampled]); y=base.label.values[ix]; pa=dfs[a].study_probability.values[ix]; pb=dfs[b].study_probability.values[ix]
  row=[roc_auc_score(pb,y) if False else roc_auc_score(y,pb)-roc_auc_score(y,pa),average_precision_score(y,pb)-average_precision_score(y,pa)]
  for target in (.85,.90): row.append(select_sensitivity_constrained_threshold(y,pb,target)['specificity']-select_sensitivity_constrained_threshold(y,pa,target)['specificity'])
  vals.append(row)
 vals=np.array(vals)
 for i,m in enumerate(['AUROC','AUPRC','Specificity@85','Specificity@90']):rows.append({'comparison':f'{b} - {a}','metric':m,'difference':float(vals[:,i].mean()),'ci_lower':float(np.quantile(vals[:,i],.025)),'ci_upper':float(np.quantile(vals[:,i],.975)),'n_bootstrap':2000,'seed':23})
out=pd.DataFrame(rows);out.to_csv('reports/phase2b/paired_bootstrap.csv',index=False);open('reports/phase2b/paired_comparison.md','w').write(out.to_markdown(index=False))
