"""Run Phase 2A calibration/operating-policy analysis from validation predictions only."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from src.evaluation.phase2a import (bootstrap_policy_ci, calibration_metrics, cross_fitted_temperature,
    load_validation_predictions, operating_metrics, select_sensitivity_constrained_threshold)

def write_reliability(path: Path, y, before, after):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    for p, label, color in [(before, 'Uncalibrated', '#1f77b4'), (after, 'Cross-fitted temperature scaled', '#d62728')]:
        frac, mean = calibration_curve(y, p, n_bins=10, strategy='uniform')
        ax.plot(mean, frac, 'o-', label=label, color=color)
    ax.plot([0,1],[0,1],'--',color='0.4',label='Perfect calibration')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted abnormal probability',ylabel='Observed abnormal fraction',title='Model B validation reliability')
    ax.legend(frameon=False); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path); plt.close(fig)

def main():
    pa=argparse.ArgumentParser(); pa.add_argument('--validation-predictions', required=True); pa.add_argument('--output-dir', default='reports/phase2a'); args=pa.parse_args()
    output=Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    df=load_validation_predictions(args.validation_predictions); y=df.label.to_numpy(); p=df.abnormal_probability.to_numpy()
    scaled, temps=cross_fitted_temperature(y,p)
    before=calibration_metrics(y,p); after=calibration_metrics(y,scaled)
    write_reliability(output/'calibration_before.png',y,p,scaled)
    policies=[]
    original=operating_metrics(y,p,0.3860229253768921); original['policy']='P1 historical Youden-J'; policies.append(original)
    for target,name in [(0.85,'P2 sensitivity >= 0.85'),(0.90,'P3 sensitivity >= 0.90')]:
        row=select_sensitivity_constrained_threshold(y,p,target); row['policy']=name; row['sensitivity_target']=target; row.update(bootstrap_policy_ci(y,p,float(row['threshold']))); policies.append(row)
    pd.DataFrame(policies).to_csv(output/'operating_points.csv',index=False)
    # Error policy transitions, validation only.
    flags={str(r['policy']): p >= float(r['threshold']) for r in policies}
    rows=[]
    for name, flag in flags.items():
        rows.append({'policy':name,'false_negatives':int(((y==1)&~flag).sum()),'false_positives':int(((y==0)&flag).sum()),'true_positives':int(((y==1)&flag).sum()),'true_negatives':int(((y==0)&~flag).sum())})
    pd.DataFrame(rows).to_csv(output/'validation_error_transitions.csv',index=False)
    # Workload tradeoff figure.
    fig,ax=plt.subplots(figsize=(6.4,4.8),dpi=180)
    for r in policies:
        ax.scatter(r['specificity'],r['sensitivity'],s=float(r['flagged_abnormal_percent'])*8,label=f"{r['policy']} ({r['flagged_abnormal_percent']:.1f}% flagged)")
    ax.set(xlabel='Specificity',ylabel='Sensitivity',xlim=(0.5,1),ylim=(0.65,1),title='Validation policy workload tradeoff'); ax.grid(alpha=.25); ax.legend(fontsize=7,frameon=False); fig.tight_layout(); fig.savefig(output/'workload_tradeoff.png'); plt.close(fig)
    (output/'calibration_before.md').write_text('# Calibration audit (validation only)\n\n'
        f'Uncalibrated: Brier={before["brier"]:.6f}; NLL={before["nll"]:.6f}; ECE (10 bins)={before["ece_10_bins"]:.6f}; AUROC={before["auroc"]:.6f}; AUPRC={before["auprc"]:.6f}.\n\n'
        f'Cross-fitted temperature scaling (5 folds; temperatures: {", ".join(f"{t:.4f}" for t in temps)}): Brier={after["brier"]:.6f}; NLL={after["nll"]:.6f}; ECE={after["ece_10_bins"]:.6f}; AUROC={after["auroc"]:.6f}; AUPRC={after["auprc"]:.6f}.\n\n'
        'The scaled metrics are cross-fitted: each study was calibrated by a temperature fitted without that study. Temperature scaling is monotonic, so any unchanged ranking metric is expected and is not presented as a ranking improvement.\n',encoding='utf-8')
    table = pd.DataFrame(policies)
    cols=['policy','threshold','sensitivity','specificity','precision','f1','false_positive_count','false_negative_count','flagged_abnormal_percent','sensitivity_ci_lower','sensitivity_ci_upper','specificity_ci_lower','specificity_ci_upper']
    lines=['| '+' | '.join(cols)+' |','|'+ '|'.join(['---']*len(cols))+'|']
    for _, row in table.iterrows():
        lines.append('| '+' | '.join(str(row.get(col,'')) for col in cols)+' |')
    (output/'operating_point_analysis.md').write_text('# Validation-only operating policy analysis\n\n'+'\n'.join(lines)+'\n\nBootstrap intervals are stratified study-level 95% percentile intervals (2,000 replicates, seed 23).\n',encoding='utf-8')
    (output/'phase2a_metrics.json').write_text(json.dumps({'uncalibrated':before,'temperature_cross_fitted':after,'fold_temperatures':temps,'policies':policies},indent=2),encoding='utf-8')
if __name__=='__main__': main()
