# Validation-only operating policy analysis

| policy | threshold | sensitivity | specificity | precision | f1 | false_positive_count | false_negative_count | flagged_abnormal_percent | sensitivity_ci_lower | sensitivity_ci_upper | specificity_ci_lower | specificity_ci_upper |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 historical Youden-J | 0.3860229253768921 | 0.8 | 0.8782201405152225 | 0.803030303030303 | 0.8015122873345936 | 52 | 53 | 38.15028901734104 | nan | nan | nan | nan |
| P2 sensitivity >= 0.85 | 0.2803132832050323 | 0.8528301886792453 | 0.7985948477751756 | 0.7243589743589743 | 0.783362218370884 | 86 | 39 | 45.08670520231214 | 0.8075471698113208 | 0.8943396226415095 | 0.7587822014051522 | 0.8360655737704918 |
| P3 sensitivity >= 0.90 | 0.1609840840101242 | 0.9018867924528302 | 0.7142857142857143 | 0.6620498614958449 | 0.7635782747603834 | 122 | 26 | 52.16763005780347 | 0.8641509433962264 | 0.939622641509434 | 0.6697892271662763 | 0.7564402810304449 |

Bootstrap intervals are stratified study-level 95% percentile intervals (2,000 replicates, seed 23).
