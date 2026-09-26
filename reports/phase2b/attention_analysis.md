# Attention analysis

Attention is a pooling weight, not lesion localization. Padded views are excluded and have zero saved weight.

Median attention entropy: 1.073965.

| group   |   n |   attention_auroc |   mean_auroc |
|:--------|----:|------------------:|-------------:|
| low     | 231 |          0.971845 |     0.971929 |
| medium  | 230 |          0.908758 |     0.909555 |
| high    | 231 |          0.767991 |     0.774206 |