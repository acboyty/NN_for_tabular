# Benchmarks

## results

|                  | cat     | lgb        | xgb     | mlp  | deepfm | xdeepfm | tabnet |
| ---------------- | ------- | ---------- | ------- | ---- | ------ | ------- | ------ |
| Adult            | 0.27532 | 0.27643    | 0.27597 |      |        |         |        |
| Amazon           | 0.13972 | 0.18123(+) | 0.17363 |      |        |         |        |
| Click prediction | 0.39045 | 0.39989    | MEMOF   |      |        |         |        |
| KDD appetency    | 0.07273 | 0.11375(+) | 0.07838 |      |        |         |        |
| KDD churn        | 0.23218 | 0.27123(+) | 0.23955 |      |        |         |        |
| KDD upselling    | 0.16717 | 0.21178(+) | 0.17360 |      |        |         |        |

Metric: Logloss for binary classification(lower is better)