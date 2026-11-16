# b0_baseline

Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds.

| variant | params | seeds | macro AUC | macro AP | F1@0.5 | F1@tuned | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fcn1d_100hz_all12 | 0.3M | 3 | 0.9192 ± 0.0013 | 0.8065 ± 0.0012 | 0.7231 ± 0.0055 | 0.7368 ± 0.0035 | 0.9395 ± 0.0022 | 0.9232 ± 0.0028 | 0.9322 ± 0.0019 | 0.9063 ± 0.0049 | 0.8947 ± 0.0046 |
