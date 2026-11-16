# b0_baseline: performance by sex

Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds; CI = 95% bootstrap interval of the macro AUROC (mean of the per-seed bounds). Per-class AUROC on fewer than 10 positives is shown as -. All metrics: higher is better.

## Test-set composition by sex

| group | n | % of test | % NORM | % MI | % STTC | % CD | % HYP |
|---|---|---|---|---|---|---|---|
| all | 2163 | 100.0 | 44.6 | 25.6 | 24.2 | 23.0 | 12.2 |
| male | 1115 | 51.5 | 46.4 | 26.9 | 20.7 | 24.0 | 12.2 |
| female | 1048 | 48.5 | 42.7 | 24.1 | 27.9 | 21.9 | 12.1 |

## fcn1d_100hz_all12: test AUROC by sex

| group | n | macro AUROC | 95% CI | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|
| all | 2163 | 0.9192 ± 0.0013 | [0.912, 0.926] | 0.940 | 0.923 | 0.932 | 0.906 | 0.895 |
| male | 1115 | 0.9305 ± 0.0008 | [0.921, 0.939] | 0.945 | 0.938 | 0.944 | 0.919 | 0.906 |
| female | 1048 | 0.9071 ± 0.0029 | [0.895, 0.919] | 0.934 | 0.905 | 0.919 | 0.894 | 0.884 |
