# b0_baseline: performance by age

Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds; CI = 95% bootstrap interval of the macro AUROC (mean of the per-seed bounds). Per-class AUROC on fewer than 10 positives is shown as -. All metrics: higher is better.

## Test-set composition by age

| group | n | % of test | % NORM | % MI | % STTC | % CD | % HYP |
|---|---|---|---|---|---|---|---|
| all | 2163 | 100.0 | 44.6 | 25.6 | 24.2 | 23.0 | 12.2 |
| <40 | 284 | 13.1 | 83.1 | 2.5 | 7.0 | 13.0 | 3.5 |
| 40-59 | 642 | 29.7 | 60.4 | 17.9 | 15.6 | 16.4 | 6.9 |
| 60-79 | 916 | 42.3 | 34.8 | 30.6 | 28.4 | 24.1 | 16.3 |
| 80+ | 306 | 14.1 | 6.9 | 46.7 | 42.8 | 42.2 | 18.3 |
| missing | 15 | 0.7 | 0.0 | 53.3 | 80.0 | 40.0 | 26.7 |

## fcn1d_100hz_all12: test AUROC by age

| group | n | macro AUROC | 95% CI | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|
| all | 2163 | 0.9192 ± 0.0013 | [0.912, 0.926] | 0.940 | 0.923 | 0.932 | 0.906 | 0.895 |
| <40 | 284 | 0.9225 ± 0.0054 | [0.890, 0.949] | 0.853 | - | 0.955 | 0.861 | 0.958 |
| 40-59 | 642 | 0.9159 ± 0.0019 | [0.899, 0.931] | 0.914 | 0.939 | 0.942 | 0.893 | 0.892 |
| 60-79 | 916 | 0.9099 ± 0.0011 | [0.899, 0.921] | 0.932 | 0.895 | 0.916 | 0.907 | 0.899 |
| 80+ | 306 | 0.8734 ± 0.0010 | [0.848, 0.895] | 0.898 | 0.873 | 0.889 | 0.889 | 0.818 |
