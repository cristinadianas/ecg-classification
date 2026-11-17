# e1_depth: performance by sex

Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds; CI = 95% bootstrap interval of the macro AUROC (mean of the per-seed bounds). Per-class AUROC on fewer than 10 positives is shown as -. All metrics: higher is better.

## Test-set composition by sex

| group | n | % of test | % NORM | % MI | % STTC | % CD | % HYP |
|---|---|---|---|---|---|---|---|
| all | 2163 | 100.0 | 44.6 | 25.6 | 24.2 | 23.0 | 12.2 |
| male | 1115 | 51.5 | 46.4 | 26.9 | 20.7 | 24.0 | 12.2 |
| female | 1048 | 48.5 | 42.7 | 24.1 | 27.9 | 21.9 | 12.1 |

## resnet1d18_100hz_all12: test AUROC by sex

| group | n | macro AUROC | 95% CI | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|
| all | 2163 | 0.9127 ± 0.0017 | [0.905, 0.920] | 0.936 | 0.905 | 0.928 | 0.899 | 0.895 |
| male | 1115 | 0.9176 ± 0.0015 | [0.907, 0.927] | 0.938 | 0.917 | 0.939 | 0.903 | 0.890 |
| female | 1048 | 0.9068 ± 0.0023 | [0.895, 0.918] | 0.933 | 0.891 | 0.915 | 0.895 | 0.900 |

## resnet1d34_100hz_all12: test AUROC by sex

| group | n | macro AUROC | 95% CI | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|
| all | 2163 | 0.9131 ± 0.0017 | [0.905, 0.920] | 0.937 | 0.904 | 0.926 | 0.904 | 0.894 |
| male | 1115 | 0.9198 ± 0.0006 | [0.910, 0.929] | 0.942 | 0.917 | 0.940 | 0.905 | 0.895 |
| female | 1048 | 0.9057 ± 0.0028 | [0.894, 0.917] | 0.931 | 0.890 | 0.911 | 0.903 | 0.893 |
