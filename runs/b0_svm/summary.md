# b0_svm

Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds.

| variant | params | seeds | macro AUC | macro AP | F1@0.5 | F1@tuned | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| svm-raw_100hz_all12 | – | 1 | 0.7488 | 0.4874 | 0.2089 | 0.4994 | 0.7983 | 0.7102 | 0.7164 | 0.7150 | 0.8040 |
| svm-statistical_100hz_all12 | – | 1 | 0.8884 | 0.7419 | 0.6845 | 0.6816 | 0.9117 | 0.8561 | 0.9044 | 0.8703 | 0.8995 |
| svm-wavelet_100hz_all12 | – | 1 | 0.9031 | 0.7696 | 0.7068 | 0.7171 | 0.9232 | 0.8702 | 0.9264 | 0.8931 | 0.9027 |
