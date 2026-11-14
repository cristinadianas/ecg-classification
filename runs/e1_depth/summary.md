# e1_depth

Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds.

| variant | params | seeds | macro AUC | macro AP | F1@0.5 | F1@tuned | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet1d18_100hz_all12 | 6.3M | 3 | 0.9127 ± 0.0017 | 0.7978 ± 0.0036 | 0.7049 ± 0.0160 | 0.7255 ± 0.0066 | 0.9356 ± 0.0021 | 0.9049 ± 0.0028 | 0.9283 ± 0.0002 | 0.8995 ± 0.0054 | 0.8951 ± 0.0058 |
| resnet1d34_100hz_all12 | 11.9M | 3 | 0.9131 ± 0.0017 | 0.7973 ± 0.0040 | 0.6926 ± 0.0098 | 0.7239 ± 0.0053 | 0.9368 ± 0.0007 | 0.9044 ± 0.0037 | 0.9264 ± 0.0025 | 0.9036 ± 0.0037 | 0.8942 ± 0.0071 |
