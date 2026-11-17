# b0_benchmark

Test set = fold 10, checkpoint = best validation macro AUC. Mean ± std over seeds.

| variant | params | seeds | macro AUC | macro AP | F1@0.5 | F1@tuned | AUC NORM | AUC MI | AUC STTC | AUC CD | AUC HYP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet1d_wang_100hz_all12_aug-crop_std-global_benchmark | 0.5M | 3 | 0.9324 ± 0.0011 | 0.8369 ± 0.0013 | 0.7557 ± 0.0034 | 0.7577 ± 0.0049 | 0.9490 ± 0.0007 | 0.9395 ± 0.0021 | 0.9346 ± 0.0002 | 0.9295 ± 0.0004 | 0.9093 ± 0.0039 |
