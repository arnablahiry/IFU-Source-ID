# Segmentation analysis — `train`

## Training
- Epochs trained: **1**
- Final train loss: **1.1660e+00**
- Final val loss: **9.8971e-01**
- Best val loss: **9.8971e-01** at epoch 1
- Final intra-cluster spread: **0.819**
- Final inter-cluster min sep: **4.206**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- Seeding mode: **meanshift**
- `matched_iou_mean`: mean=0.3234  median=0.3045  std=0.1623  min=0.09264  max=0.8618
- `per_source_mse`: mean=4.716e-05  median=2.66e-05  std=6.802e-05  min=3.876e-06  max=0.0005321
- `flux_relative_error`: mean=1.25  median=0.6369  std=1.848  min=0.004634  max=10.72
- `n_pred`: mean=1.925  median=2  std=0.854  min=1  max=5
- `n_gt`: mean=3.425  median=3  std=1.12  min=2  max=5

![eval distributions](eval_distributions.png)

## Qualitative samples
![sample_cube_1140](sample_cube_1140.png)

![sample_cube_1145](sample_cube_1145.png)

![sample_cube_1174](sample_cube_1174.png)

![sample_cube_1284](sample_cube_1284.png)

![sample_cube_1483](sample_cube_1483.png)

![sample_cube_283](sample_cube_283.png)

![sample_cube_294](sample_cube_294.png)

![sample_cube_580](sample_cube_580.png)
