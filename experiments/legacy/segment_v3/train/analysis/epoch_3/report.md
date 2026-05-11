# Segmentation analysis — `train`

## Training
- Epochs trained: **3**
- Final train loss: **8.4187e-01**
- Final val loss: **1.0067e+00**
- Best val loss: **9.1907e-01** at epoch 2
- Final intra-cluster spread: **0.745**
- Final inter-cluster min sep: **3.911**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- Seeding mode: **meanshift**
- `matched_iou_mean`: mean=0.3257  median=0.311  std=0.1477  min=0.09681  max=0.8592
- `per_source_mse`: mean=4.439e-05  median=2.553e-05  std=7.01e-05  min=3.976e-06  max=0.0005321
- `flux_relative_error`: mean=1.235  median=0.6409  std=1.843  min=0.007619  max=10.76
- `n_pred`: mean=2.07  median=2  std=0.8456  min=1  max=5
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
