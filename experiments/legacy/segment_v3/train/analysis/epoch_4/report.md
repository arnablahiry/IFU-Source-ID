# Segmentation analysis — `train`

## Training
- Epochs trained: **4**
- Final train loss: **7.8640e-01**
- Final val loss: **8.1811e-01**
- Best val loss: **8.1811e-01** at epoch 4
- Final intra-cluster spread: **0.693**
- Final inter-cluster min sep: **3.916**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- Seeding mode: **meanshift**
- `matched_iou_mean`: mean=0.348  median=0.3275  std=0.1666  min=0.09291  max=0.8817
- `per_source_mse`: mean=5.446e-05  median=3.003e-05  std=8.409e-05  min=3.112e-06  max=0.0006771
- `flux_relative_error`: mean=1.252  median=0.6409  std=1.851  min=0.002273  max=10.76
- `n_pred`: mean=1.585  median=1  std=0.6729  min=1  max=4
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
