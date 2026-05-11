# Segmentation analysis — `train`

## Training
- Epochs trained: **2**
- Final train loss: **9.1619e-01**
- Final val loss: **9.1907e-01**
- Best val loss: **9.1907e-01** at epoch 2
- Final intra-cluster spread: **0.847**
- Final inter-cluster min sep: **4.634**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- Seeding mode: **meanshift**
- `matched_iou_mean`: mean=0.3234  median=0.2903  std=0.171  min=0.0916  max=0.8709
- `per_source_mse`: mean=3.857e-05  median=2.317e-05  std=5.007e-05  min=3.278e-06  max=0.0003382
- `flux_relative_error`: mean=1.22  median=0.6157  std=1.834  min=0.0009511  max=10.72
- `n_pred`: mean=2.145  median=2  std=0.956  min=1  max=5
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
