# Segmentation analysis — `train`

## Training
- Epochs trained: **30**
- Final train loss: **8.4746e-02**
- Final val loss: **2.0417e-01**
- Best val loss: **2.0260e-01** at epoch 26
- Final intra-cluster spread: **0.397**
- Final inter-cluster min sep: **3.919**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- Seeding mode: **meanshift**
- `matched_iou_mean`: mean=0.2755  median=0.2684  std=0.09241  min=0.07403  max=0.6211
- `per_source_mse`: mean=2.759e-05  median=1.797e-05  std=3.132e-05  min=3.084e-06  max=0.0001826
- `flux_relative_error`: mean=1.087  median=0.5892  std=1.658  min=0.0003984  max=10.69
- `n_pred`: mean=2.77  median=3  std=1.295  min=1  max=7
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
