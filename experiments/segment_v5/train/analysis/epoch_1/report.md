# Segmentation analysis — `train`

## Training
- Epochs trained: **2**
- Final train loss: **4.2369e-01**
- Final val loss: **3.7272e-01**
- Best val loss: **3.7272e-01** at epoch 2
- Final intra-cluster spread: **0.646**
- Final inter-cluster min sep: **4.256**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- Seeding mode: **meanshift**
- `matched_iou_mean`: mean=0.2316  median=0.2197  std=0.08613  min=0.06985  max=0.6631
- `per_source_mse`: mean=2.621e-05  median=1.639e-05  std=3.11e-05  min=4.965e-06  max=0.0002051
- `flux_relative_error`: mean=1.073  median=0.5669  std=1.344  min=0.1099  max=8.119
- `n_pred`: mean=3.69  median=3  std=1.518  min=1  max=8
- `n_gt`: mean=3.445  median=3  std=1.103  min=2  max=5

![eval distributions](eval_distributions.png)

## Qualitative samples
![sample_cube_11391](sample_cube_11391.png)

![sample_cube_14780](sample_cube_14780.png)

![sample_cube_1611](sample_cube_1611.png)

![sample_cube_3378](sample_cube_3378.png)

![sample_cube_8028](sample_cube_8028.png)

![sample_cube_8363](sample_cube_8363.png)

![sample_cube_9215](sample_cube_9215.png)

![sample_cube_9912](sample_cube_9912.png)
