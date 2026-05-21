# Analysis report — `train`

## Training
- Epochs trained: **479**
- Final train loss: **1.4879e-04**
- Final val loss: **1.4579e-04**
- Best val loss: **1.3686e-04** at epoch 379
- Final flux relative error: **1.240e-01**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 1000
- `per_slot_mse`: mean=1.726e-06  median=1.358e-06  std=1.206e-06  min=3.432e-07  max=1.262e-05
- `flux_relative_error`: mean=0.1096  median=0.1036  std=0.05401  min=0.009748  max=0.449
- `residual_fraction_of_input`: mean=1.16e-15  median=1.094e-15  std=8.771e-16  min=1.379e-19  max=4.344e-15
- `mean_peak_distance_px`: mean=16.8  median=17.81  std=5.322  min=1.414  max=29.21

![eval distributions](eval_distributions.png)

## Qualitative samples
![sample_cube_1422](sample_cube_1422.png)

![sample_cube_31](sample_cube_31.png)

![sample_cube_4021](sample_cube_4021.png)

![sample_cube_4184](sample_cube_4184.png)

![sample_cube_5163](sample_cube_5163.png)

![sample_cube_6546](sample_cube_6546.png)

![sample_cube_7240](sample_cube_7240.png)

![sample_cube_7266](sample_cube_7266.png)
