# Analysis report — `train`

## Training
- Epochs trained: **30**
- Final train loss: **1.7574e-05**
- Final val loss: **1.8738e-05**
- Best val loss: **1.8738e-05** at epoch 30
- Final flux relative error: **5.779e-02**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 1000
- `per_slot_mse`: mean=2.897e-07  median=2.164e-07  std=2.479e-07  min=3.574e-08  max=2.474e-06
- `flux_relative_error`: mean=0.04972  median=0.04048  std=0.03544  min=8.004e-05  max=0.2419
- `residual_fraction_of_input`: mean=6.421e-15  median=6.459e-15  std=8.169e-16  min=4.301e-15  max=8.76e-15
- `mean_peak_distance_px`: mean=21.85  median=21.37  std=8.374  min=2.291  max=46.23

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
