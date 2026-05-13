# Analysis report — `train`

## Training
- Epochs trained: **140**
- Final train loss: **2.8742e-04**
- Final val loss: **8.6335e-04**
- Best val loss: **5.5310e-04** at epoch 50
- Final flux relative error: **1.877e-01**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 1000
- `per_slot_mse`: mean=1.386e-06  median=1.15e-06  std=7.928e-07  min=3.615e-07  max=6.488e-06
- `flux_relative_error`: mean=0.2005  median=0.1826  std=0.08839  min=0.04748  max=0.85
- `residual_fraction_of_input`: mean=4.074e-15  median=4.079e-15  std=1.15e-15  min=1.984e-15  max=6.832e-15
- `mean_peak_distance_px`: mean=16.39  median=16.03  std=8.545  min=1  max=47.98

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
