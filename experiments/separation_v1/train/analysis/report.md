# Analysis report — `train`

## Training
- Epochs trained: **122**
- Final train loss: **9.7222e-06**
- Final val loss: **1.3744e-05**
- Best val loss: **1.3226e-05** at epoch 102
- Final flux relative error: **2.708e-02**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 1000
- `per_slot_mse`: mean=2.167e-07  median=1.475e-07  std=2.197e-07  min=2.016e-08  max=1.982e-06
- `flux_relative_error`: mean=0.03066  median=0.0253  std=0.02153  min=0.0007466  max=0.1514
- `residual_fraction_of_input`: mean=5.128e-15  median=5.039e-15  std=1.325e-15  min=2.786e-15  max=8.461e-15
- `mean_peak_distance_px`: mean=21.79  median=21.48  std=8.56  min=2.828  max=47.47

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
