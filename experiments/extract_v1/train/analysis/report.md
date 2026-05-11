# Analysis report — `train`

## Training
- Epochs trained: **20**
- Final train loss: **1.6971e-04**
- Final val loss: **1.4265e-04**
- Best val loss: **1.4091e-04** at epoch 18
- Final flux relative error: **4.102e-01**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- `per_slot_mse`: mean=1.896e-06  median=9.933e-07  std=5.972e-06  min=2.56e-07  max=7.194e-05
- `flux_relative_error`: mean=0.3283  median=0.2433  std=0.2553  min=0.03768  max=1.141
- `residual_fraction_of_input`: mean=0.2617  median=0.1986  std=0.1827  min=0.05962  max=0.8888
- `mean_peak_distance_px`: mean=22.62  median=22.31  std=8.407  min=5.743  max=43.64

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
