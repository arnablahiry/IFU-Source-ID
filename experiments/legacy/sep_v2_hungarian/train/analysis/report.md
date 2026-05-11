# Analysis report — `train`

## Training
- Epochs trained: **30**
- Final train loss: **3.4168e-05**
- Final val loss: **3.4365e-05**
- Best val loss: **3.4365e-05** at epoch 30
- Final flux relative error: **9.087e-01**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- `per_slot_mse`: mean=5.928e-07  median=4.659e-07  std=3.979e-07  min=1.696e-07  max=2.523e-06
- `flux_relative_error`: mean=1.035  median=0.8188  std=0.7221  min=0.09928  max=4.02
- `residual_fraction_of_input`: mean=0.238  median=0.1769  std=0.1778  min=0.03999  max=0.8349
- `mean_peak_distance_px`: mean=22.99  median=23.09  std=8.163  min=6.218  max=42.63

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
