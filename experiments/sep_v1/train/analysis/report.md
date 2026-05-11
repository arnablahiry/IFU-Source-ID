# Analysis report — `train`

## Training
- Epochs trained: **30**
- Final train loss: **6.1528e-07**
- Final val loss: **6.3162e-07**
- Best val loss: **6.3162e-07** at epoch 30
- Final flux relative error: **9.478e-01**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- `per_slot_mse`: mean=6.619e-07  median=5.112e-07  std=4.695e-07  min=1.841e-07  max=3.165e-06
- `flux_relative_error`: mean=1.073  median=0.8728  std=0.7484  min=0.115  max=3.968
- `residual_fraction_of_input`: mean=0.2377  median=0.1716  std=0.18  min=0.03764  max=0.8532
- `mean_peak_distance_px`: mean=23.46  median=23.78  std=8.124  min=6.137  max=42.77

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
