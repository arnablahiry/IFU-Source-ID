# Analysis report — `train`

## Training
- Epochs trained: **30**
- Final train loss: **2.0111e-05**
- Final val loss: **2.1310e-05**
- Best val loss: **2.1310e-05** at epoch 30
- Final flux relative error: **4.452e-02**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 200
- `per_slot_mse`: mean=3.686e-07  median=2.464e-07  std=3.243e-07  min=5.845e-08  max=1.687e-06
- `flux_relative_error`: mean=0.04894  median=0.0381  std=0.03904  min=0.0026  max=0.2769
- `residual_fraction_of_input`: mean=4.74e-15  median=4.76e-15  std=4.049e-16  min=3.864e-15  max=5.737e-15
- `mean_peak_distance_px`: mean=21.65  median=20.96  std=7.746  min=6.44  max=39.95

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
