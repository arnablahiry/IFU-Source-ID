# Analysis report — `train`

## Training
- Epochs trained: **91**
- Final train loss: **1.2445e-02**
- Final val loss: **1.2962e-01**
- Best val loss: **4.7106e-02** at epoch 20
- Final flux relative error: **2.628e-01**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 1000
- `per_slot_mse`: mean=9.652e-06  median=5.939e-06  std=1.115e-05  min=1.469e-06  max=0.0001466
- `flux_relative_error`: mean=0.4096  median=0.2465  std=0.4459  min=0.01423  max=3.676
- `residual_fraction_of_input`: mean=0.2145  median=0.1826  std=0.1243  min=0.04599  max=0.735
- `mean_peak_distance_px`: mean=20.96  median=20.12  std=8.373  min=2.791  max=47.23

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
