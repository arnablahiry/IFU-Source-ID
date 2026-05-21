# Analysis report — `train`

## Training
- Epochs trained: **127**
- Final train loss: **2.1652e-03**
- Final val loss: **1.8889e-02**
- Best val loss: **6.1831e-03** at epoch 27
- Final flux relative error: **2.897e-01**

![training curves](loss_curves.png)

## Validation evaluation
- Cubes evaluated: 1000
- `per_slot_mse`: mean=0.002197  median=0.002011  std=0.0008288  min=0.0009045  max=0.004388
- `flux_relative_error`: mean=17.55  median=16.03  std=9.344  min=3.9  max=56.6
- `residual_fraction_of_input`: mean=117.3  median=109.6  std=56.25  min=15.29  max=340.3
- `mean_peak_distance_px`: mean=21.14  median=20.36  std=8.711  min=2.266  max=48.47

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
