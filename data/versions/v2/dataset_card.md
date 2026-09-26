# Chest X-ray dataset - v2

Source: Kaggle paultimothymooney/chest-xray-pneumonia (CC BY 4.0).
Image root on the machine that logged it: `E:\DGG_VanierCS\Block2\Data Mining Project\a3_modelversion_dashbord\data\versions\v2\images`
Split: A1 re-split, stratified 85/15 train/val, seed 42; original Kaggle test set.

Derived from **v1** by a simulated scanner drift:
- `contrast_factor` = 0.6
- `brightness_factor` = 1.15
- `blur_rel_radius` = 0.003

Labels and split membership are identical to the parent; only pixels differ.

## Summary

| metric | value |
|---|---|
| n_images | 5856 |
| n_test | 624 |
| pct_pneumonia_test | 62.5 |
| n_train | 4447 |
| pct_pneumonia_train | 74.21 |
| n_val | 785 |
| pct_pneumonia_val | 74.27 |
| mean_intensity_avg | 140.265 |
| std_intensity_avg | 38.857 |
| width_avg | 1327.881 |
| height_avg | 970.689 |
| psi_mean_intensity | 0.7752 |
| psi_std_intensity | 6.8421 |
| psi_mean_intensity_test | 0.841 |
