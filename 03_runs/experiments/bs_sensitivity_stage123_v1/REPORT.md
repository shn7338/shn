# Three-stage model BS-sensitivity report

## Frozen baseline

| Stage | Test RMSE (dB) | Test MAE (dB) |
|---|---:|---:|
| Stage1 DPM | 2.613 | 1.383 |
| Stage2-A isotropic | 5.512 | 2.701 |
| Stage2-B directional (100 points) | 5.765 | 3.062 |

## Reproduced pipeline sensitivity

| Condition | Error | Stage1 RMSE | Stage2-A RMSE | Final RMSE | Final unmeasured RMSE | masked-SSIM | Delta vs reproduced baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_true_bs | 0 | 2.613 | 5.512 | 5.765 | 5.794 | 0.7773 | 0.000 |
| location_1px | 1 px | 2.687 | 5.592 | 5.804 | 5.833 | 0.7747 | 0.039 |
| location_2px | 2 px | 3.516 | 5.608 | 5.808 | 5.838 | 0.7741 | 0.043 |
| location_4px | 4 px | 3.901 | 5.829 | 5.915 | 5.944 | 0.7673 | 0.149 |
| location_8px | 8 px | 4.928 | 7.347 | 6.615 | 6.649 | 0.7241 | 0.850 |
| power_-5db | -5 dB | 2.613 | 5.512 | 5.955 | 5.985 | 0.7668 | 0.190 |
| power_-3db | -3 dB | 2.613 | 5.512 | 5.839 | 5.868 | 0.7731 | 0.074 |
| power_-1db | -1 dB | 2.613 | 5.512 | 5.769 | 5.799 | 0.7771 | 0.004 |
| power_+1db | +1 dB | 2.613 | 5.512 | 5.778 | 5.807 | 0.7765 | 0.012 |
| power_+3db | +3 dB | 2.613 | 5.512 | 5.849 | 5.879 | 0.7721 | 0.084 |
| power_+5db | +5 dB | 2.613 | 5.512 | 5.957 | 5.987 | 0.7663 | 0.191 |

## Intrinsic direction sensitivity

The current networks have no explicit azimuth input. The table therefore measures the ground-truth map difference between available Sionna directions, using only jointly valid pixels and the same link-budget offset.

| Angular bin | Mean angle | Direction pairs | Map RMSE difference (dB) | Map MAE difference (dB) |
|---|---:|---:|---:|---:|
| about_5_deg | 5.4 deg | 937 | 3.040 | 1.215 |
| about_15_deg | 15.5 deg | 948 | 4.628 | 2.617 |
| about_30_deg | 30.5 deg | 1788 | 7.461 | 4.643 |
| 40_to_80_deg | 60.3 deg | 3648 | 12.349 | 8.533 |
| 80_to_120_deg | 100.3 deg | 3634 | 16.763 | 12.881 |
| 120_to_180_deg | 150.2 deg | 5431 | 19.171 | 15.951 |

## Interface conclusion

- Stage1 explicitly consumes TX location through a marker map and a distance map.
- Stage2-A inherits location dependence through the frozen Stage1 prediction.
- Stage2-B receives no explicit power or azimuth parameter; both are implicit in the absolute sparse signal values and directional target.
- Reusing all three stages with a latent BS branch therefore requires an adapter that converts estimated location into the two Stage1 TX features. Explicit power/azimuth estimates need new conditioning channels or must remain auxiliary-only outputs.
- masked-SSIM uses an 11x11 local window, fixed data range of six training standard deviations, and only windows with at least 50% valid outdoor pixels.
