# Three-stage model BS-sensitivity report

## Frozen baseline

| Stage | Test RMSE (dB) | Test MAE (dB) |
|---|---:|---:|
| Stage1 DPM | 2.613 | 1.383 |
| Stage2-A isotropic | 5.512 | 2.701 |
| Stage2-B directional (100 points) | 5.765 | 3.062 |

## Reproduced pipeline sensitivity

| Condition | Error | Stage1 RMSE | Stage2-A RMSE | Final RMSE | Final unmeasured RMSE | Delta vs reproduced baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline_true_bs | 0 | 2.492 | 4.178 | 4.549 | 4.571 | 0.000 |
| location_1px | 1 px | 2.566 | 4.276 | 4.602 | 4.624 | 0.053 |
| power_-1db | -1 dB | 2.492 | 4.178 | 4.562 | 4.585 | 0.014 |
| power_+1db | +1 dB | 2.492 | 4.178 | 4.560 | 4.583 | 0.012 |

## Intrinsic direction sensitivity

The current networks have no explicit azimuth input. The table therefore measures the ground-truth map difference between available Sionna directions, using only jointly valid pixels and the same link-budget offset.

| Angular bin | Mean angle | Direction pairs | Map RMSE difference (dB) | Map MAE difference (dB) |
|---|---:|---:|---:|---:|
| about_15_deg | 17.0° | 3 | 4.897 | 2.787 |
| about_30_deg | 26.5° | 2 | 6.490 | 4.022 |
| 40_to_80_deg | 61.7° | 6 | 13.032 | 9.325 |
| 80_to_120_deg | 101.2° | 6 | 16.853 | 12.874 |
| 120_to_180_deg | 147.6° | 7 | 20.461 | 17.483 |

## Interface conclusion

- Stage1 explicitly consumes TX location through a marker map and a distance map.
- Stage2-A inherits location dependence through the frozen Stage1 prediction.
- Stage2-B receives no explicit power or azimuth parameter; both are implicit in the absolute sparse signal values and directional target.
- Reusing all three stages with a latent BS branch therefore requires an adapter that converts estimated location into the two Stage1 TX features. Explicit power/azimuth estimates need new conditioning channels or must remain auxiliary-only outputs.
