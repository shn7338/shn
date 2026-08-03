# WinProp IRT label and Stage-2 pipeline

This directory is the resumable, non-overwriting **WinProp** pipeline for the
Stage2-A isotropic correction network and the Stage2-B directional signal-map
network. A separate Sionna RT paper-parameter pilot is included for an isolated
A/B decision; its labels are not mixed with either WinProp dataset.

## Physics used by the formal dataset

The installed WinProp 2020 build cannot represent the paper-style eight-bounce
profile. Its supported ceiling is 6 reflections, 2 diffractions and 6 combined
reflection/diffraction interactions. A `tile_000001` run at that ceiling did
not finish inside the practical validation window.

The accepted pilot and production profile therefore uses:

- 3.5 GHz, 512 m × 512 m, 4 m cells, 128 × 128 arrays.
- WinProp 3D IRT with multiple interactions.
- At most 2 reflections, 2 diffractions and 2 combined interactions.
- Scattering disabled.
- Receiver height 2 m.
- Transmitter at the exact grid centre and `max(building height) + 5 m`.
- One direct isotropic run and four direct directional WinProp runs per tile.
- At most 20 retained IRT rays per receiver pixel. A no-limit `tile_000007`
  diagnostic changed the maps by only about 0.19-0.21 dB RMSE but raised one
  process to about 2.9 GB, so the no-limit setting is unsuitable for safe
  parallel generation on this computer.
- Directional pattern: 65°/8° HPBW, 6.3 dBi maximum gain and 10° electrical
  downtilt embedded in the APA pattern. The project rotates azimuth only.
- Four distinct integer azimuths in `[0, 359]`, deterministically sampled per
  tile with seed `20260724`.

The output called `Antenna Path Loss.txt` by WinProp contains the negative-dB
path-gain values used by the existing Stage-1 dataset. Formal compact arrays
store those raw dB values as `float16`; invalid cells remain `NaN`.

## Coordinate and database policy

- Array row 0 is north and column 0 is west.
- If preprocessing snaps or expands the output extent, values are bilinearly
  interpolated in linear-power space onto the canonical 128 × 128 cell centres.
  The source grid, offset and interpolation decision are recorded per variant.
- The checked ODB is tried first. If the WinProp preprocessing DLL rejects it,
  the original READY ODB is retried into a separately named OIB.
- Existing OIBs, projects, raw results, compact labels and completion records
  are never overwritten. Re-running resumes completed work.

`progress.json` under the selected output root records the current tile, phase,
completed count and last error, so progress remains visible without relying on
the PowerShell window.

`tile_000001` has passed direct isotropic plus 0°/90° validation for coordinate
orientation, pattern gain and azimuth rotation. Its QA report is stored under
`E:\dac_winprop_irt2_direct_v1\qa\tile_000001`.

The WinProp User Guide states that APA values are gain/attenuation relative to
an isotropic radiator. A direct `tile_000007` repeat with the project antenna
gain field changed from 6.3 to 0 dBi produced the same map (about 0.026 dB
run-to-run RMSE), confirming that the 6.3 dBi pattern gain is not double-added.
Rare directional-minus-isotropic values above 6.3 dB persisted even with the
ray limit disabled; the pilot audit therefore checks the 99th percentile and
outlier fraction as well as reporting the absolute maximum.

## Dataset workflow

Run commands with the `sigmap` Python environment:

```powershell
$python = 'C:\Users\pc\miniconda3\envs\sigmap\python.exe'
```

### 1. Direct 32-tile pilot

```powershell
& $python projects\irt_label_pipeline\run_winprop_direct_pilot.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt2_direct_pilot32.json

& $python projects\irt_label_pipeline\summarize_irt_pilot.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt2_direct_pilot32.json

& $python projects\irt_label_pipeline\build_irt_shards.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt2_direct_pilot32.json

& $python projects\irt_label_pipeline\verify_irt_shards.py `
  --metadata E:\dac_winprop_irt2_direct_pilot32_v1\shard_metadata.json
```

The formal gate passes only when all 32 tiles contain one isotropic and four
directional 128 × 128 labels, the audit succeeds, and the shard hashes verify.

### 2. Spatially stratified 3,200-tile dataset

The committed selection contains 2,560 train, 320 validation and 320 test tiles
and preserves the existing 2,048 m spatial-block split. Production runs the
five required maps concurrently; measured pilot subtask durations predict about
1.85x propagation-stage speedup over four workers, while a sixth worker would
have no map to run.

Six verified pilot tiles are also present in the 3,200-tile selection. After
the pilot audit passes, `reuse_verified_pilot_tiles.py` copies their compact
labels and records the original raw-result provenance, avoiding unnecessary
recomputation.

The pilot measured about 6.4 MB of retained raw results per tile, so 3,200
tiles are projected to use about 20 GB. Production deletes each generated OIB
only after that tile's compact label and completion record are verified; the
final mmap shards add about 0.5 GB.

```powershell
& $python projects\irt_label_pipeline\run_winprop_direct_pilot.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt2_direct_3200.json

& $python projects\irt_label_pipeline\build_irt_shards.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt2_direct_3200.json

& $python projects\irt_label_pipeline\verify_irt_shards.py `
  --metadata E:\dac_winprop_irt2_direct_3200_v1\shard_metadata.json

& $python projects\irt_label_pipeline\compute_irt_shard_stats.py `
  --shard-root E:\dac_winprop_irt2_direct_3200_v1\shards `
  --shard-manifest E:\dac_winprop_irt2_direct_3200_v1\shard_manifest.csv `
  --selection-csv E:\dac_winprop_irt2_direct_3200_v1\selection_tiles.csv `
  --output E:\dac_winprop_irt2_direct_3200_v1\normalization_irt.json
```

### 2a. Positive-coordinate V2 selection

WinProp 2020 urban IRT rejects a prediction area whose border is negative.
The original selection contained 289 such tiles. The V2 selection retains the
other 2,911 tiles and replaces only those 289 entries with READY,
Stage1-complete, positive-coordinate tiles from the same immutable spatial
split. It preserves 2,560/320/320 train/validation/test tiles and has zero
2,048 m block overlap between splits.

The replacement audit is stored under
`E:\dac_winprop_irt2_direct_3200_positive_v2`. Stage-1 DPM predictions for all
3,200 V2 tiles are stored under
`E:\dac_stage1_dpm_predictions_positive3200_v2`; Stage2-A and Stage2-B load
these frozen predictions directly.

```powershell
& $python projects\irt_label_pipeline\replace_negative_irt_selection.py `
  --source-selection E:\dac_winprop_irt2_direct_3200_v1\selection_tiles.csv `
  --ready-manifest C:\Users\pc\Documents\大创\prediction_ready_all_odb.csv `
  --normalized-root D:\桌面\dac\normalized_4m_v1 `
  --prepared-root D:\桌面\dac\prepared_4m `
  --output-root E:\dac_winprop_irt2_direct_3200_positive_v2

& $python D:\桌面\dac\code\dpm_unet_stage1\batch_infer.py `
  --model-dir D:\桌面\dac\models\stage1_dpm_unet_baseline_v1 `
  --input-root D:\桌面\dac\normalized_4m_v1 `
  --output-root E:\dac_stage1_dpm_predictions_positive3200_v2 `
  --tiles-file E:\dac_winprop_irt2_direct_3200_positive_v2\selection_tiles.txt

& $python projects\irt_label_pipeline\reuse_verified_irt_tiles.py `
  --source-config projects\irt_label_pipeline\config_winprop2020_irt2_direct_3200.json `
  --target-config projects\irt_label_pipeline\config_winprop2020_irt2_direct_3200_positive_v2.json

& $python projects\irt_label_pipeline\run_winprop_direct_pilot.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt2_direct_3200_positive_v2.json
```

### 2b. User-selected WinProp 2020 maximum-interaction V3

The selected formal physics profile is six reflections, two diffractions, six
combined reflection/diffraction interactions and zero scatterings. IRT2 compact
labels remain preserved as a rollback dataset but must not be mixed with IRT6
targets. The positive-coordinate V2 selection and all frozen Stage-1 DPM
predictions are reused because neither depends on the IRT interaction count.

The IRT6 pilot uses two concurrent WinPropCLI processes, a 7,200 second
per-variant timeout and a 20-ray-per-pixel retention cap. Formal 3,200-tile
production is gated on a complete one-isotropic-plus-four-directional pilot
because a previous 6/2/6 run on `tile_000001` exceeded the practical
30-minute validation window.

```powershell
& $python projects\irt_label_pipeline\run_winprop_direct_pilot.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt6_direct_pilot_positive_v3.json `
  --tile tile_002865

& $python projects\irt_label_pipeline\run_winprop_direct_pilot.py `
  --config projects\irt_label_pipeline\config_winprop2020_irt6_direct_3200_positive_v3.json
```

### 2c. Isolated Sionna RT paper-parameter pilot

The Sionna pilot reproduces the synthetic parameters explicitly stated by
Geo2SigMap: 3.66 GHz, 512 m x 512 m, 4 m cells, 128 x 128 output, 7 million
rays, maximum depth 8, reflections and diffractions enabled, scattering
disabled, BS at `max(building height) + 5 m`, UE height 2 m, dual VH
polarization, one isotropic map and four random azimuth-plane directional maps.
The directional pattern uses 6.3 dBi boresight gain and 65 degree/8 degree
horizontal/vertical HPBW.

This experiment deliberately uses zero synthetic-data downtilt. The paper's
10/16/18 degree values describe three real measurement cells and are not stated
as synthetic-label parameters. The 30 dB pattern attenuation cap is an explicit
implementation assumption matching the existing WinProp sector pattern.

The paper used Sionna 0.15.1. This computer runs the isolated, pinned Sionna RT
2.0.1 environment because it supports the installed RTX 5070 Laptop GPU and
Python 3.12. Current Sionna radio-map diffraction is first-order, so these
labels are a paper-parameter reproduction rather than numerically equivalent
WinProp 6/2/6 labels. Treat the Sionna and WinProp label families as separate
experiments.

```powershell
$sionnaPython = 'E:\dac_sionna_rt_env\Scripts\python.exe'
$config = 'projects\irt_label_pipeline\config_sionna201_paper_tile002865.json'

python -m venv E:\dac_sionna_rt_env
& $sionnaPython -m pip install `
  -r projects\irt_label_pipeline\requirements_sionna201_paper.txt

& $sionnaPython projects\irt_label_pipeline\build_sionna_scene.py `
  --config $config --overwrite

& $sionnaPython projects\irt_label_pipeline\run_sionna_paper_pilot.py `
  --config $config --samples 7000000 --variants all --overwrite

& $sionnaPython projects\irt_label_pipeline\validate_sionna_paper_pilot.py `
  --config $config --samples 7000000
```

The completed `tile_002865` pilot is stored under
`E:\dac_sionna_paper_pilot\tile_002865`. All five maps were generated in
1.908 seconds, outdoor valid coverage was 98.98-99.06%, and validation passed
for array shape, indoor masks, finite values, non-identical directions and
requested azimuth orientation. A monitored 7-million-ray isotropic run used
about 1.16 GiB peak GPU memory. The directional maps are roughly 21-22 dB below
the isotropic map on average because the 8 degree vertical beam has no downtilt
while the BS is 47 m high; this is expected for the paper-faithful pilot.

#### 3.5 GHz WinProp-aligned frequency experiment

The paper-frequency result above remains immutable. The active 3.5 GHz Sionna
experiment has a separate config and output root:

- Config: `config_sionna201_35ghz_tile002865.json`.
- Output: `E:\dac_sionna_35ghz_pilot\tile_002865`.
- Comparison report:
  `E:\dac_sionna_35ghz_pilot\tile_002865\comparison_winprop_irt2\comparison.json`.

Changing only Sionna's frequency from 3.66 GHz to 3.5 GHz produced a median
increase of 0.389 dB, 0.469 dB MAE, 0.983 dB RMSE and 0.997 correlation on the
isotropic map. The frequency change is therefore small.

The same 3.5 GHz Sionna depth-8 isotropic map differs substantially from the
available 3.5 GHz WinProp IRT2 result for the same tile: on 8,449 common valid
pixels, Sionna minus WinProp has +13.46 dB mean bias, 14.10 dB MAE, 15.00 dB
RMSE, 6.61 dB bias-removed RMSE and 0.811 correlation.

An additional diagnostic aligned the readily controllable settings with the
WinProp IRT2 run: 3.5 GHz, depth 2, vertical-only TX/RX polarization and
10 degree directional downtilt. Its isotropic comparison still has +12.95 dB
mean bias, 13.63 dB MAE and 14.50 dB RMSE. This confirms that the main mismatch
comes from the two tools' scene, material, reflection/diffraction and radio-map
implementations rather than the 160 MHz frequency change. The aligned
diagnostic is not a production label profile.

```powershell
& $sionnaPython projects\irt_label_pipeline\run_sionna_paper_pilot.py `
  --config projects\irt_label_pipeline\config_sionna201_35ghz_tile002865.json `
  --samples 7000000 --variants all --overwrite

& $sionnaPython projects\irt_label_pipeline\compare_sionna_winprop.py `
  --sionna-root E:\dac_sionna_35ghz_pilot\tile_002865 `
  --reference-sionna-root E:\dac_sionna_paper_pilot\tile_002865 `
  --winprop-compact E:\dac_winprop_irt2_direct_3200_positive_v2\compact_tiles\tile_002865.npz `
  --samples 7000000 `
  --output-dir E:\dac_sionna_35ghz_pilot\tile_002865\comparison_winprop_irt2
```

#### Retained 3,200-tile 3.5 GHz depth-8 Sionna subset

The production configuration is
`config_sionna201_35ghz_depth8_3200.json`. It uses the frozen positive-coordinate
3,200-tile selection (train/validation/test = 2,560/320/320), one isotropic map
and four deterministic random-azimuth directional maps per tile. A 32-tile
pilot (26/3/3) is generated into the same output root as production, so accepted
pilot tiles are reused and never generated twice.

Physics and raster parameters are: 3.5 GHz, 512 m by 512 m area, 4 m cells,
128 by 128 arrays, 7,000,000 rays per transmitter, maximum depth 8, line of
sight/reflection/diffraction enabled, diffuse reflection/refraction disabled,
BS at the tile centre and `max(building height) + 5 m`, UE height 2 m, VH/VH
polarization, and zero synthetic downtilt. The directional pattern uses 6.3 dBi
boresight gain, 65 degree horizontal HPBW, 8 degree vertical HPBW and the
project's explicit 30 dB attenuation cap.

The batch runner is resumable and memory bounded. Each completed tile is fully
validated, compacted immediately to one compressed float16 NPZ, and committed
atomically before temporary raw maps and scene meshes are removed. At most two
tiles run concurrently on the current 8 GiB GPU. Create the pilot first, inspect
status, then run the full set and finalize shards:

```powershell
$launcher = 'C:\Users\pc\Documents\大创\projects\irt_label_pipeline\start_sionna_35ghz_dataset.ps1'

powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Pilot32 -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Status
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Full -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Finalize
```

To stop safely, run `-Mode Stop` in another PowerShell window. Active tiles are
allowed to finish; rerunning `Pilot32` or `Full` removes the stop flag and resumes
from completed tiles. The formal output root is
`E:\dac_sionna_35ghz_depth8_3200_v1`.

#### Active all-tile 3.5 GHz depth-8 Sionna dataset

The final dataset uses every physical source tile: 27,360 consecutive tile
directories from `tile_000001` through `tile_027360`. The earlier count of
27,361 was an ODB-file count that included an extra checked ODB; it was not the
number of independent tiles. All 27,360 source shapefiles, prepared metadata,
building-height arrays and Stage-1 inputs were audited as present.

The existing spatially isolated split is retained: train 21,789, validation
2,840 and test 2,731. Each tile has one isotropic plus four directional maps,
for 136,800 maps in total. The physics parameters are identical to the retained
3,200-tile configuration above. The all-tile run has its own immutable manifest
and output root: `E:\dac_sionna_35ghz_depth8_27360_v1`.

```powershell
$launcher = 'C:\Users\pc\Documents\大创\projects\irt_label_pipeline\start_sionna_35ghz_all_tiles.ps1'

powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode DryRun
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Pilot32 -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Status
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Full -Workers 2
powershell -NoProfile -ExecutionPolicy Bypass -File $launcher -Mode Finalize
```

Use `-Mode Stop` for a safe stop. Starting `Pilot32` or `Full` again clears the
stop flag and resumes without regenerating completed tiles. Finalization refuses
to create the formal shards until every manifest tile has completed.

Each 256-tile shard is memory-mappable and contains:

- `p_iso_XXXX.npy`: `[N, 128, 128]`, `float16`.
- `p_dir_XXXX.npy`: `[N, 4, 128, 128]`, `float16`.
- `azimuth_deg_XXXX.npy`: `[N, 4]`, `uint16`.
- `tile_number_XXXX.npy`: `[N]`, `uint32`.

## Stage2-A: DPM to isotropic IRT residual

The trained Stage-1 network is frozen. Stage2-A implements:

```text
P_iso_IRT_hat =
    P_iso_DPM_hat + U_IsoRefine(B, P_iso_DPM_hat)
```

It learns only the IRT-minus-DPM residual and reports the original DPM baseline
RMSE beside the corrected RMSE.

```powershell
& $python projects\irt_label_pipeline\train_stage2a_iso_refine.py `
  --config projects\irt_label_pipeline\config_stage2a_irt2_direct3200.json
```

## Stage2-B: directional sparse signal map

Stage-1 and the accepted Stage2-A checkpoint are both frozen. Stage2-B input is:

```text
[building, corrected isotropic IRT, sparse directional SS, sparse mask]
```

Direct directional WinProp path gain is converted online to signal strength:

```text
SS = P_dir_IRT + P_TX + G_TX + G_RX - IL
```

The link-budget draw is fixed per tile-direction. Training resamples 1–200
valid sparse pixels per epoch; final tests report 50, 100 and 200 points.

```powershell
& $python projects\irt_label_pipeline\train_stage2b_directional_ss.py `
  --config projects\irt_label_pipeline\config_stage2b_directional_ss_irt2_direct3200.json
```

## Rollback point

The pre-pipeline state is recoverable from:

- Git tag: `pre-irt-8bounce-20260724`
- Commit: `1153103160fe7ff0d997029e4182abee9dec6133`
- Verified backup: `E:\dac_checkpoints\pre_irt_8bounce_20260724_1153103`

Do not use `git reset --hard` in a dirty working tree. Restore into a separate
worktree or copy only the required versioned files from the verified backup.
