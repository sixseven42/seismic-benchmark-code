# First-Break Picking

Binary mask segmentation for seismic first-arrival time picking. The model reads SEG-Y shot gathers and outputs a probability map where each trace is 0 before the first break and 1 from the first break onward. Pick times are extracted as the time index of the first pixel whose probability exceeds a threshold.

## Directory layout

```
scripts/first_break_picking/
    first_break_data.py              Dataset, index builder, DataLoader factory
    train_pick_common.py             Shared training loop (run_training)
    train_pick_<model>.py            Training entry point per model
    launch_single_dataset/
        <dataset>/run_<model>_gpu<N>.sh   Single-GPU launch scripts

configs/first_break_picking/
    pick_<model>_seed<42|43|44>.yaml              Mixed-dataset configs (15 files)
    single_dataset/<dataset>/
        pick_<model>_<dataset>_seed<42|43|44>.yaml   Single-dataset configs (60 files)

model/first_break_picking/
    <model>.py                       Model definition (registered via @register_model)

results/first_break_picking/
    <exp_name>/                      Output dir per experiment
        checkpoints/                 best.pt + epoch_*.pt
        logs/                        train_log.txt, loss_history.csv, metrics_history.csv, *.png
        visualizations/              input/probability/target/overlay (4-panel)
```

## Models

| Script | Config prefix | Model description |
|--------|--------------|-------------------|
| `train_pick_unet.py` | `pick_unet` | Standard 2D U-Net |
| `train_pick_atten_unet.py` | `pick_atten_unet` | U-Net with CBAM attention gates |
| `train_pick_res_unet.py` | `pick_res_unet` | Residual U-Net |
| `train_pick_dncnn_seg.py` | `pick_dncnn_seg` | DnCNN-style segmentation CNN |
| `train_pick_dsu_net.py` | `pick_dsu_net` | Dynamic Snake U-Net |

## Datasets

### Single-dataset configs

Each config uses one specific SEG-Y file specified via `data.files`:

| Dataset | Config subdirectory |
|---------|-------------------|
| Brunswick (validation) | `single_dataset/brunswick_valid/` |
| Dongbei | `single_dataset/dongbei/` |
| Halfmile (validation) | `single_dataset/halfmile_valid/` |
| Lalor (validation) | `single_dataset/lalor_valid/` |

### Mixed-dataset configs

Configs under `configs/first_break_picking/` (no `single_dataset` subdirectory) have `data.files: null`, which means all SEG-Y files under `data.data_dir` are used together.

## Data preparation

### Required format

Two directories containing paired SEG-Y files with identical geometry:

```
<data_root>/
    data/          # Raw shot gathers (input)
        <name>.sgy
    label/         # Binary step masks (label)
        <name>_mask.sgy    # or <name>_mask.segy
```

### Mask convention

Label files are **binary step masks**:

- `0` — samples **before** the first break
- `1` — samples **from** the first break **onward**

Each trace must be either all 0 (no pick) or a single 0→1 transition that stays 1 to the end. Validity is checked at startup when `data.validate_labels: true`.

### File matching

The dataset loader pairs data and label files automatically: for `data/<name>.sgy` it looks for:

1. `label/<name>_mask.sgy`
2. `label/<name>_mask.segy`
3. Any single file matching `label/<name>*_mask.*` (sgy/segy)

When a data/mask pair lives outside the common directories, list the data path
in `data.files` and map it explicitly with `data.label_path_overrides`. Relative
override paths are resolved from `data.root`; absolute paths are used as-is.

### FFID / receiver-line segmentation

The dataset splits data at the **FFID** (FieldRecord) level for train/val/test, then **optionally** subdivides each FFID gather into receiver-line segments based on:

1. A trace-header field (e.g. `INLINE_3D`) via `gather_segment.line_id_header`
2. Geometry-based inference (GroupX/GroupY distance jumps) via `gather_segment.infer_line_from_geometry`

Patches are then sampled inside one receiver-line segment — this avoids mixing traces across physically distant receiver lines.

## Configuration reference

Example: `configs/first_break_picking/pick_unet_seed42.yaml`

### experiment

```yaml
experiment:
  name: first_break_pick_unet_geomseg_seed42
  output_dir: /path/to/results/first_break_picking
  seed: 42
  device: cuda:0
```

### data

```yaml
data:
  root: /path/to/dataset/segy_with_masks
  data_dir: data                    # relative to root (or absolute)
  label_dir: label                  # relative to root (or absolute)
  files: null                       # null = all .sgy/.segy under data_dir;
                                    # list = specific filenames, e.g. ["Dongbei.segy"]
  label_path_overrides: []          # optional [{data_path: ..., label_path: ...}]

  label_threshold: 0.5              # threshold for binarizing label masks
  prediction_threshold: 0.5         # threshold for binarizing model output at test time
  validate_labels: true             # check mask convention at startup
  label_check_traces: 2048          # max traces to sample for validation
  max_patches_per_split: null       # int cap or {train: N, val: M, test: K}

  gather_segment:
    enabled: true                   # subdivide FFIDs into receiver-line segments
    line_id_header: INLINE_3D       # SEG-Y trace header for receiver-line id
    infer_line_from_geometry: true  # fallback: use GroupX/GroupY distance jumps
    distance_floor: 1000.0          # minimum distance threshold (m)
    median_multiplier: 5.0          # threshold = max(floor, multiplier × median_neighbor_dist)

  split:
    train: 0.8                      # ratio of FFID blocks for training
    val: 0.1
    test: 0.1
    shuffle_ffids: true             # shuffle FFID blocks before splitting

  patch:
    trace: 128                      # number of traces per patch
    time: 512                       # number of time samples per patch
    trace_stride: 64                # sliding window stride (traces)
    time_stride: 256                # sliding window stride (time)

  loader:
    batch_size: 64
    num_workers: 4
    pin_memory: true
```

### preprocess

```yaml
preprocess:
  normalize_mode: max_abs           # max_abs | mean_std | minmax | none
  normalize_scope: gather           # gather (per receiver-line segment) | segment | patch
  clip_percentile: 99.5             # optional float; clip abs value before normalization
  normalize_eps: 1.0e-6             # epsilon to avoid division by zero
```

Per-segment normalization stats are computed once then cached so that every patch from the same segment uses consistent scaling.

### model / loss / metrics / optim / scheduler

```yaml
model:
  type: unet                        # unet | atten_unet | res_unet | dncnn_seg
  params:
    in_channels: 1
    out_channels: 1
    base_channels: 32
    depth: 4

loss:
  type: bce_dice                    # bce_dice | masked_bce
  params:
    bce_weight: 0.5
    dice_weight: 0.5
    smooth: 1.0
    pos_weight: null                # optional positive class weight for BCE

metrics:
  - name: dice                      # binary Dice (F1) score
    params: { threshold: 0.5 }
  - name: iou                       # intersection-over-union
    params: { threshold: 0.5 }
  - name: f1                        # alias for Dice
    params: { threshold: 0.5 }
  - name: HitRate1px                # pick within ±1 sample tolerance
    params: { threshold: 0.5 }
  - name: HitRate3px                # pick within ±3 samples
    params: { threshold: 0.5 }
  - name: HitRate5px                # pick within ±5 samples
    params: { threshold: 0.5 }
  - name: HitRate7px
    params: { threshold: 0.5 }
  - name: HitRate9px
    params: { threshold: 0.5 }
  - name: MeanAbsoluteError         # |pred_pick − true_pick| in samples
    params: { threshold: 0.5 }
  - name: RootMeanSquaredError      # RMSE of pick error in samples
    params: { threshold: 0.5 }
  - name: MeanBiasError             # signed error (pred − true)
    params: { threshold: 0.5 }
  - name: GatherCoverage            # fraction of traces where model produced any pick
    params: { threshold: 0.5 }

optim:
  type: adamw
  params:
    lr: 1.0e-4
    weight_decay: 1.0e-5

scheduler:
  type: cosine
  params:
    min_lr: 1.0e-6

train:
  epochs: 20
  grad_clip: 1.0
  progress_bar: true
  log_step: false
  log_interval: 20
  eval_interval: 1
  ckpt_interval: 1
  vis_interval: 1
  resume: null

log:
  log_dir: logs
  plot_interval: 1
```

#### Metric names reference

| Config name | Internally mapped to | Description |
|-------------|---------------------|-------------|
| `dice` | `DiceMetric` | Mean binary Dice score |
| `f1` | `F1Metric` | Alias for Dice |
| `iou` | `IoUMetric` | Binary IoU |
| `HitRate<N>px` | `PickWithin(tolerance=N)` | Fraction of picks within ±N samples |
| `pick_within_<N>` | `PickWithin(tolerance=N)` | Alternative naming |
| `MeanAbsoluteError` | `MeanAbsoluteError` | Mean absolute pick error (samples) |
| `RootMeanSquaredError` | `RootMeanSquaredError` | RMSE of pick error (samples) |
| `MeanBiasError` | `MeanBiasError` | Mean signed pick error |
| `GatherCoverage` | `GatherCoverage` | Fraction of traces with any pick |

Dice / F1 / IoU / GatherCoverage are **higher-is-better**. MAE / RMSE / MeanBiasError are **lower-is-better**.

## Running training

### Single run (manual)

```bash
python scripts/first_break_picking/train_pick_unet.py \
    --config configs/first_break_picking/pick_unet_seed42.yaml
```

The default config is auto-resolved to `<script_stem>_seed42.yaml` under `configs/first_break_picking/`, so `--config` can be omitted for the mixed-dataset configs.

### Launch scripts (single GPU per model × dataset)

Each shell script iterates over 3 seeds (42, 43, 44):

```bash
bash scripts/first_break_picking/launch_single_dataset/dongbei/run_unet_gpu0.sh
```

GPU allocation across models:

| GPU | Model |
|-----|-------|
| GPU 0 | UNet |
| GPU 1 | ResUNet |
| GPU 2 | AttnUNet |
| GPU 3 | DnCNN-Seg |

To run all 4 models on a single dataset in parallel:

```bash
for gpu in 0 1 2 3; do
  bash scripts/first_break_picking/launch_single_dataset/dongbei/run_*_gpu${gpu}.sh &
done
wait
```

## Evaluation

After training, metrics are logged per epoch in:

- `results/<exp_name>/logs/metrics_history.csv` — per-epoch metric values
- `results/<exp_name>/logs/step_loss_history.csv` — per-step training loss
- `results/<exp_name>/visualizations/epoch_*.png` — 4-panel diagnostic plots

Panel layout (left to right): **input** (grayscale) | **probability** (magma heatmap) | **target mask** | **overlay** (green = target picks, red = predicted picks)

## Adding a new model

1. Create `model/first_break_picking/<name>.py` with a `@register_model("<name>")` class.
2. Add `from . import <name>  # noqa: F401` to `model/first_break_picking/__init__.py`.
3. Copy `scripts/first_break_picking/train_pick_unet.py` to `train_pick_<name>.py` and update the docstring.
4. Copy a config from `configs/first_break_picking/` and update `model.type` + `model.params`.
5. (Optional) Add a launch script under `launch_single_dataset/<dataset>/`.

No edits to `utils/` or `train_pick_common.py` are required.
