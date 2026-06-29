"""
Upload seismic paired SEG-Y datasets to a Hugging Face dataset repository.

Usage:
    export HF_NAMESPACE="your-org-name"   # or HF_USERNAME (personal account)
    export HF_TOKEN="your_hf_token"
    python tools/upload_dataset_to_hf.py                       # ground-roll default
    python tools/upload_dataset_to_hf.py --task multiples      # multiples dataset

Optional:
    --task TASK           Dataset task: ground_roll or multiples (default: ground_roll)
    --repo-name NAME      HF repo name (default depends on --task)
    --data-root PATH      Override data root (default depends on --task)
    --asset-dir PATH      Asset images directory (default depends on --task)
    --levels LIST         Comma-separated noise levels to upload for ground_roll (default: all)
    --dry-run             Scan and print what would be uploaded without uploading
"""

import argparse
import logging
import os
import re
import sys

from huggingface_hub import HfApi, create_repo, upload_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TASK_DEFAULTS = {
    "ground_roll": {
        "repo": "ground-roll",
        "data_root": "/data/shared/benchmark/ground_roll",
        "asset_dir": "./asset/ground_roll",
        "asset_repo_dir": "assets/ground_roll",
    },
    "multiples": {
        "repo": "multiples",
        "data_root": "/data/shared/benchmark/multiples",
        "asset_dir": "./asset/multiples",
        "asset_repo_dir": "assets/multiples",
    },
}

FILE_PATTERN = re.compile(r"SEGC3_shots1_9_(noisy|noise)_([\d.]+)\.sgy")
ASSET_EXTENSIONS = (".png", ".jpg", ".jpeg")
MULTIPLES_FILES = {
    "noisy": "total_nodw.sgy",
    "noise": "multiples.sgy",
}


def _asset_index(filename: str):
    """Return trailing numeric sample index from an asset filename, if present."""
    stem = os.path.splitext(filename)[0]
    m = re.search(r"(\d+)$", stem)
    if not m:
        return None
    return int(m.group(1))


def generate_triplet_gallery(
    asset_entries,
    *,
    noise_heading: str,
    caption_suffix: str,
) -> str:
    """Build image gallery grouped as noisy / clean / noise triplets."""
    if not asset_entries:
        return ""

    labels = {
        "noisy": "Noisy Data",
        "clean": "Clean Data",
        "noise": noise_heading,
    }
    by_kind = {kind: {} for kind in labels}
    for entry in asset_entries:
        kind = entry["kind"]
        if kind not in by_kind:
            continue
        idx = _asset_index(entry["filename"])
        if idx is None:
            continue
        by_kind[kind][idx] = entry

    indices = sorted(
        set(by_kind["noisy"]) & set(by_kind["clean"]) & set(by_kind["noise"])
    )
    if not indices:
        return ""

    sections = []
    for idx in indices:
        img_blocks = "\n\n".join(
            f'  <p><b>{label}</b></p>\n'
            f'  <img src="{by_kind[kind][idx]["repo_path"]}" '
            f'alt="Example {idx} {label.lower()}" width="92%">'
            for kind, label in labels.items()
        )
        sections.append(
            f"""### Example {idx}

<div align="center">

{img_blocks}

</div>

<div align="center"><i>{caption_suffix}</i></div>

"""
        )
    return "".join(sections)


def generate_dataset_card(entries, repo_id: str, asset_entries=None) -> str:
    """Generate a Hugging Face dataset card README.md."""
    levels = sorted({e["level"] for e in entries}, key=float)
    levels_str = ", ".join(levels)
    n_noisy = sum(1 for e in entries if e["kind"] == "noisy")
    n_noise = sum(1 for e in entries if e["kind"] == "noise")

    level_rows = "\n".join(
        f"| {lv} | SEGC3_shots1_9_noisy_{lv}.sgy | SEGC3_shots1_9_noise_{lv}.sgy | ~951 MB × 2 |"
        for lv in levels
    )

    image_section = generate_triplet_gallery(
        asset_entries,
        noise_heading="Ground-Roll Noise Label",
        caption_suffix="representative ground-roll attenuation sample at noise level 3.0.",
    )

    card = f"""---
tags:
- seismic
- ground-roll
- denoising
- seg-c3
- geophysics
- synthetic
task_categories:
- image-to-image
- other
size_categories:
- 10M-100M
pretty_name: SEG C3 Ground-Roll Dataset
viewer: false
---

# SEG C3 Ground-Roll Dataset

Paired noisy-input / noise-label SEG-Y volumes for supervised ground-roll attenuation, derived from the SEG C3 synthetic velocity model.

## Task

**Noise-label regression**: given a noisy pre-stack shot gather, predict the additive ground-roll noise component. The clean signal is recovered as:

```
denoised = noisy_input - predicted_noise
```

The noise labels serve as regression targets. Both input and label are 3D SEG-Y volumes with identical geometry.

## Dataset Description

- **Source**: SEG C3 synthetic velocity model ([wiki.seg.org/wiki/C3](https://wiki.seg.org/wiki/C3))
- **Geometry**: 9 regular shot gathers, 201 traces × 625 time samples per shot, dt = 2 ms
- **Noise modeling**: Reflection signals modeled with the acoustic wave equation; ground roll modeled with the elastic wave equation to capture its dispersive, low-velocity character
- **Noise intensity levels**: {levels_str}
- **Format**: Pre-stack SEG-Y (revision 1), IBM float
{image_section}
## File Structure

Each noise level has a matched pair of SEG-Y files:

| Level | Noisy Input | Noise Label | Size (approx) |
|-------|------------|-------------|---------------|
{level_rows}

**Total**: {n_noisy} noisy + {n_noise} noise SEG-Y files

## Loading Data

```python
import segyio
import numpy as np

def read_shot_gather(path, traces_per_shot=201):
    '''Read a regular SEG-Y file into (n_shots, n_traces, n_time).'''
    with segyio.open(path, "r", strict=False) as src:
        n_traces_total = src.tracecount
        n_shots = n_traces_total // traces_per_shot
        n_time = src.samples.size
        data = np.zeros((n_shots, traces_per_shot, n_time), dtype=np.float32)
        for i in range(n_shots):
            for j in range(traces_per_shot):
                data[i, j, :] = src.trace[i * traces_per_shot + j]
    return data

# Load a level-3.0 pair
noisy  = read_shot_gather("noisy/SEGC3_shots1_9_noisy_3.0.sgy")
noise  = read_shot_gather("noise/SEGC3_shots1_9_noise_3.0.sgy")
signal = noisy - noise   # clean reference
```

With `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="{repo_id}",
    filename="noisy/SEGC3_shots1_9_noisy_3.0.sgy",
    repo_type="dataset",
)
```

## Train / Val / Test Split

Shot-level sequential split by FFID (field file ID), avoiding trace leakage:

| Split | Shots | Fraction |
|-------|-------|----------|
| Train | 7     | 77.8%    |
| Val   | 1     | 11.1%    |
| Test  | 1     | 11.1%    |

The split is done at loading time (not pre-saved as separate files) so users can adjust the ratios.

## Preprocessing Recipe

The companion benchmark applies:

1. **Normalization**: `max_abs`, global scope — the entire noisy volume scaled to [-1, 1]; same stats applied to the noise label
2. **Patching**: Overlapping 2D patches (128 traces × 256 time samples), 50% overlap, yielding (1, H, W) tensors

No spherical-divergence correction is applied (raw amplitudes are used).

## Benchmark Results

See the companion model repository for full benchmark results across UNet, ResUNet, DnCNN, and Attention UNet architectures at each noise level.

## Citation

If you use this dataset, please cite the SEG C3 model and the companion benchmark:

```bibtex
@misc{{seg_c3_ground_roll,
  title={{SEG C3 Ground-Roll Attenuation Benchmark}},
  howpublished={{https://huggingface.co/datasets/{repo_id}}},
}}
```

## References

- SEG C3 Velocity Model: https://wiki.seg.org/wiki/C3
- `segyio` library: https://github.com/equinor/segyio
"""
    return card


def generate_multiples_dataset_card(entries, repo_id: str, asset_entries=None) -> str:
    """Generate a Hugging Face dataset card README.md for the multiples dataset."""
    n_noisy = sum(1 for e in entries if e["kind"] == "noisy")
    n_noise = sum(1 for e in entries if e["kind"] == "noise")

    rows = "\n".join(
        f"| {e['kind']} | `{e['repo_path']}` | {e['size_mb']:.1f} MB |"
        for e in sorted(entries, key=lambda x: x["kind"])
    )

    image_section = generate_triplet_gallery(
        asset_entries,
        noise_heading="Multiples Noise Label",
        caption_suffix="representative multiples attenuation sample.",
    )

    card = f"""---
tags:
- seismic
- multiples
- denoising
- marine-seismic
- geophysics
- synthetic
task_categories:
- image-to-image
- other
size_categories:
- 1G-10G
pretty_name: Marine Multiples Attenuation Dataset
viewer: false
---

# Marine Multiples Attenuation Dataset

Paired noisy-input / multiples-noise-label SEG-Y volumes for supervised marine multiples attenuation.

## Task

**Noise-label regression**: given a noisy pre-stack shot gather, predict the additive multiples component. The denoised signal is recovered as:

```text
denoised = noisy_input - predicted_multiples
```

The uploaded noise label is the supervised target. The clean reference used by the benchmark is computed as:

```text
clean_reference = noisy_input - multiples_label
```

## Dataset Description

- **Noisy input**: `noisy/total_nodw.sgy`
- **Multiples label**: `noise/multiples.sgy`
- **Geometry used by benchmark configs**: 638 traces per shot, 1,976 time samples per trace
- **Format**: Pre-stack SEG-Y, paired volumes with matching geometry

{image_section}
## File Structure

| Kind | Path | Size |
|------|------|------|
{rows}

**Total**: {n_noisy} noisy + {n_noise} noise SEG-Y files

## Loading Data

```python
import segyio
import numpy as np

def read_shot_gather(path, traces_per_shot=638):
    '''Read a regular SEG-Y file into (n_shots, n_traces, n_time).'''
    with segyio.open(path, "r", strict=False) as src:
        n_traces_total = src.tracecount
        n_shots = n_traces_total // traces_per_shot
        n_time = src.samples.size
        data = np.zeros((n_shots, traces_per_shot, n_time), dtype=np.float32)
        for i in range(n_shots):
            for j in range(traces_per_shot):
                data[i, j, :] = src.trace[i * traces_per_shot + j]
    return data

noisy = read_shot_gather("noisy/total_nodw.sgy", traces_per_shot=638)
multiples = read_shot_gather("noise/multiples.sgy", traces_per_shot=638)
clean_reference = noisy - multiples
```

With `huggingface_hub`:

```python
from huggingface_hub import hf_hub_download

noisy_path = hf_hub_download(
    repo_id="{repo_id}",
    filename="noisy/total_nodw.sgy",
    repo_type="dataset",
)
```

## Benchmark Split

The companion benchmark uses shot-level FFID splitting to avoid trace leakage. The current multiples configs use:

| Split | Shots |
|-------|-------|
| Train | 510 |
| Val | 64 |
| Test | 64 |

The split is done at loading time, so users can adjust it in their own configs.

## Preprocessing Recipe

The companion benchmark applies:

1. **Normalization**: `max_abs`, global scope on the noisy input; the same scale is applied to the multiples label.
2. **Patching**: overlapping 2D patches on the trace-time plane.
3. **Metric handling**: SNR can skip near-zero clean-reference patches with `min_signal_energy`.

No spherical-divergence correction is applied in the denoising training script.

## Citation

If you use this dataset, cite the dataset repository:

```bibtex
@misc{{marine_multiples_attenuation,
  title={{Marine Multiples Attenuation Dataset}},
  howpublished={{https://huggingface.co/datasets/{repo_id}}},
}}
```

## References

- `segyio` library: https://github.com/equinor/segyio
"""
    return card


def scan_ground_roll_data(data_root: str, levels=None):
    """Scan ground-roll data directory for SEG-Y file pairs and return upload entries."""
    entries = []
    noisy_dir = os.path.join(data_root, "noisy")
    noise_dir = os.path.join(data_root, "noise")

    if not os.path.isdir(noisy_dir):
        logger.warning("Noisy directory not found: %s", noisy_dir)
    if not os.path.isdir(noise_dir):
        logger.warning("Noise directory not found: %s", noise_dir)

    for dirpath, kind in [(noisy_dir, "noisy"), (noise_dir, "noise")]:
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            m = FILE_PATTERN.match(fname)
            if not m:
                continue
            file_kind, level = m.group(1), m.group(2)
            if file_kind != kind:
                continue
            if levels is not None and level not in levels:
                continue
            fpath = os.path.join(dirpath, fname)
            fsize = os.path.getsize(fpath)
            entries.append(
                {
                    "kind": kind,
                    "level": level,
                    "filename": fname,
                    "local_path": fpath,
                    "repo_path": f"{kind}/{fname}",
                    "size_mb": fsize / (1024 * 1024),
                }
            )
    entries.sort(key=lambda x: (float(x["level"]), x["kind"]))
    return entries


def scan_multiples_data(data_root: str, levels=None):
    """Scan fixed multiples SEG-Y pair and return upload entries."""
    if levels is not None:
        logger.warning("--levels is ignored for task=multiples; this dataset has one fixed pair.")

    entries = []
    for kind, fname in MULTIPLES_FILES.items():
        fpath = os.path.join(data_root, kind, fname)
        if not os.path.isfile(fpath):
            logger.warning("Expected %s file not found: %s", kind, fpath)
            continue
        fsize = os.path.getsize(fpath)
        entries.append(
            {
                "kind": kind,
                "level": "single",
                "filename": fname,
                "local_path": fpath,
                "repo_path": f"{kind}/{fname}",
                "size_mb": fsize / (1024 * 1024),
            }
        )
    entries.sort(key=lambda x: x["kind"])
    return entries


def scan_data(task: str, data_root: str, levels=None):
    """Scan data directory for task-specific SEG-Y upload entries."""
    if task == "ground_roll":
        return scan_ground_roll_data(data_root, levels)
    if task == "multiples":
        return scan_multiples_data(data_root, levels)
    raise ValueError(f"Unsupported task: {task!r}")


def generate_card(task: str, entries, repo_id: str, asset_entries=None) -> str:
    """Generate the task-specific Hugging Face dataset card."""
    if task == "ground_roll":
        return generate_dataset_card(entries, repo_id, asset_entries)
    if task == "multiples":
        return generate_multiples_dataset_card(entries, repo_id, asset_entries)
    raise ValueError(f"Unsupported task: {task!r}")


def scan_assets(asset_dir: str, repo_dir: str = "assets"):
    """Scan asset directory for sample images and return upload entries."""
    if not os.path.isdir(asset_dir):
        logger.warning("Asset directory not found: %s", asset_dir)
        return []
    entries = []
    for fname in sorted(os.listdir(asset_dir)):
        if not fname.lower().endswith(ASSET_EXTENSIONS):
            continue
        fpath = os.path.join(asset_dir, fname)
        fsize = os.path.getsize(fpath)
        # Determine kind from filename prefix: noisy1.png -> noisy, clean2.png -> clean, noise3.png -> noise
        stem = os.path.splitext(fname)[0]
        kind = stem.rstrip("0123456789")
        entries.append(
            {
                "kind": kind,
                "filename": fname,
                "local_path": fpath,
                "repo_path": f"{repo_dir.strip('/')}/{fname}",
                "size_mb": fsize / (1024 * 1024),
            }
        )
    return entries


def main():
    parser = argparse.ArgumentParser(description="Upload paired seismic SEG-Y dataset to Hugging Face.")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_DEFAULTS),
        default="ground_roll",
        help="Dataset task to upload (default: ground_roll)",
    )
    parser.add_argument(
        "--repo-name",
        default=None,
        help="HF repo name (default depends on --task)",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Data root (default depends on --task)",
    )
    parser.add_argument(
        "--asset-dir",
        default=None,
        help="Asset images directory (default depends on --task)",
    )
    parser.add_argument(
        "--levels",
        default=None,
        help="Comma-separated noise levels to upload for ground_roll (default: all found)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan and print what would be uploaded"
    )
    parser.add_argument(
        "--no-dataset-card", action="store_true", help="Skip uploading the dataset card"
    )
    parser.add_argument(
        "--no-assets", action="store_true", help="Skip uploading asset images"
    )
    args = parser.parse_args()

    defaults = TASK_DEFAULTS[args.task]
    repo_name = args.repo_name or defaults["repo"]
    data_root = args.data_root or defaults["data_root"]
    asset_dir = args.asset_dir or defaults["asset_dir"]
    asset_repo_dir = defaults["asset_repo_dir"]

    namespace = os.environ.get("HF_NAMESPACE") or os.environ.get("HF_USERNAME")
    token = os.environ.get("HF_TOKEN")

    level_set = None
    if args.levels:
        level_set = {s.strip() for s in args.levels.split(",")}

    entries = scan_data(args.task, data_root, level_set)
    if not entries:
        logger.warning("No matching SEG-Y files found in %s", data_root)
        sys.exit(0)

    asset_entries = [] if args.no_assets else scan_assets(asset_dir, asset_repo_dir)

    repo_id = f"{namespace}/{repo_name}" if namespace else repo_name
    total_size = sum(e["size_mb"] for e in entries)
    asset_size = sum(a["size_mb"] for a in asset_entries)
    n_noisy = sum(1 for e in entries if e["kind"] == "noisy")
    n_noise = sum(1 for e in entries if e["kind"] == "noise")
    n_levels = len({e["level"] for e in entries})

    logger.info(
        "Task=%s | root=%s | repo=%s | assets=%s",
        args.task, data_root, repo_id, "disabled" if args.no_assets else asset_dir,
    )
    logger.info(
        "Found %d SEG-Y files (%d noisy + %d noise) across %d dataset group(s), %.1f MB total",
        len(entries), n_noisy, n_noise, n_levels, total_size,
    )
    if asset_entries:
        logger.info("Found %d asset image(s), %.1f MB total", len(asset_entries), asset_size)

    if args.dry_run:
        logger.info("Dry-run mode — files that would be uploaded:")
        for e in entries:
            logger.info(
                "  [upload] %-50s  %7.1f MB  →  %s",
                e["filename"], e["size_mb"], e["repo_path"],
            )
        for a in asset_entries:
            logger.info(
                "  [upload] %-50s  %7.1f MB  →  %s",
                a["filename"], a["size_mb"], a["repo_path"],
            )
        if not args.no_dataset_card:
            logger.info("  [upload] README.md (dataset card)")
        return

    if not token:
        logger.error("HF_TOKEN environment variable is not set.")
        sys.exit(1)

    api = HfApi()
    logger.info("Creating / ensuring dataset repo: %s", repo_id)
    create_repo(repo_id, token=token, exist_ok=True, private=False, repo_type="dataset")

    # Fetch existing files in the repo for incremental (skip-already-existing) upload
    try:
        existing = set(api.list_repo_files(repo_id, repo_type="dataset", token=token))
        logger.info("Repo has %d existing file(s); will skip those.", len(existing))
    except Exception:
        logger.info("Could not list repo files (repo may be empty); uploading all.")
        existing = set()

    uploaded = 0
    skipped = 0
    failed = 0

    # Upload dataset card (always overwrite — it is regenerated each run)
    if not args.no_dataset_card:
        card = generate_card(args.task, entries, repo_id, asset_entries)
        try:
            api.upload_file(
                path_or_fileobj=card.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info("  [OK]     README.md (dataset card)")
            uploaded += 1
        except Exception as exc:
            logger.error("  [FAIL]   README.md — %s", exc)
            failed += 1

    # Upload SEG-Y files
    for e in entries:
        if e["repo_path"] in existing:
            logger.info("  [SKIP]   %-50s  %7.1f MB  (already exists)", e["repo_path"], e["size_mb"])
            skipped += 1
            continue
        try:
            upload_file(
                path_or_fileobj=e["local_path"],
                path_in_repo=e["repo_path"],
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info("  [OK]     %-50s  %7.1f MB", e["repo_path"], e["size_mb"])
            uploaded += 1
        except Exception as exc:
            logger.error("  [FAIL]   %s — %s", e["repo_path"], exc)
            failed += 1

    # Upload asset images
    for a in asset_entries:
        if a["repo_path"] in existing:
            logger.info("  [SKIP]   %-50s  %7.1f MB  (already exists)", a["repo_path"], a["size_mb"])
            skipped += 1
            continue
        try:
            upload_file(
                path_or_fileobj=a["local_path"],
                path_in_repo=a["repo_path"],
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
            )
            logger.info("  [OK]     %-50s  %7.1f MB", a["repo_path"], a["size_mb"])
            uploaded += 1
        except Exception as exc:
            logger.error("  [FAIL]   %s — %s", a["repo_path"], exc)
            failed += 1

    logger.info("Done. %d uploaded, %d skipped, %d error(s).", uploaded, skipped, failed)


if __name__ == "__main__":
    main()
