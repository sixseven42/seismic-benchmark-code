# Updates

> Chronological log of **important updates**. Record only: added/removed files, model-structure changes, loss/metric changes, dependency upgrades, API changes, critical bugfixes, open-source references. Trivia (typos, renames, reformatting) is **not** recorded.

## Entry template

```markdown
## YYYY-MM-DD - Title
- Context:
- Change:
- Impact:
- Follow-up:
```

---

## 2026-04-22 - Initial scaffold
- Context: First commit of the benchmark template library; empty skeleton only, no business logic.
- Change:
  - Created the repository layout: `tools/`, `model/`, `utils/`, `configs/`, `scripts/`, `results/`, `memory/`, `.cursor/rules/`.
  - Added per-folder `README.md` describing purpose and planned contents.
  - Added six Cursor rules under `.cursor/rules/` (all `alwaysApply: true`): `memory-first`, `no-duplication`, `efficiency-first`, `research-first`, `clarify-before-execute`, `no-auto-run`.
  - Initialized `memory/` templates: `README.md`, `code_design.md`, `techniques.md`, `updates.md`, `research_first.md`.
  - Added `.gitignore` that keeps `results/` out of version control while preserving `results/README.md`.
- Impact: A reusable starting point is available; contributors can now add data pipelines, models, and training scripts on top of a fixed structure.
- Follow-up: Define the shared config schema (`configs/default.yaml`), then the CLI skeletons in `scripts/`, then start filling `utils/`.

## 2026-04-22 - Add english-only rule and translate all docs to English
- Context: The project's documentation and rule files initially mixed Chinese and English, making it hard to share with non-Chinese collaborators and inconsistent with the agent's generated artifacts.
- Change:
  - Added `.cursor/rules/english-only.mdc` (`alwaysApply: true`), requiring all project docs, READMEs, source comments, log strings, config comments, and commit messages to be written in English. Agent↔user chat is explicitly out of scope.
  - Translated to English: root `README.md`; `tools/`, `model/`, `utils/`, `configs/`, `scripts/`, `results/` READMEs; all existing `.cursor/rules/*.mdc`; all `memory/` templates; comments in `.gitignore`.
  - Updated the root `README.md` rule list from six to seven entries to include `english-only`.
- Impact: The repository is now fully in English. Future contributors are expected to keep it that way; mixed-language content should be translated within the same change set.
- Follow-up: Enforce the rule in future PR reviews; when filling `scripts/` and `utils/`, keep docstrings and log strings in English.

## 2026-04-22 - Add modular training skeleton (registry + factory)
- Context: The user plans to add datasets, models, losses, and metrics incrementally. A uniform plugin layout is needed so that new implementations do not require edits to `scripts/train.py` or `utils/__init__.py`.
- Change:
  - Added `configs/default.yaml` with a unified `{ type, params }` schema for every pluggable block (`data`, `model`, `loss`, `metrics`, `optim`, `scheduler`).
  - Added `utils/datasets.py` (`BaseArrayDataset`, `NpyDataset`, `MatDataset`, `DATASET_REGISTRY`, `register_dataset`, `build_dataset`, `build_dataloader`).
  - Added `utils/losses.py` (`BaseLoss`, `MSELoss`, `L1Loss`, `WeightedMSELoss`, `LOSS_REGISTRY`, `register_loss`, `build_loss`).
  - Added `utils/metrics.py` (`BaseMetric`, `PSNR`, `SSIM`, `MAE`, `SNR`, `METRIC_REGISTRY`, `register_metric`, `build_metrics`, `compute_metrics`).
  - Added `utils/visualization.py` (`plot_sample`, `plot_loss_curve`, `plot_metrics_curve`).
  - Added `utils/logger.py` (`TrainingLogger` with text log + loss CSV + metrics CSV).
  - Added `utils/train_utils.py` (`set_seed`, `load_config`, `setup_experiment_dir`, `build_optimizer`, `build_scheduler`, `save_checkpoint`, `load_checkpoint`, `find_latest_checkpoint`, `train_one_epoch`, `evaluate`).
  - Added `model/registry.py`, `model/placeholder_model.py`; updated `model/__init__.py` to re-export and trigger registration.
  - Updated `utils/__init__.py` to re-export the public API and trigger registrations.
  - Added `scripts/train.py` as a component-agnostic entry point (only CLI parsing + factories + epoch loop).
  - Appended "How to add" sections to every subfolder README (`tools/`, `model/`, `utils/`, `configs/`, `scripts/`).
  - Appended a "Maintenance Guide" section to `memory/code_design.md` that captures architectural invariants and the per-change checklist.
  - All function bodies are placeholders (`raise NotImplementedError` or `# TODO`) so the skeleton is not runnable but the interfaces are fixed.
- Impact: Contributors can add new losses / metrics / datasets / models by creating a single new file with a `@register_*` decorator and referencing it from YAML; the training script stays untouched. The maintenance guide in `memory/code_design.md` documents the contract so the pattern does not drift as the project grows.
- Follow-up: Fill in the placeholder implementations in the order that matches the first real experiment; when adding a real model, update `research_first.md` with the open-source reference that was used.

## 2026-04-22 - Add tools/segy_read.py (shot-gather SEG-Y reader)
- Context: The first real data task uses pre-stack SEG-Y files in `/data/shared/SEGC3/`. The file `SEG_45Shot_shots1-9.sgy` is a regular dataset (9 shots, equal trace count per shot); the `SEG_C3NA_ffid_*.sgy` files are irregular and require trace-header-based splitting.
- Change:
  - Added `tools/segy_read.py` with:
    - `read_regular_shots(path, n_shots=9, ...) -> (ndarray[n_shots, n_traces, n_samples], headers_dict)` — default path, assumes equal trace count per shot and optionally verifies regularity by checking that each slice shares a single `FieldRecord` value.
    - `read_irregular_shots_by_header(path, header_key="FieldRecord", ...)` — placeholder. Docstring documents the full segyio-based recipe: read the header array, find shot boundaries via `np.unique(..., return_index=True)`, and split the trace array accordingly.
    - `inspect_segy(path)` — quick metadata probe (trace count, samples, sample interval, unique FFID count, per-FFID trace count min/max).
    - `_demo()` / `__main__` — prints metadata and shapes for `SEG_45Shot_shots1-9.sgy`. Safe to run manually.
  - Default returned headers: `FieldRecord`, `SourceX`, `SourceY`, `GroupX`, `GroupY` as raw integers (no coordinate scaling applied).
  - Updated `tools/README.md` to list `segy_read.py` under "Available modules".
  - Added an entry to `memory/research_first.md` documenting the selection of `segyio` over `obspy` and a hand-rolled parser.
- Impact: The project can now consume pre-stack SEG-Y data end-to-end for the regular case. Dataset wrappers under `utils/datasets.py` can import from `tools.segy_read` once they need shot-gather tensors.
- Follow-up:
  1. Implement `read_irregular_shots_by_header` when the first `SEG_C3NA_ffid_*.sgy` file is actually consumed.
  2. Add a SEG-Y-backed dataset under `utils/datasets.py` (e.g. `@register_dataset("segy_shots")`) that wraps `read_regular_shots` for the benchmark training loop.
  3. Record the `segyio` version in the project dependency list once it is pinned.

## 2026-04-22 - Rework tools/segy_read.py API (traces_per_shot + time_downsample)
- Context: The previous API took `n_shots` as input, which did not match the common seismic convention of describing a shot gather by its *length in traces*. Downstream benchmark datasets also need to optionally halve the temporal sampling rate without aliasing.
- Change:
  - `read_regular_shots` signature: replaced `n_shots: int = 9` with a required `traces_per_shot: int`. `n_shots` is now derived as `n_traces // traces_per_shot`. The output shape is explicitly documented as `(n_shots, shot_length, time_length)`.
  - Added `time_downsample: int = 1` to both `read_regular_shots` and `read_irregular_shots_by_header`. When `> 1`, anti-aliased decimation is applied along the time axis using `scipy.signal.decimate(..., axis=-1, zero_phase=True)` (lazy import so scipy is only required when actually downsampling). Typical value is 2.
  - `_demo()` now calls `inspect_segy` first, infers `traces_per_shot` from `traces_per_ffid_min_max`, and prints both `time_downsample=1` and `time_downsample=2` shapes for side-by-side comparison.
  - Updated `tools/README.md` to reflect the new API and the anti-aliasing policy.
  - The verify_ffid / header handling semantics are unchanged; regularity is still validated against the `FieldRecord` header.
- Impact: The public API of `tools.segy_read` is now better aligned with seismic vocabulary (shot-gather length) and supports temporal downsampling that is safe for signal content. Any future code that wrapped the old `n_shots=9` signature must be updated; there are no such callers inside this repository yet.
- Follow-up: When the SEG-Y-backed dataset under `utils/datasets.py` is written, expose `traces_per_shot` and `time_downsample` via YAML so experiments can vary them without code changes.

## 2026-04-22 - Add tools/preprocessing.py (per-shot preprocessing primitives)
- Context: Downstream benchmarks need a small, reproducible toolbox that operates directly on the `(n_shots, n_traces, n_time)` arrays produced by `tools.segy_read`, without pulling in training-time dependencies (torch, Dataset wrappers). The first four primitives needed are: noise injection, trace masking, spherical-divergence gain, and normalization with percentile clipping.
- Change:
  - Added `tools/preprocessing.py` with four independent functions; all are pure-numpy, vectorized over the shot axis, and accept either `(n_shots, n_traces, n_time)` or `(n_traces, n_time)` input.
    - `add_noise(shots, kind, snr_db, rng)` — per-shot SNR control with `SNR_dB = 10*log10(var_signal/var_noise)`; `kind="gaussian"` uses `N(0, sigma_i)`, `kind="poisson"` uses a scaled-Poisson sampler shifted to be non-negative and scale-matched to the target SNR (Foi et al. 2008 style).
    - `mask_traces(shots, mode, ratio, rng)` — three missing patterns (`"uniform"`, `"random"`, `"continuous"`); masked traces are filled with zeros and the boolean mask `(n_shots, n_traces)` is returned for downstream loss weighting.
    - `spherical_divergence_correction(shots, dt, t0, power)` — multiplies each sample by `(t + t0)**power`; `power=2.0` default per Yilmaz 2001.
    - `normalize(shots, mode, clip_percentile, per)` — supports `"minmax"`, `"max_abs"`, `"mean_std"` with optional absolute-percentile clipping applied **before** computing reducers; scope of the reducers is selectable via `per = "shot" | "trace" | "global"` (default `"shot"`).
  - Randomness flows through a local `np.random.Generator` (`rng` or `seed`) so functions never touch the global RNG.
  - Updated `tools/README.md` to list `preprocessing.py` under "Available modules".
- Impact: The SEG-Y reader now has a matching preprocessing layer that can feed into `utils/datasets.py` (once the SEG-Y-backed dataset is added). The four primitives are decoupled, composable, and reproducible.
- Follow-up:
  1. Expose the preprocessing primitives as optional transforms in the upcoming `@register_dataset("segy_shots")` wrapper under `utils/datasets.py`.
  2. If per-trace SNR or per-sample dead-trace masks are needed later, extend `add_noise` / `mask_traces` with additional modes rather than forking new files.

## 2026-04-22 - Add test.ipynb (smoke test for tools/preprocessing.py)
- Context: A visual sanity check is needed when iterating on the four preprocessing primitives. A single self-contained notebook is easier to share than ad-hoc scripts.
- Change:
  - Added `test.ipynb` at the project root. Loads one shot gather from `/data/shared/SEGC3/SEG_45Shot_shots1-9.sgy` via `tools.segy_read.read_regular_shots`, then exercises `add_noise` (gaussian/poisson), `mask_traces` (uniform/random/continuous), `spherical_divergence_correction`, and `normalize` (all three modes and all three `per` scopes, plus the `clip_percentile` effect).
  - Plots use the `seismic` colormap with a percentile-based symmetric clip (`+/- q99(|x|)`) to keep strong samples from washing out the display.
  - The notebook is a local testing artifact; it has no external dependencies beyond `numpy`, `matplotlib`, and the two already-used project modules.
- Impact: Provides a reproducible visual reference for the preprocessing primitives; no production code changes.
- Follow-up: If the number of test artifacts grows, move them into a dedicated `notebooks/` folder and mirror the convention in `code_design.md`.

## 2026-04-22 - Rework mask_traces uniform mode (stride semantics)
- Context: The `"uniform"` branch of `tools.preprocessing.mask_traces` was originally parameterized through `ratio` (fraction missing), which does not match the seismic convention "keep every k-th trace". The user asked for an explicit stride.
- Change:
  - Added a keyword-only `uniform_stride: Optional[int] = None` argument to `mask_traces`. When `mode="uniform"`:
    - If `uniform_stride` is given, it is used directly (`>= 2` required; raises otherwise); the mask keeps indices `0, stride, 2*stride, ...`.
    - If `uniform_stride` is omitted, a stride is derived from `ratio` as `max(2, round(1 / (1 - ratio)))` for backward compatibility.
  - `uniform_stride` is rejected (ValueError) when `mode != "uniform"`.
  - `"random"` and `"continuous"` modes are unchanged (still driven by `ratio`).
  - Updated `tools/README.md` entry for `mask_traces` and the mask demo cell in `test.ipynb` (uses `uniform_stride=3` explicitly).
- Impact: `mask_traces(shots, mode="uniform", uniform_stride=k)` now matches the seismic convention directly. Existing call sites that only pass `ratio` keep working via the derived stride.
- Follow-up: If an explicit `ratio` value is later needed for the `"uniform"` branch (e.g. pixel-level schedulers), switch the signature to accept a `ratio` float for uniform mode as well, at the cost of the current compact derivation.

## 2026-04-25 - Migrate SEG-Y demo path and add coherent-noise placeholders
- Context: The repository was copied to a new server where the original demo SEG-Y path `/data/shared/SEGC3/SEG_45Shot_shots1-9.sgy` is no longer reachable; the actual file now lives at `/data/liuqi/code/MAE/5d-transformer/data/SEGC3-45/SEG_45Shot_shots1-9.sgy`. Independently, downstream benchmarks will need to inject coherent (structured) noise in addition to the existing IID Gaussian / Poisson — specifically linear-moveout (ground-roll-like) and hyperbolic-moveout (multiples-like) events — so the API surface needs reserved entry points before any caller is written.
- Change:
  - Updated the hard-coded demo path in `tools/segy_read.py::_demo` and in the two cells of `test.ipynb` (markdown intro + the `SEGY_PATH = Path(...)` cell) to the new location. Historical entries in `memory/updates.md` and `memory/research_first.md` are intentionally left untouched.
  - Added two placeholder functions to `tools/preprocessing.py`: `add_linear_noise(shots, dt, *, rng)` and `add_hyperbolic_noise(shots, dt, *, rng)`. Both raise `NotImplementedError`, document the planned moveout equation (`t(x) = t0 + (x - x0)/v` and `t(x) = sqrt(t0**2 + ((x - x0)/v)**2)` respectively), and deliberately keep the parameter list minimal so that the final signature is fixed by the first real use case rather than guessed now.
  - Refreshed the module docstring of `tools/preprocessing.py` to list "implemented primitives" and "placeholders", and extended `tools/README.md` with two bullets that explicitly mark both functions as "NOT IMPLEMENTED YET (placeholder)" and remind the implementer to update this README and `memory/updates.md` once they are filled in.
- Impact: The demo SEG-Y reader is runnable again on this server. The preprocessing module now has reserved, documented entry points for coherent noise, so future code can `from tools.preprocessing import add_linear_noise, add_hyperbolic_noise` against a stable name and fail loudly if it is called before the bodies are written. No behaviour change for the four implemented primitives; downstream training code is unaffected.
- Follow-up:
  1. Implement `add_linear_noise` / `add_hyperbolic_noise` (band-limited Ricker wavelet, multi-event support, SNR vs absolute-amplitude switch) when the first experiment that needs structured noise is queued; finalise the signature at that point and remove the "NOT IMPLEMENTED YET" tags here, in `tools/README.md`, and in the module docstring.
  2. If the new SEG-Y location is itself temporary, parameterise the demo path through an environment variable (e.g. `BENCHMARK_SEGY_DEMO`) so future moves do not require code edits.

## 2026-04-25 - Override-stats normalization, patching module, shared array helpers
- Context: Two follow-ups against `tools/preprocessing.py` and the `tools/` package:
  (a) downstream training/evaluation needs to apply pre-computed normalisation
  scalars (e.g. statistics computed once on the training split) at inference
  time, which the existing `normalize()` did not support; (b) the next
  benchmark stage needs patchify / unpatchify primitives over the
  `(trace, time)` plane, with both an overlapping regular grid (for input
  reconstruction, e.g. denoising / inpainting) and random sampling (for
  training-time augmentation), and configurable 3D vs 4D output for direct
  consumption by `nn.Conv2d`-style models.
- Change:
  - `tools/preprocessing.py::normalize` — added an `override_stats: Optional[Dict[str, float]] = None`
    keyword argument. Required keys per mode: `"minmax"` -> `{"min", "max"}`,
    `"max_abs"` -> `{"max_abs"}`, `"mean_std"` -> `{"mean", "std"}`. When set,
    `per` must be `"global"` and `clip_percentile` must be `None`
    (user-supplied scalars are authoritative); both rules raise
    `ValueError` explicitly. The returned `stats` dict echoes the supplied
    values for symmetry with the auto-computed path. Default behaviour
    (`override_stats=None`) is unchanged, so existing callers are
    unaffected. Parameter name is `override_stats` (not `stats`) to avoid
    shadowing the existing local return-dict variable.
  - `tools/patching.py` — **new module**. Pure numpy, no torch / no I/O.
    Three functions:
      * `patchify_uniform(data, patch_size, overlap=0.0, output_ndim=3)` —
        sliding regular grid; stride per axis = `max(1, round(p * (1 - overlap)))`;
        the last patch on each axis is anchored to the tail so the full
        input is always covered. Vectorised via
        `numpy.lib.stride_tricks.sliding_window_view` + a single advanced-
        indexing call (no per-patch Python loop, satisfies `efficiency-first`).
      * `patchify_random(data, patch_size, n_patches, output_ndim=3, rng=None)` —
        independent uniform sampling per shot; `n_patches` is per shot
        (total output count = `n_shots * n_patches`). Vectorised via
        broadcast fancy indexing; no Python loop over patches.
      * `unpatchify_uniform(patches, info)` — inverse of the uniform mode;
        averages overlapping regions (`sum / count`); restores 2D shape
        when the original input was 2D. Accepts both 3D and 4D patch
        tensors (auto-squeezes the singleton channel axis). The single
        Python loop is over the patch grid (typically O(10^2)), not over
        data points, so the inner ops stay vectorised.
      Input shapes accepted by all `patchify_*` functions: 2D
      `(n_traces, n_time)` (auto-promoted) or 3D
      `(n_shots, n_traces, n_time)`; output is always
      `(n_patches_total, ...)` with `n_patches_total = n_shots * n_per_shot`.
      Output ndim is selectable: 3 -> `(P, h, w)`, 4 -> `(P, 1, h, w)`.
      The `info` dict carries `shape`, `was_2d`, `patch_size`,
      `trace_starts`, `time_starts`, `n_shots`, `n_per_shot`, `output_ndim`,
      `mode`, which is everything needed to reconstruct.
  - `tools/_array_utils.py` — **new private helper module** (leading
    underscore on the file name marks it package-internal). Hosts the
    `as_3d` / `restore` / `as_generator` helpers and the `RNGLike` type
    alias. Both `preprocessing.py` and `patching.py` now import from here
    instead of carrying their own copies. This consolidates a
    >10-line duplication that the new `patching.py` would otherwise have
    introduced and satisfies the project-wide `no-duplication` rule. The
    public API of `preprocessing.py` is unchanged; `_float` (integer ->
    float promotion) stays in `preprocessing.py` because it is only
    relevant to the noise / normalisation primitives.
  - `tools/README.md` — extended the `normalize` bullet to document
    `override_stats`; added a new `patching.py` block listing all three
    functions with their semantics; added a one-line note for
    `_array_utils.py` clarifying that it is package-internal.
  - Module docstring of `tools/preprocessing.py` updated so the
    "Implemented primitives" line mentions `override_stats`.
- Impact: Downstream code can now (1) apply training-set statistics
  uniformly at inference via `normalize(..., per="global", override_stats=...)`,
  and (2) patchify shot gathers into `(P, 1, h, w)` tensors for direct
  consumption by 2D conv models, with reconstruction support for the
  uniform mode. The `_array_utils.py` extraction is API-neutral but is a
  prerequisite for any future `tools/` module that needs the same shape /
  RNG helpers. No existing call site breaks; all changes are additive.
- Follow-up:
  1. Once `utils/datasets.py` gets a SEG-Y-backed dataset, expose the
     uniform / random patching choice and `override_stats` via YAML so
     experiments can switch them without code changes.
  2. If a `blend != "average"` reconstruction is ever needed (e.g. taking
     the central-pixel value of each patch to avoid edge bleeding), add
     a `blend` keyword to `unpatchify_uniform` rather than forking a new
     function.
  3. Consider adding a `denormalize(...)` helper that consumes the
     `stats` dict returned by `normalize(...)` for symmetric round-trips
     when the next experiment needs it.

## 2026-04-25 — Add `concise-docs` rule; trim docstrings; extend `test.ipynb`.

- Context: Module and function docstrings had grown verbose (history, "how to add" tutorials, multi-paragraph design rationale), duplicating content already in the `README.md` / `memory/` files.
- Change:
  - New rule `.cursor/rules/concise-docs.mdc`; listed alongside the others in the root `README.md`.
  - Trimmed module / class / function docstrings across `tools/` (`segy_read`, `preprocessing`, `patching`, `_array_utils`), `utils/` (`datasets`, `losses`, `metrics`, `train_utils`, `logger`, `visualization`), `model/` (`registry`, `placeholder_model`), and `scripts/train.py` to one-sentence + `Parameters` / `Returns` (or a single `NOT IMPLEMENTED YET` line for placeholders).
  - `tools/README.md`: condensed the `preprocessing.py` / `patching.py` / `_array_utils.py` bullets to match.
  - `test.ipynb`: three new sections — (6) `normalize(override_stats=...)` round-trip + error paths, (7) `patchify_uniform` + `unpatchify_uniform` full coverage + round-trip, (8) `patchify_random` shape / reproducibility / non-invertibility; Summary bullets updated.
- Impact: No runtime behaviour changes; imports and signatures are unchanged. Future docstrings / memory entries are expected to follow `concise-docs`.
- Follow-up: Apply `concise-docs` to any new comments going forward; historical `memory/updates.md` entries are **not** rewritten (per `memory-first`).

## 2026-04-25 — Implement reconstruction metrics (`mse` / `rmse` / `mae` / `snr` / `psnr` / `ssim`).

- Context: `utils/metrics.py` only had empty placeholders; training evaluation needs concrete scalar metrics for denoising / restoration.
- Change:
  - Implemented the six metrics with torch. All reductions are **global** (`mean_all` / `sum_all`), so `RMSE == sqrt(MSE)` and `PSNR == 10·log10(data_range**2 / MSE)` hold exactly (matching the torchmetrics convention). `SSIM` self-implemented via Gaussian-window depthwise conv following Wang et al. 2004 (no new dependency); accepts `(B, C, H, W)` or `(B, H, W)`. Inputs are `.detach()`-ed and cast to float; formulas are embedded in each docstring.
  - `utils/README.md`: metrics bullet lists the implemented names + formulas.
  - `test.ipynb`: new Section 9 smoke-tests all six metrics on SEG-Y patches, asserts the algebraic identities, and checks error paths (shape mismatch, too-small spatial).
- Impact: `scripts/train.py` / the training loop can now build real metrics via YAML (`metrics: [{name: psnr, params: {data_range: 1.0}}, ...]`) without monkey-patching. No signature changes to the registry or factories.
- Follow-up: Add MS-SSIM / LPIPS / FID only when a specific experiment needs them; keep the module single-file until then.

## 2026-04-25 — Add `reduction` to `RMSE` / `SNR` / `PSNR` (default `per_sample`).

- Context: For batched seismic input `(B, 1, H, W)`, the original global reduction reported a single SNR/PSNR over the whole batch; experiments typically want the mean of per-shot SNR/PSNR instead.
- Change: `RMSE`, `SNR`, `PSNR` now take `reduction: Literal["global", "per_sample"]` (default `"per_sample"`); `_flatten_per_sample` reshapes to `(B, N)` and computes per-sample scores before averaging. `MSE` / `MAE` / `SSIM` are linear / uniformly weighted, so a knob would be a no-op and is kept off. Docstrings carry both formulas; `utils/README.md` and `test.ipynb` Section 9 print per-sample vs global side by side and only assert the textbook identities (`RMSE == sqrt(MSE)`, `PSNR == 10·log10(range²/MSE)`) for `reduction="global"`.
- Impact: Default numbers from `snr` / `psnr` / `rmse` change vs the previous build (per-sample is no longer equal to global). YAML configs can pin behaviour explicitly: `params: { reduction: global }` or `{ reduction: per_sample }`.
- Follow-up: When adding metrics with non-linear post-reduction steps (e.g. log-spectral-distance), expose the same `reduction` knob for consistency.

## 2026-04-25 — Implement `utils/visualization.py` + `utils/logger.py`.

- Context: Both modules were placeholders; training visualizations (input / pred / target / residual) and per-epoch logging are needed for diagnostics.
- Change:
  - `visualization.py`: `plot_sample` renders 4 panels (`input | prediction | target | residual = pred - target`) with `target`, 2 panels without; auto-squeezes `(B, C, H, W)` to a 2D map; symmetric `±q99(|x|)` per panel with `gray` cmap. `plot_loss_curve` / `plot_metrics_curve` skip all-NaN series and auto log-Y when losses span > 2 decades. **New** `visualize_random_sample(model, loader, save_path, device, title=None, seed=None)` draws a random `dataset[idx]`, runs the model under `eval() + no_grad`, restores `model.training`, suffixes `idx=...` to the title for reproducibility, and forwards to `plot_sample`. `seed=None` -> different sample each call.
  - `logger.py`: `TrainingLogger` opens `train_log.txt` (append) plus `loss_history.csv` (cols `epoch, lr, *loss_keys`) and `metrics_history.csv` (cols `epoch, *metric_keys`); headers only written when the file is empty so resumed runs keep growing. `info()` prefixes a timestamp and tees to stdout. `log_epoch(epoch, losses, metrics, extras)` appends one row per CSV (NaN-fills missing keys, `extras["lr"]` flows into the loss CSV) and routes a one-line summary through `info()`. `flush() / close()` are idempotent; `__del__` is best-effort.
  - `__init__.py` re-exports `visualize_random_sample`. `scripts/train.py` swaps the TODO `plot_sample(None, None, None, ...)` for `visualize_random_sample(...)` wrapped in `try/except (NotImplementedError, AttributeError)` for the placeholder dataset state.
  - `utils/README.md`: condensed bullets describe both modules + the random-viz seeding contract.
  - `test.ipynb` Section 10: round-trips `plot_sample` (4 / 2 panels), `plot_loss_curve` / `plot_metrics_curve` on synthetic histories, `TrainingLogger` (asserts CSV columns + row counts + text-log line count), and `visualize_random_sample` with a tiny in-memory `(noisy, clean)` dataset + a 5-pt smoothing model (verifies `model.training` is restored).
- Impact: Diagnostics path is end-to-end runnable today; training script will use random validation samples per epoch by default. Resumed runs keep CSV continuity.
- Follow-up: When a real `BaseArrayDataset._build_index` is implemented, the `try/except` guard in `scripts/train.py` becomes unnecessary; remove it then.

## 2026-04-25 — `TrainingLogger` auto-refreshes loss / metric curves.

- Context: We want loss / SNR / PSNR / SSIM curves to update during long runs, not just after `close()`.
- Change: `TrainingLogger` gains `plot_interval: int = 5` (0 disables) and now keeps in-memory `_loss_history` / `_metric_history` rehydrated from any existing CSV on construction. `log_epoch` pushes the latest values, and every `plot_interval` epochs redraws `loss_curve.png` / `metrics_curve.png` via `plot_loss_curve` / `plot_metrics_curve`. `close()` always does one final refresh. `scripts/train.py` now passes `cfg["log"]["plot_interval"]` and drops the redundant tail `plot_*_curve` calls; `configs/default.yaml` ships `log.plot_interval: 5`. `test.ipynb` Section 10 asserts the suppression-then-trigger boundary at `plot_interval=2`, that `close()` re-draws with the latest data, and that resuming hydrates 3 prior rows from CSV before appending a 4th.
- Impact: Curves stay current during training without per-epoch overhead (matplotlib write is cheap but not free); resumed runs draw a continuous curve across restarts.
- Follow-up: Wire the same `plot_interval` into a future TensorBoard branch so both backends share one knob.

## 2026-04-25 — Add baseline UNet + SEG-Y interpolation training pipeline.

- Context: The repo had a generic scaffold but no real baseline model and no end-to-end interpolation experiment script for SEG-C3-45.
- Change: Added `model/unet.py` (`@register_model("unet")`) and imported it in `model/__init__.py`. Added `scripts/train_interpolation_unet.py` and `configs/interpolation_unet.yaml` to run: `read_regular_shots(traces_per_shot=201)` -> spherical-divergence compensation (`power=1.2`) -> uniform missing traces (`uniform_stride=2`, i.e. every 2 traces miss 1) -> `patchify_uniform` with `(patch_trace, patch_time)=(128,256)` and `overlap=0.5` -> UNet training with existing modular utilities (`build_loss`, `build_metrics`, `train_one_epoch`, `evaluate`, `TrainingLogger`, `visualize_random_sample`).
- Change: Filled core placeholders in `utils/train_utils.py`, `utils/losses.py`, and `utils/datasets.py` so modular training/factory flow is runnable (config loading, seeding, optimizer/scheduler builders, train/eval loops, checkpoint I/O, `npy/mat` dataset backends, MSE/L1/weighted-MSE losses).
- Change: Added `interpolation_debug.ipynb` for step-by-step visual debugging of each pipeline stage and a 2-epoch smoke training pass.
- Impact: The project now has a concrete, runnable interpolation baseline and a notebook to validate data/patch/mask/training behavior before long runs.

## 2026-04-26 - Align DnCNN with UNet style + register as `dncnn`
- Context: `model/dncnn.py` was a partial snippet (missing imports, debug print, no registry hook).
- Change: Rewrote the module to match `model/unet.py` layout (docstring, `from __future__ import annotations`, `@register_model("dncnn")`, typed `forward`). Same topology: first Conv+ReLU (bias), `depth-2` × Conv+BN+ReLU (`eps=0.0001`, `momentum=0.95`), final Conv to `out_channels`; output `x - net(x)`; orthogonal init. Params aligned with UNet names (`in_channels`, `out_channels`, `base_channels`, `depth`, `kernel_size`); enforce `in_channels == out_channels`. Added `from . import dncnn` in `model/__init__.py`. Cited Zhang et al. 2017 in `memory/research_first.md`.
- Impact: `build_model` / YAML can use `type: dncnn` with the same param naming style as `unet`.
- Follow-up: Add a config example under `configs/` when the first DnCNN experiment is scripted.

## 2026-04-26 - Paired SEG-Y denoise patch builder in `train_interpolation_unet.py`
- Context: Denoising needs separate noisy/clean SEG-Y volumes instead of synthetic masking from one file.
- Change: Added `_build_denoise_patch_pairs(cfg)` — reads `data.segy_pair` (`input_path`, `target_path`), optional `max_shots`, shared `normalize` + `patchify_uniform` (no spherical gain, no `mask_traces`). When `normalize_scope` is `global` and `clip_percentile` is unset, target stats drive `normalize(..., override_stats=...)` on the input.
- Impact: Callers can build `(noisy_patches, clean_patches)` for supervised denoise training; wire `main` / YAML when an experiment script is added.
- Follow-up: Add `configs/*_denoise.yaml` and a CLI or script entry that uses `_build_denoise_patch_pairs` + `_build_loaders` variant.

## 2026-04-26 - `train_denoise_unet.py` + `configs/denoise_unet.yaml` for paired SEG-Y
- Context: Denoise training should use `data.segy_pair` end-to-end instead of duplicating the interpolation script.
- Change: `scripts/train_denoise_unet.py` is denoise-only (removed interpolation `_build_patch_pairs` / unused preprocess imports); `_build_loaders` calls `_build_denoise_patch_pairs`; default `--config` is `configs/denoise_unet.yaml`; log/viz strings say Denoise. `configs/denoise_unet.yaml` defines `segy_pair.input_path` / `target_path` and a preprocess block without spherical/mask fields. Removed duplicate `_build_denoise_patch_pairs` from `train_interpolation_unet.py` (single definition in denoise script).
- Impact: Run `python scripts/train_denoise_unet.py --config configs/denoise_unet.yaml` after setting real noisy/clean paths.
- Follow-up: Replace placeholder `target_path` when a true clean volume is available.

## 2026-04-26 - Denoise normalize: noisy-derived stats for both volumes
- Context: User requested scaling from noisy statistics only, not from clean target statistics.
- Change: `_build_denoise_patch_pairs` now runs `normalize` on **input** first (optional `clip_percentile`), builds `override_stats` from that result, and normalizes **target** with those scalars. Non-global `normalize_scope` raises (override path is global-only).
- Impact: Clean patches share the noisy volume’s amplitude scale; target values may lie outside e.g. [-1, 1] for `max_abs` if signal is stronger than the noisy scale.
- Follow-up: None.
## 2026-04-29 - Denoise shell scripts: `torchrun` loop + per-model entry points

- Context: Repeated multi-GPU denoise runs should rotate `experiment.seed` without manual YAML edits or shell env vars from the CLI; each architecture should have a launcher that matches its Python script and default YAML.
- Change:
  - `scripts/train_denoise_dncnn.sh` — editable block at top (`CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `N`, `START_SEED`, optional `TORCHRUN_EXTRA`); loop generates a temp config from `configs/denoise_dncnn.yaml` with `sed` replacing only `experiment.seed` and `experiment.name` (`<base>_seed<seed>`); each iteration runs `torchrun --nproc_per_node=... scripts/train_denoise_dncnn.py --config <tmp>`. Removed single-GPU `CUDA_DEVICE_INDEX` / `device:` rewriting (DDP uses `cuda:local_rank`).
  - `scripts/train_denoise_unet.sh`, `scripts/train_denoise_res_unet.sh`, `scripts/train_denoise_atten_unet.sh` — same structure; paths set to `configs/denoise_{unet,res_unet,atten_unet}.yaml` and `scripts/train_denoise_{unet,res_unet,atten_unet}.py` respectively; header comments updated.
- Impact: Benchmark seeds are reproducible from script edits alone; `torchrun` invocations match the docstring examples on each training script.

## 2026-04-29 - Denoise metrics: six reconstruction metrics + `resolve_denoise_metrics`

- Context: Denoise runs should always log SNR, PSNR, SSIM, MAE, MSE, and RMSE into `TrainingLogger` and `metrics_history.csv` without duplicating lists in every script.
- Change: Added `resolve_denoise_metrics(cfg)` in `utils/train_utils.py` (defaults merged with YAML `metrics` per metric name; extra YAML-only metrics appended). All four `scripts/train_denoise_*.py` set `cfg["metrics"] = resolve_denoise_metrics(cfg)` before `build_metrics`. `configs/denoise_{unet,res_unet,dncnn,atten_unet}.yaml` document all six. `utils/metrics.py` module docstring maps registered names (lowercase) to acronyms.
- Impact: Logs/CSVs gain MAE / RMSE when they were omitted; YAML overrides (e.g. `data_range`, `window_size`) still apply.

## 2026-04-29 - DDP helpers in `train_utils`; denoise scripts use `DistributedSampler` + rank-0 eval

- Context: Train denoise entry points should support multi-process data parallelism without forks of the training loop; checkpoint state must omit `DistributedDataParallel` prefixes.
- Change: `utils/train_utils.py` adds `init_distributed`, `destroy_distributed`, `barrier_if_distributed`, `training_device`, `setup_experiment_dir_distributed`, `unwrap_ddp`, `maybe_wrap_ddp`, `sampler_set_epoch`. `train_one_epoch` all-reduces summed batch loss/count when multi-process (`WORLD_SIZE` > 1). `save_checkpoint` / `load_checkpoint` use ``unwrap_ddp`` for state dict keys. Exported from `utils/__init__.py`. `scripts/train_{denoise_unet,denoise_*}.py`: call `init_distributed()`, build train `DataLoader` with `DistributedSampler` when distributed, wrap model with ``DistributedDataParallel`` on CUDA, checkpoint/viz/logs only ``rank == 0``, train-set eval uses a shuffle-false loader on rank 0 (not the sharded train loader).
- Impact: Single-GPU path unchanged when `WORLD_SIZE` is unset or 1. Multi-GPU: `torchrun --nproc_per_node=K ...` with one process per GPU; `experiment.device` is ignored in favor of `LOCAL_RANK` when CUDA is used.
- Follow-up: Optional distributed evaluation (sharded val + metric all-reduce) if eval memory becomes limiting.

## 2026-04-29 - Denoise scripts: config path + experiment dir follow `model.type`
- Context: `train_denoise_{atten_unet,dncnn,res_unet}.py` defaulted to `configs/denoise_unet.yaml`, used UNet-centric log/viz strings, and copied YAML left `experiment.name: denoise_unet_base`, colliding across architectures.
- Change: `utils/train_utils.py` adds `default_config_relpath_for_train_script(script_file)` mapping `scripts/train_<name>.py` → `configs/<name>.yaml` and `apply_denoise_experiment_name_from_model(cfg)`, which rewrites the legacy placeholder `denoise_unet_base` to `denoise_<model.type>_base` when `model.type` is not `unet`. Three denoise scripts use both helpers and embed `cfg["model"]["type"]` in startup logs, viz titles, and completion messages; `configs/denoise_atten_unet.yaml` and `configs/denoise_res_unet.yaml` added; `configs/denoise_dncnn.yaml` renamed experiment to `denoise_dncnn_base`; `model/__init__.py` imports `res_unet` for registry side effects.

## 2026-04-29 - Attention U-Net in `model/atten_unet.py`
- Context: A dedicated attention variant was requested on top of the minimal U-Net without broad refactors; configs should select it via registry `type` like other variants.
- Change: `model/atten_unet.py` adds additive **attention gates** (Oktay et al. 2018–style gate: `ψ(σ(W_g g + W_x x))`) on encoder skip tensors before concatenation with upsampled decoder features; each stage uses `F_int = max(c // 2, 8)` for the bottleneck of the gate. Registers as `@register_model("atten_unet")` (distinct from `"unet"`). `model/__init__.py` imports `atten_unet` so decorators run with `model` package import per project convention.
- Impact: YAML can use `model.type: atten_unet` with the same `params` as the baseline U-Net; additive skip gating preserves tensor shapes and keeps the decoder contract unchanged.
- Follow-up: Add or duplicate a denoise YAML example that sets `type: atten_unet` once that experiment is scripted; optionally cite Attention U-Net in `memory/research_first.md`.

## 2026-04-30 - Rewrite `train_interpolation_unet.py` with DDP + update SEG-Y path
- Context: The interpolation script was single-GPU only; the user requested DDP support and a fixed demo data path.
- Change: `scripts/train_interpolation_unet.py` now mirrors the DDP pattern from `train_denoise_unet.py`: `init_distributed`, `DistributedSampler`, `maybe_wrap_ddp`, rank-0 evaluate/checkpoint/viz/log, `destroy_distributed`. `_build_loaders` gains `rank/world_size/distributed` kwargs and returns `(train_loader, test_loader, train_sampler, eval_train_loader)`. `configs/interpolation_unet.yaml` path updated to `/data/shared/SEGC3/SEG_45Shot_shots1-9.sgy`.
- Impact: Multi-GPU interpolation runs via `torchrun --nproc_per_node=N scripts/train_interpolation_unet.py`; single-GPU path unchanged.
- Follow-up: Add a matching `.sh` launcher if repeated seed sweeps are needed for interpolation.

## 2026-04-30 - Add inference pipeline: `scripts/inference_interpolation.py` + `utils/inference_utils.py`
- Context: The project had training scripts but no way to run full-shot reconstruction, report per-shot metrics, or visualize random shots in the original amplitude domain.
- Change:
  - `tools/preprocessing.py` adds `denormalize` (inverse of `normalize` using saved `stats`) and `inverse_spherical_divergence_correction` (divide by the gain). Both handle `"shot"` / `"trace"` / `"global"` scopes.
  - `utils/inference_utils.py` (new) provides four generic helpers:
    - `inference_on_shots` — `patchify_uniform` → batched model forward → `unpatchify_uniform`.
    - `compute_shot_metrics` — vectorized numpy per-shot `mse` / `rmse` / `mae` / `snr` / `psnr`; `ssim` uses the existing `SSIM` class shot-by-shot. Returns `(per_shot_dict, mean_dict)`.
    - `select_random_shots` — reproducible random subset without replacement.
    - `save_shot_visualizations` — loops over selected indices and calls `plot_sample` for each, saving one 4-panel PNG per shot (`input | prediction | target | residual`).
  - `scripts/inference_interpolation.py` (new) wires everything: load config + checkpoint → read SEG-Y → forward preprocess → mask → infer patches → inverse preprocess → per-shot metrics → save CSV/JSON → random-shot viz. Operates in the **original amplitude domain** (denormalize + undo spherical gain) per user request.
- Impact: End-to-end interpolation inference is now available. The helpers in `utils/inference_utils.py` are task-agnostic and can be reused for future `inference_denoise.py` etc.
- Follow-up: Consider adding a `--metrics` CLI override or YAML-driven `inference` block if users want to switch metric sets at inference time without editing the training config.

## 2026-04-30 - Expand data input to npy/mat; configurable paired preprocessing
- Context: The user needs to train on `.npy` and `.mat` volumes (not just SEG-Y), both in single-file interpolation mode and paired input/target denoise mode. In paired mode, preprocessing steps should be optional so users can skip normalization or spherical gain when data is already pre-conditioned.
- Change:
  - `tools/preprocessing.py::normalize` now accepts `clip_threshold` inside `override_stats`. When present, the symmetric clip is applied before normalization using the other override scalars. This lets paired denoise training share the exact same clip value between noisy input and clean target.
  - `tools/array_io.py` (new) provides `load_volume(data_cfg)` which dispatches by file extension to `read_npy_volume`, `read_mat_volume` (scipy), or `read_sgy_volume` (wrapper around `segy_read`). All return `(n_shots, n_traces, n_time)` float32.
  - `scripts/train_interpolation_unet.py::_build_patch_pairs` now auto-detects `data.segy`, `data.npy`, or `data.mat` and loads via `load_volume`. Preprocessing steps (spherical divergence, normalize, mask) are guarded by a `preprocess.skip` list so any step can be omitted.
  - `scripts/train_denoise_unet.py::_build_denoise_patch_pairs` now supports `data.segy_pair`, `data.npy_pair`, and `data.mat_pair`. Same `skip` mechanism applies; normalization still computes stats on the input volume and applies them to the target via `override_stats` (now including `clip_threshold` when clipping is used).
  - `configs/interpolation_unet.yaml` updated with commented-out `npy` / `mat` blocks and a `skip` example.
  - `tools/README.md` documents `array_io.py`.
- Impact: Training scripts are no longer tied to SEG-Y. Users can plug in numpy or MAT files by changing one YAML block. Paired-mode preprocessing is fully configurable via `preprocess.skip: ["normalize", "spherical_divergence_correction"]` etc.
- Follow-up: Add a matching `configs/denoise_unet.yaml` example with `npy_pair` or `mat_pair` once a real non-SEG-Y denoise dataset is available.

## 2026-04-30 - Bash launchers for interpolation train / inference
- Context: The user requested `.sh` wrappers in the same style as `train_denoise_atten_unet.sh` to run multi-seed interpolation training loops and single-checkpoint inference without manual CLI typing.
- Change:
  - `scripts/train_interpolation_unet.sh` (new) loops `N` times over `train_interpolation_unet.py` via `torchrun`; each iteration patches `seed` and `experiment.name` in a temp copy of `configs/interpolation_unet.yaml` so output directories never collide. Configuration block at the top sets `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `N`, `START_SEED`.
  - `scripts/inference_interpolation.sh` (new) is a single-run wrapper around `inference_interpolation.py`; top-of-file variables set `CONFIG`, `CHECKPOINT`, `OUTPUT_DIR`, `N_VIZ_SHOTS`, `SEED`, `DEVICE`. Empty strings fall back to config defaults.
  - `scripts/inference_interpolation.py` updated to use `load_volume` (supports `.npy`/`.mat`/`.sgy`) instead of hard-coding `read_regular_shots`; added `preprocess.skip` support and inverse-preprocessing path (`denormalize` + `inverse_spherical_divergence_correction`) so metrics are reported in the **original amplitude domain**.
- Impact: Training sweeps and inference are now one-command operations. The inference script is format-agnostic and supports the same `skip` semantics as training.
- Follow-up: If inference needs to run on *every* checkpoint in a directory automatically, extend the `.sh` with a `for ckpt in results/<name>/checkpoints/*.pt` loop.

## 2026-04-30 - Consolidate denoise scripts: keep only `train_denoise_res_unet.py`
- Context: The project accumulated four nearly identical denoise training scripts (`train_denoise_{unet,atten_unet,dncnn,res_unet}.py`) plus their configs and `.sh` launchers. As a template repository this creates maintenance overhead and confusion about which file to copy.
- Change:
  - **Deleted** `scripts/train_denoise_atten_unet.py`, `train_denoise_dncnn.py`, `train_denoise_unet.py`, and their `.sh` launchers (`train_denoise_atten_unet.sh`, `train_denoise_dncnn.sh`).
  - **Deleted** `configs/denoise_atten_unet.yaml`, `configs/denoise_dncnn.yaml`, `configs/denoise_unet.yaml`.
  - **Retained** `scripts/train_denoise_res_unet.py` and `configs/denoise_res_unet.yaml` as the single denoise template.
  - **Updated** `train_denoise_res_unet.py` to the modern paired-data pattern:
    - Replaced `read_regular_shots` with `load_volume`, supporting `segy_pair`, `npy_pair`, and `mat_pair`.
    - Added `preprocess.skip` mechanism (supports skipping `spherical_divergence_correction` and `normalize`).
    - Fixed `override_stats` to pass the full `in_stats` dict (including `clip_threshold` when percentile clipping is used) rather than manually extracting only `mode_keys` scalars.
  - **Updated** `configs/denoise_res_unet.yaml` with `dt`/`t0`/`spherical_power` fields, commented `npy_pair`/`mat_pair` examples, and a `skip` comment.
- Impact: Only one denoise training entry point remains; it uses the same `load_volume` + `skip` infrastructure as `train_paired_unet.py` and `train_interpolation_unet.py`. Users who want a different denoise architecture simply change `model.type` in the retained YAML (registry handles the rest).
- Follow-up: If the user later needs the deleted architectures back, they can be restored from git history or recreated by changing `model.type` in `configs/denoise_res_unet.yaml`.

## 2026-04-30 - Inference metric domain switched to normalized; add `.npy` output + independent `batch_size`

- Context: The initial inference pipeline computed metrics after inverse preprocessing (original amplitude domain). The user requested metrics in the **normalized domain** for consistency with training, and added two convenience features: saving predictions/inputs/targets as `.npy` files, and overriding the inference batch size independently from training.
- Change:
  - `scripts/inference_interpolation.py`: moved `compute_shot_metrics` call to operate on `pred_norm` vs `shots_norm` **before** `denormalize` + `inverse_spherical_divergence_correction`. Inverse preprocessing now runs only for visualization and optional `.npy` export.
  - Added `--save-npy` CLI flag and `inference.save_npy` YAML field (default `false`). When enabled, saves `pred_shots.npy`, `target_shots.npy`, and `input_shots.npy` under `<output_dir>/npy/` in the original amplitude domain.
  - Added `inference.batch_size` YAML field and `--batch-size` CLI override; falls back to `data.loader.batch_size` when omitted. This lets users reduce GPU memory during inference without touching the training config.
  - Updated `configs/interpolation_unet.yaml` inference block with commented `save_npy` and `batch_size` examples.
  - Updated `使用说明.md` FAQ and consistency table to state that metrics are computed in the normalized domain.
- Impact: Metric values from inference now align with the normalized-domain losses seen during training. Users can run large-volume inference on smaller GPUs by lowering `inference.batch_size`, and archive raw arrays via `--save-npy`.
- Follow-up: If cross-dataset inference requires strictly identical normalization scales, save training stats into checkpoints and load them at inference time rather than recomputing on the test volume.

## 2026-05-02 — Systematic code review: preprocessing, metrics, and training-script fixes

- Context: User-requested systematic review of preprocessing and metrics against geophysical literature. 16 issues identified; issues 1–4, 7–10, 14–16 fixed; 5–6, 11–13 acknowledged as no-fix.
- Change:
  - `tools/preprocessing.py`:
    - `denormalize`: fixed `per="trace"` reshape bug (was `(-1, 1, 1)`, now `(n_shots, n_traces, 1)`).
    - `spherical_divergence_correction` / `inverse_spherical_divergence_correction`: default `power` changed from `2.0` to `1.0` (3D amplitude compensation per Yilmaz 2001); replaced `if t0 == 0.0: t[0] = dt` with `t = np.maximum(t, dt)` to prevent NaN from negative fractional powers.
    - `normalize`: `override_stats` no longer forces `per="global"`; supports `shot` and `trace` scopes with proper broadcasting. `clip_threshold` inside `override_stats` is reshaped to match `per` axes.
  - `tools/array_io.py`: `read_mat_volume` raises `ValueError` when multiple non-internal variables exist and no `key` is specified.
  - `utils/metrics.py`:
    - Added numpy core functions (`_mse_numpy`, `_mse_per_sample_numpy`, `_mae_numpy`, `_mae_per_sample_numpy`, `_rmse_per_sample_numpy`, `_snr_numpy`, `_snr_per_sample_numpy`, `_psnr_numpy`, `_psnr_per_sample_numpy`). Torch metric classes (`MSE`, `RMSE`, `MAE`, `SNR`, `PSNR`) are now thin wrappers over the numpy implementations.
    - PSNR docstring updated to clarify `data_range` means **peak amplitude** (`max |x|`), not peak-to-peak range.
    - SNR: zero-noise cases handled explicitly — `noise == 0, signal > 0` returns `+inf`; `noise == 0, signal == 0` returns `nan`.
  - `utils/inference_utils.py`:
    - `compute_shot_metrics` now calls the shared numpy metric functions instead of inline formulas.
    - `inference_on_shots` saves/restores `model.training` state.
    - `compute_shot_metrics` signature split into `psnr_peak` (PSNR) and `ssim_data_range` (SSIM) to reflect different semantics.
  - `utils/train_utils.py`: extracted generic `build_loaders(cfg, build_patch_pairs_fn, rank, world_size, distributed)` to eliminate ~90-line duplication across three training scripts.
  - `scripts/train_interpolation_unet.py`, `train_paired_unet.py`, `train_denoise_res_unet.py`: replaced local `_build_loaders` with imported `build_loaders`.
  - `scripts/inference_interpolation.py`: infers separate `psnr_peak` / `ssim_data_range` defaults per normalization mode; `mean_std` branch auto-infers from actual data with zero-signal fallback.
  - `configs/interpolation_unet.yaml`, `configs/denoise_res_unet.yaml`: `psnr.data_range` changed from `2.0` to `1.0`.
- Impact: Preprocessing is more robust (no NaN, correct inverse operations, flexible scope). Metrics have a single source of truth (numpy core) shared between training and inference. Training scripts are DRY-er. Inference metrics align with training-domain normalization.
- Follow-up: Save training-set normalization stats into checkpoints so inference can reuse them exactly across different datasets.

## 2026-05-03 - Arbitrary output paths + script subfolder support

- Context: `output_dir` was resolved relative to the current working directory, so running a script from `/tmp` produced `/tmp/results`. Scripts also hard-coded `_REPO_ROOT = Path(__file__).resolve().parents[1]`, breaking `sys.path` if a script were moved to a subfolder like `scripts/interpolation/train.py`.
- Change:
  - `utils/train_utils.py` adds `resolve_repo_root(script_file)` — walks upward from the script until it finds a directory containing both `model/` and `utils/`, then returns that path. Falls back to `parents[1]` after 10 levels.
  - `setup_experiment_dir` gains an optional `base_dir` parameter. When `output_dir` is relative and `base_dir` is provided, the final path is `base_dir / output_dir / name`; when `output_dir` is absolute it is used as-is.
  - `setup_experiment_dir_distributed` forwards `base_dir`.
  - `default_config_relpath_for_train_script` now uses `resolve_repo_root(script_file)` instead of `script_path.parent.parent`.
  - All four Python scripts (`train_interpolation_unet.py`, `train_denoise_res_unet.py`, `train_paired_unet.py`, `inference_interpolation.py`) import and use `resolve_repo_root(__file__)` for `sys.path`; training scripts pass `base_dir=_REPO_ROOT` to `setup_experiment_dir_distributed`.
  - `configs/default.yaml` and `configs/interpolation_unet.yaml` update the `output_dir` comment to note that absolute paths are supported.
  - `utils/__init__.py` exports `resolve_repo_root`.
- Impact: Scripts can now be reorganized into task-specific subfolders (e.g., `scripts/interpolation/`) without breaking imports. `output_dir: /data/experiments` works as expected regardless of where the script is launched.
- Follow-up: Move task-specific scripts into subfolders if the user wants a cleaner `scripts/` layout.

## 2026-05-03 - Denoise metrics computed on denoised signal; drop spherical-divergence
- Context: The denoise pipeline learns to predict additive noise (target is the noise label), so SNR / PSNR / SSIM on `(pred, target)` measure noise-vs-noise rather than reconstruction quality. Spherical-divergence amplitude compensation was also unwanted for this experiment.
- Change:
  - `utils/train_utils.py` `evaluate(...)` gains kw-only `metrics_on_denoised_signal: bool = False`; when `True`, metrics receive `(x - pred, x - target)` instead of `(pred, target)`. Loss is unchanged (still `loss_fn(pred, y)`).
  - `scripts/coherent_noise_attenuation/train_denoise_res_unet.py`: removed `spherical_divergence_correction` from `_build_denoise_patch_pairs`; both `evaluate(...)` calls pass `metrics_on_denoised_signal=True`.
  - `configs/coherent_noise_attenuation/denoise_res_unet.yaml`: removed `spherical_power`; rewrote header and `metrics` comments to describe the noise-prediction setup.
- Impact: SNR / PSNR / SSIM now reflect signal-domain reconstruction in the normalized space; MAE / MSE / RMSE are numerically unchanged. Loss curves are unaffected. `(x - pred)` and `(x - target)` use the same noisy-input-derived `max_abs` scale, so amplitudes are comparable. Visualization still shows noisy input, predicted noise, and label noise.
- Follow-up: Tune `psnr.data_range` / `ssim.data_range` if residual amplitudes in normalized space exceed `[-1, 1]`.

## 2026-05-03 - Per-subtask layout for configs / scripts / model
- Context: Repo reorganized around four task subfolders (`coherent_noise_attenuation/`, `interpolation/`, `random_noise_suppression/`, `first-break_picking/`). The top-level `model/{atten_unet,dncnn,res_unet,unet}.py` were deleted in favor of one copy per subtask under `model/<task>/`, breaking `from model import build_model`.
- Change:
  - `utils/train_utils.py` `default_config_relpath_for_train_script`: now mirrors the script's subtask subdirs (`scripts/<task>/train_<stem>.py` → `configs/<task>/<stem>.yaml`); flat layout still works for back-compat.
  - `model/__init__.py`: no longer auto-imports any task model; exposes registry primitives + `placeholder` only. Each `model/<task>/__init__.py` (new) imports its 4 model files (triggering `@register_model`) and re-exports `build_model`. Each task model file changed `from .registry` → `from ..registry`.
  - Entry scripts: `from model import build_model` → `from model.<task> import build_model` in `train_denoise_res_unet.py`, `train_denoise_unet.py`, `train_interpolation_unet.py`, `train_paired_unet.py`, `inference_interpolation.py`.
  - Shell wrappers: `REPO_ROOT="${SCRIPT_DIR}/../.."`; `BASE_CONFIG` / `PY_SCRIPT` updated to subtask paths.
  - Hard-coded inference default `--config` updated to `configs/interpolation/interpolation_unet.yaml`. Module docstring example commands updated.
  - `configs/coherent_noise_attenuation/denoise_unet.yaml`: `experiment.name` corrected from the copy-pasted `denoise_res_unet_base0502` to `denoise_unet_base0502` (otherwise it would have shared the output directory with the res_unet experiment).
- Impact: Each task is self-contained; new task models go under its `model/<task>/`. Two limitations to note: (1) loading two subtask packages in the same Python process raises `KeyError` on shared registry keys (`unet`, `res_unet`, …) — one subtask per run. (2) `model.first-break_picking` cannot be imported with dotted syntax due to the hyphen; use `importlib.import_module` or rename the folder.
- Follow-up: Rename `first-break_picking/` → `first_break_picking/` once a script consumer appears; consider deduplicating the byte-identical `train_denoise_unet.py` / `train_denoise_res_unet.py`.

## 2026-05-03 - Fix sys.path bootstrap order in entry scripts
- Context: `torchrun <abs/path/script.py>` makes `sys.path[0]` the script's own directory only. All five entry scripts ran `from utils import resolve_repo_root` *before* inserting the repo root into `sys.path`, so the first import crashed with `ModuleNotFoundError: No module named 'utils'`. The bug pre-dated the subtask reorg but was masked by ad-hoc `PYTHONPATH` / CWD setups.
- Change: `train_denoise_unet.py`, `train_denoise_res_unet.py`, `train_interpolation_unet.py`, `train_paired_unet.py`, `inference_interpolation.py` replaced the leading `from utils import resolve_repo_root` block with an inline `pathlib`-only walk: from `Path(__file__).resolve().parents`, pick the first directory containing both `model/` and `utils/`, insert it into `sys.path`, then import. `_REPO_ROOT` is still defined for `setup_experiment_dir_distributed(base_dir=_REPO_ROOT)`.
- Impact: Entry scripts launch correctly via `torchrun` (or plain `python <abs path>`) from any working directory and any subfolder depth without `PYTHONPATH` setup.
- Follow-up: None.

## 2026-05-06 - Unified vmin/vmax for all visualizations
- Context: Residual panels in `plot_sample` used an independent per-panel adaptive color scale (`_symmetric_clip` per array). Because residuals are typically small, they were visually exaggerated and could not be directly compared against input / prediction / target amplitudes.
- Change:
  - `utils/visualization.py` `plot_sample`: added `vmin`, `vmax`, `share_scale` parameters. Default is `share_scale=True`. When enabled (and no explicit `vmin`/`vmax` provided), a single symmetric scale is computed from the maximum of `_symmetric_clip` over input, prediction, and target, and applied to all panels including the residual.
  - `utils/visualization.py` `visualize_random_sample`: forwards `vmin`/`vmax`/`share_scale` to `plot_sample`.
  - `utils/inference_utils.py` `save_shot_visualizations`: forwards `vmin`/`vmax`/`share_scale` to `plot_sample`.
  - `scripts/interpolation/inference_interpolation.py`: computes a run-level symmetric scale (`vmax = np.quantile(np.abs(concatenated volume), 0.995)`) and passes explicit `vmin=-vmax, vmax=vmax` to `save_shot_visualizations`, so every shot figure in the same inference run uses exactly the same color scale.
- Impact: All training and inference visualizations now share a consistent color scale by default. Residual amplitudes are directly comparable to input/prediction/target. Backward compatibility is preserved: callers can still request per-panel adaptive scaling by passing `share_scale=False`.
- Follow-up: If other inference scripts are added later, replicate the run-level `vmax` computation pattern.

## 2026-05-06 - Shot-level (FFID) train/val/test splitting
- Context: The existing `build_loaders` performed a patch-level random split, which could place patches from the same shot gather into both train and test sets, causing data leakage. The user requested splitting by unique FFID (shot number) in sequential order before patchifying, with configurable counts per split.
- Change:
  - `utils/train_utils.py`: added `build_shot_split_loaders(cfg, preprocess_fn, patchify_fn, ...)` which:
    1. Calls `preprocess_fn` to obtain `(input_shots, target_shots, per_shot_ffid)`.
    2. Derives `unique_ffids = np.unique(per_shot_ffid)` and slices them sequentially into train/val/test subsets using `cfg["data"]["shot_split"]` counts.
    3. Masks shot volumes by FFID membership, patchifies each subset independently, and builds `DataLoader`s with `DistributedSampler` when distributed.
    4. Optionally saves the unpatchified test-set shots to `test_set_dir` for later inference.
  - Extracted `_make_dataloader` helper in `utils/train_utils.py` to avoid duplication between `build_loaders` and `build_shot_split_loaders`.
  - `utils/__init__.py`: exported `build_shot_split_loaders`.
  - `scripts/interpolation/train_interpolation_unet.py`: split `_build_patch_pairs` into `_preprocess_shots` (load volume → spherical divergence → normalize → mask_traces → extract FFID from SEG-Y headers) and `_patchify_pairs`. `main()` branches on `"shot_split" in cfg.get("data", {})` to call `build_shot_split_loaders` or the legacy `build_loaders`. Logger keys changed from `test_` prefix to `val_` prefix.
  - `scripts/interpolation/train_paired_unet.py`: same structural split and `shot_split` branch; `_preprocess_shots` supports paired input/target volumes (`segy_pair` / `npy_pair` / `mat_pair`).
  - `scripts/coherent_noise_attenuation/train_denoise_unet.py` and `train_denoise_res_unet.py`: same split pattern; `evaluate()` retains `metrics_on_denoised_signal=True`.
  - `configs/interpolation/interpolation_unet.yaml`: added commented `shot_split` example block.
- Impact: When `data.shot_split` is present, train/val/test boundaries are strictly at the shot level (by FFID), eliminating leakage from overlapping patches. The sequential FFID ordering (1-7 train, 8 val, 9 test, etc.) is deterministic and reproducible. Existing configs without `shot_split` continue to use the patch-level random split for backward compatibility.
- Follow-up: Save training-set FFID list into checkpoints so that resumed runs or downstream inference can verify split consistency.

## 2026-05-06 - Save best-validation checkpoint
- Context: Training scripts only saved checkpoints at fixed `ckpt_interval` epochs. The user wanted to retain the model parameters that achieved the lowest validation loss, regardless of the interval schedule.
- Change:
  - `utils/train_utils.py`: added `maybe_save_best_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, best_val_loss, extras, logger)` which compares `val_loss` against `best_val_loss`, saves a checkpoint to `path` when improved, logs the event, and returns the updated best value.
  - `utils/__init__.py`: exported `maybe_save_best_checkpoint`.
  - All four training scripts (`train_interpolation_unet.py`, `train_paired_unet.py`, `train_denoise_unet.py`, `train_denoise_res_unet.py`):
    - Initialize `best_val_loss = float("inf")` before the epoch loop.
    - After each validation evaluation, call `maybe_save_best_checkpoint(..., path=exp_dir / "checkpoints" / "best.pt", ...)` so the best model is overwritten in-place when validation loss decreases.
- Impact: A `best.pt` checkpoint is always available after training, containing the model state, optimizer, scheduler, and epoch that yielded the lowest validation loss. The periodic `epoch_*.pt` checkpoints are unaffected.
- Follow-up: Allow YAML-configurable `best_metric` (e.g. maximize SNR instead of minimizing loss) if different tasks need different criteria.

## 2026-05-08 — Denoise shell scripts: noise-level × seed nested loop + per-model configs

- Context: The user has multiple paired noisy/noise SEG-Y volumes at different noise intensity levels (`noisy_1.0.sgy` / `noise_1.0.sgy`, `noisy_3.0.sgy` / `noise_3.0.sgy`, etc.) under `/data/shared/benchmark/ground_roll/`. The existing `.sh` launchers only looped over seeds against a single hard-coded noise level. The DnCNN and AttenUNet launchers also had stale paths pointing to `denoise_res_unet` config/script. Additionally, `torchrun` default port 29500 caused `EADDRINUSE` when launching multiple scripts concurrently.

- Change:
  - **Shell script rework** — all four `train_denoise_{unet,res_unet,dncnn,atten_unet}.sh` now use a nested loop: outer loop over `NOISE_LEVELS` array, inner loop over `N_SEEDS` seeds. Experiment name becomes `<base>_level<level>_seed<seed>`, guaranteeing output dirs never collide. Each run increments `MASTER_PORT` (`--master_port=29500 + run_idx - 1`) to avoid port conflicts between consecutive and concurrent runs.
  - **Shell script path fixes** — `train_denoise_dncnn.sh` and `train_denoise_atten_unet.sh` corrected from stale `denoise_res_unet.yaml` / `train_denoise_res_unet.py` to their own config and Python script.
  - **Python script docstrings** — `train_denoise_dncnn.py` and `train_denoise_atten_unet.py` docstrings updated to reference their own script names and config paths.
  - **New configs** — `configs/coherent_noise_attenuation/denoise_dncnn.yaml` (model: dncnn, batch_size: 8, depth: 17, base_channels: 64) and `configs/coherent_noise_attenuation/denoise_atten_unet.yaml` (model: atten_unet, batch_size: 196, depth: 4, base_channels: 32). Both follow the same paired segy_pair schema as the existing UNet configs.
  - **Config cleanup** — removed `dt`/`t0` from preprocess blocks in all four denoise configs (spherical-divergence is already excluded from the denoise pipeline); added `MASTER_PORT` and `NOISE_LEVELS` comments in the shell scripts.

- Impact: Each script can now run a full grid of `len(NOISE_LEVELS) × N_SEEDS` experiments in one command. Output directories follow the pattern `<output_dir>/<name>_level<level>_seed<seed>/`, fully isolated per combination. Port conflicts are eliminated. Users can edit the four-line config block at the top of each `.sh` to select noise levels, seed count, starting seed, GPUs, and base port — no command-line arguments needed.

- Follow-up: If additional noise levels are added to the data directory, simply append them to the `NOISE_LEVELS` array. For running multiple scripts in parallel, assign each script a different `MASTER_PORT` base (e.g. 29500, 29520, 29540, 29560) to prevent port-range overlap.

## 2026-05-10 — HF model upload, benchmark results, model card

- Context: After training, the best.pt checkpoints need to be uploaded to Hugging Face for sharing. The benchmark documentation required actual evaluation results instead of placeholders. A corresponding Hugging Face model card was needed.
- Change:
  - `tools/upload_to_hf.py` (new): scans experiment directories (`denoise_{model}_base*_level*_seed*`), uploads `checkpoints/best.pt` + `config.yaml` per experiment to a single HF repo under `models/{arch}/level{level}_seed{seed}/`. Generates and uploads a comprehensive model card (README.md) describing task, dataset, architectures, training details, usage, and results.
  - `docs/benchmark_coherent_noise_attenuation.md`: replaced placeholder results table with full evaluation data across five noise levels (1.0–9.0) for all four architectures, including SNR summary and per-level breakdowns (PSNR, SSIM, MAE, MSE, RMSE) with mean±std over 3 seeds. Added Key Observations section.
  - `scripts/coherent_noise_attenuation/batch_evaluate.py`: added `num_params_m` to evaluation output for parameter count tracking.
- Impact: Benchmark results are now documented. Trained models can be uploaded to Hugging Face via a single CLI command. The HF repo includes a complete model card with task description, usage examples, and evaluation metrics.
- Follow-up: Add a standalone `inference_denoise.py` for per-shot visualization of individual checkpoints.

## 2026-05-10 — Benchmark doc updates + batch evaluation script

- Context: The coherent noise attenuation benchmark page had a generic "supervised deep learning" description that didn't name the architectures or explain the data source. There was also no automated way to evaluate all trained models on the held-out test sets and aggregate results across seeds.

- Change:
  - `docs/benchmark_coherent_noise_attenuation.md`: task overview now lists all four architectures (UNet, ResUNet, DnCNN, Attention UNet) and clarifies that the 9-shot SEG-C3 dataset is synthetic — generated via forward modeling on the official SEG C3 velocity model, with reflection signals from the acoustic wave equation and ground-roll noise from the elastic wave equation. Fixed geometry table: `n_time` corrected from 1501 to 625.
  - `scripts/coherent_noise_attenuation/batch_evaluate.py` (new): end-to-end batch evaluation script. Discovers experiment directories by name pattern (`denoise_{model}_base{date}_level{level}_seed{seed}`), loads `checkpoints/best.pt` and `test_set/`, runs `inference_on_shots`, flattens data to 2D before metric computation, computes SNR/PSNR/SSIM/MAE/MSE/RMSE both before and after denoising, aggregates mean±std across seeds per (level, model), and outputs one Excel sheet per noise level. Rows = Raw/DnCNN/UNet/ResUNet/Attention UNet, columns = metrics, cells = `mean±std`. MAE/MSE/RMSE means at 6 decimal places, all other values at 2 decimal places; raw row shows `value±0.00`.

- Impact: Benchmark task description is now specific and technically accurate. Batch evaluation is a single command: `python scripts/coherent_noise_attenuation/batch_evaluate.py --root_dir <results_dir> --output <output.xlsx>`.

- Follow-up: Add a standalone inference script (`inference_denoise.py`) if per-shot visualization or original-amplitude-domain metrics are needed for individual checkpoints.

## 2026-06-29 - Split coherent-noise attenuation task references

- Context: The former `coherent_noise_attenuation` task was split into `ground_roll_attenuation` and `multiples_attenuation`; copied configs/scripts/models still contained stale imports, paths, comments, and documentation references to the old task name.
- Change:
  - Updated ground-roll scripts, shell launchers, model imports, README examples, and batch-evaluation defaults to use `ground_roll_attenuation`.
  - Updated multiples scripts, shell launchers, model imports, configs, README examples, and batch-evaluation defaults to use `multiples_attenuation`; multiples model package now imports only the model files that exist in that subtask.
  - Renamed the benchmark document to `docs/benchmark_ground_roll_attenuation.md`; updated top-level README, Chinese usage guide, and HF model upload/download defaults away from the old coherent-noise task name.
- Impact: New task directories are self-contained and no regular source/docs path references the deleted `coherent_noise_attenuation` package. `memory/updates.md` keeps older entries unchanged as historical records.
- Follow-up: When multiples SEG-Y data is generated, verify the configured `/data/shared/benchmark/multiples` paths against the actual data root.

## 2026-07-01 - Add DFB-CNN model for ground-roll attenuation

- Context: Added the Dual-Filter-Bank CNN (Zhang & van der Baan, 2022, IEEE TGRS) as a new ground-roll attenuation model. The paper proposes two DnCNN-style subnetworks with different kernel sizes (5×5 for low-freq, 3×3 for high-freq) processing data in the radial-trace (RT) domain after a Gaussian low-pass frequency split.
- Change:
  - `model/ground_roll_attenuation/dfb_cnn.py` (new): Implements DFB-CNN with:
    - `_GaussianLowPass` — spatial-domain Gaussian filter for frequency separation.
    - `_RadialTraceTransform` — differentiable forward/inverse RT transform via `F.grid_sample`.
    - `_DnCNNBlock` — DnCNN-style sub-network with residual output (shared helper).
    - `DFBCNN` (`@register_model("dfb_cnn")`) — full pipeline: low-pass split → RT → dual CNN → IRT → combine. Noise-predicting (residual learning), compatible with existing `metrics_on_denoised_signal=True`.
  - `model/ground_roll_attenuation/__init__.py`: added `from . import dfb_cnn` for registry side-effects.
  - `configs/ground_roll_attenuation/denoise_dfb_cnn.yaml` (new): follows existing paired-SEG-Y denoise config pattern; model params match paper defaults (CNN1: k=5, depth=9, 100 feat; CNN2: k=3, depth=5, 64 feat; v_max=3200; batch_size=32, lr=1e-3→1e-5, 50 epochs).
  - `scripts/ground_roll_attenuation/train_denoise_dfb_cnn.py` (new): standard denoise training entry point (identical structure to `train_denoise_dncnn.py`), auto-detects `data.*_pair`, supports shot-level FFID splitting and DDP.
  - `model/ground_roll_attenuation/reproduction/DFB-CNN_reproduction.md` (new): algorithm reproduction notes covering pipeline, RT transform theory, training strategy, and reproduction adaptations.
- Impact: DFB-CNN is available as a drop-in model (`type: dfb_cnn`) usable with the existing denoise training pipeline without changes to `utils/`, `tools/`, or other scripts. The RT transform and frequency split are implemented inside the model's forward pass, preserving the existing patch-based data flow.
- Follow-up: If RT transform on full shots (before patchify) is preferred to match the paper exactly, move the RT+frequency-split logic into `_patchify_pairs`. The current in-model implementation is simpler and avoids changing the shared training infrastructure.
- Reference: Zhang & van der Baan, "Ground-Roll Attenuation Using a Dual-Filter-Bank Convolutional Neural Network," IEEE TGRS, vol. 60, 2022, 5907511.

## 2026-07-01 - Add Physics-Constrained DNN model for ground-roll attenuation

- Context: Added the Physics-Constrained Deep Neural Network (Liu et al., 2025, IEEE TGRS) as a new ground-roll attenuation model. The paper proposes MPIC (Multi-modality Physical Information Constraint) Blocks with parallel spatial, frequency, and Hilbert-domain branches plus an SDPAF pre-activation that boosts weak signals. Physical constraints (frequency mask suppressing low-freq noise) guide the optimizer toward physically plausible solutions, reducing underfitting under small-sample training.
- Change:
  - `model/ground_roll_attenuation/physics_dnn.py` (new): Implements Physics-Constrained DNN with:
    - `SDPAF` — smooth ``x + beta * tanh(gamma * x)`` approximation of the paper's piecewise Seismic Data Preprocessing Activation Function, amplifying weak signals while preserving strong ones.
    - `_FrequencyMask` — learnable sigmoid frequency mask computed dynamically per input shape; suppresses low frequencies (ground-roll) while passing high frequencies (reflections).
    - `_SpatialModality` / `_FrequencyModality` / `_HilbertModality` — three parallel modality branches per MPIC Block.
    - `_MPICBlock` — multi-modality block with spatial conv, frequency (rFFT → mask → irFFT), Hilbert (analytic signal) branches, residual connection, BN, and ReLU.
    - `PhysicsConstrainedDNN` (`@register_model("physics_dnn")`) — full pipeline: optional SDPAF → FC_in (1×1 Conv + coord grid) → MPIC Block × N → FC_out (1×1 Conv). Noise-predicting (residual learning), compatible with existing `metrics_on_denoised_signal=True`. Fully adaptive to arbitrary patch sizes via lazy MPIC Block initialisation and dynamic frequency mask computation.
  - `model/ground_roll_attenuation/__init__.py`: added `from . import physics_dnn` for registry side-effects.
  - `configs/ground_roll_attenuation/denoise_physics_dnn.yaml` (new): follows existing paired-SEG-Y denoise config pattern; model params: n_channels=32, n_mpic_blocks=4, all three modalities enabled, SDPAF enabled, lr=1e-3→1e-5 (cosine), 300 epochs.
  - `scripts/ground_roll_attenuation/train_denoise_physics_dnn.py` (new): standard denoise training entry point (identical structure to `train_denoise_dncnn.py`), auto-detects `data.*_pair`, supports shot-level FFID splitting and DDP.
  - `model/ground_roll_attenuation/reproduction/PhysicsConstrainedDNN_reproduction.md` (new): algorithm reproduction notes covering SDPAF, MPIC Block design, three modality types, loss function, and reproduction adaptations.
- Impact: Physics-Constrained DNN is available as `type: physics_dnn`, usable with the existing denoise training pipeline without changes to `utils/`, `tools/`, or other scripts. The physical constraints (frequency mask, Hilbert scaling) and SDPAF are implemented inside the model's forward pass. The model is lightweight (~37K params for default config, vs ~1.87M for DFB-CNN).
- Follow-up: If full-shot SDPAF preprocessing is preferred (instead of per-patch in-model), move SDPAF into `_preprocess_shots`. The current in-model implementation keeps the preprocessing pipeline unchanged.
- Reference: Liu et al., "Near-Surface-Related Nonstationary Coherent Noise Suppression Using a Physically Constrained Deep Neural Network," IEEE TGRS, vol. 63, 2025, 5903010.

## 2026-07-17 - DDPM support in ground-roll batch evaluation

- Context: `scripts/ground_roll_attenuation/batch_evaluate.py` failed on `ddpm` checkpoints with `TypeError: DDPMUNet.forward() missing 1 required positional argument: 't'` — the generic `inference_on_shots` calls `model(batch)`, but the conditional DDPM requires the reverse-diffusion sampling loop (`DDPMNoiseScheduler.sample_full`).
- Change:
  - `utils/inference_utils.py`: `inference_on_shots` gained an optional `forward_fn` argument that replaces the default `model(batch)` call. Backward compatible with all existing callers.
  - `scripts/ground_roll_attenuation/batch_evaluate.py`: `load_model_from_checkpoint` now also returns the training config from `ckpt["extras"]["config"]` (avoids re-loading the checkpoint); `evaluate_one` detects `model.type == "ddpm_unet"`, rebuilds `DDPMNoiseScheduler` from the checkpoint's `diffusion` config, and runs DDIM sampling via `forward_fn` using the training-time validation settings (`train.eval_sample_steps` default 20, `eval_use_ddim`, `eval_ddim_eta`). The callable returns `input - x_0_pred` as "predicted noise" so the downstream `denoised = input - pred_noise` equals the sampled `x_0_pred`, matching `_evaluate_ddpm` in `train_denoise_ddpm.py`. `torch.manual_seed(0)` fixes the initial noise draw for reproducibility.
  - Also renamed model key `physics_unet` -> `physics` in `MODEL_DISPLAY` / `MODEL_ROW_ORDER` to match on-disk experiment directory names (`denoise_physics_base0526_*`); display name stays "Physics CNN".
- Impact: `--models ddpm` and `--models physics` now work in ground-roll batch evaluation. DDPM evaluation is ~20x slower than single-forward models (20 UNet forwards per batch). The multiples pipeline and other `inference_on_shots` callers are unaffected.

## 2026-07-17 - Fix physics_unet and enhanced_atten_unet in ground-roll batch evaluation

- Context: two model families produced broken batch-evaluation results.
  1. `physics_unet` (`PhysicsSeparationNet`) crashed in `inference_on_shots`: its `forward()` returns the `(X, Y, Y_recover)` tuple used by the physics constraints, so `out.cpu()` raised `AttributeError`. The dedicated `denoise()` method (clean-signal estimate X, same as training-time validation in the reference `train_denoise_physics.py`) must be used instead.
  2. `enhanced_atten_unet` base0511 checkpoints were trained before commit 6de141a ("align Enhanced Attention UNet to predict noise residual"): the model outputs the denoised signal and `test_set/target_shots.npy` stores the clean signal (verified numerically: enhanced target equals `input - noise_target` of sanet's test set exactly). The noise-convention assumptions in `evaluate_one` made both `clean_ref` and `denoised` wrong.
- Change (`scripts/ground_roll_attenuation/batch_evaluate.py`):
  - New `SIGNAL_CONVENTION_MODELS = {"enhanced_atten_unet"}` constant marking signal-prediction-convention checkpoints. Remove the entry if the model is retrained with the aligned noise-residual script.
  - `evaluate_one` now dispatches model-specific inference adapters on `model.type` from the checkpoint config: `ddpm_unet` (DDIM sampling, unchanged), `physics_unet` (`batch - model.denoise(batch)`), and signal-convention models (`batch - model(batch)`). Every adapter returns predicted noise so `denoised = input - pred_noise` holds uniformly.
  - `clean_ref` is `target_shots` directly for signal-convention models, `input - target` otherwise.
- Impact: physics and enhanced_atten_unet can now be evaluated correctly. Note: existing Excel files contain wrong Enhanced Atten-UNet rows; `--merge` skips already-present entries, so delete those rows (or re-run without merge) before re-evaluating.

## 2026-07-20 - Pool ground-roll binned metrics over the full test volume

- Context: Ground-roll batch evaluation computed global SNR after reshaping the full test volume to one sample, while EB-WSE and FB-FRE averaged per-panel scores. The inconsistent reductions could reverse model rankings.
- Change:
  - `utils/inference_utils.py`: added `compute_pooled_binned_metrics`, which validates 3D inputs, preserves the time axis, reshapes shot and trace axes to `(1, -1, n_time)`, and delegates to the existing `compute_binned_metrics` without changing its behavior for other callers.
  - `scripts/ground_roll_attenuation/batch_evaluate.py`: switched EB-WSE and FB-FRE evaluation to the pooled helper.
  - `test/test_inference_utils.py`: added regression coverage for pooled EB-WSE/FB-FRE values, explicit-reshape equivalence, and pre-reshape shape validation.
- Impact: Ground-roll global, amplitude-binned, and frequency-binned metrics now use the same whole-volume reduction within each seed. Mean and standard deviation across independent seeds remain unchanged. Multiples attenuation keeps the existing per-panel behavior.
- Follow-up: Recompute all compared ground-roll models into a new workbook so pooled and legacy binned values are not mixed.

## 2026-08-23 - Split ground-roll configs and train scripts into field/sim variants

- Context: ground-roll experiments cover both simulation and field datasets, and the flat config/train-script layout made the two hard to tell apart.
- Change:
  - `configs/ground_roll_attenuation/`: flat configs moved to `sim/`; added `field/` (14 models) and `my_data/` (field1/field2) variants.
  - `scripts/ground_roll_attenuation/train/`: flat `.sh` scripts moved to `sim/`; added `field/` variants.
  - `scripts/ground_roll_attenuation/train/train_denoise_physics.py`: fix best-checkpoint value tracking — `best_val_loss` now captures the value returned by `_best(...)` instead of being set unconditionally before the call.
  - `.gitignore`: add `xibei_data`.
  - New: `scripts/multiples_attenuation/upload_model_to_hf.py`, `tools/download_results_0822.sh`, `utils/pad_segy_shots_to_201.py`, and tests `test/test_ground_roll_batch_evaluate.py`, `test/test_inference_utils.py`.
- Impact: field vs simulation runs are now separated by directory; existing experiment configs and result dirs are unchanged.

## 2026-08-26 - collect_evaluation.py: option to skip npy/ (visualizations only)

- Context: `tools/collect_evaluation.py` packaged whole `evaluation/` trees; the full-volume `npy/` arrays (~1 GB per experiment) dominate the archive when only the per-shot visualizations are wanted for sharing.
- Change: added `--exclude-npy` flag (default False, backward compatible). Copy mode passes `shutil.ignore_patterns("npy")` to `copytree`; archive mode passes a `tar.add` filter that drops the `npy` directory member (children are skipped automatically); the dry-run / per-dir size report now uses a new `_collect_size` helper that skips `npy` when the flag is set.
- Impact: `--exclude-npy` yields only `evaluation/visualizations/` (~37 MB vs ~1 GB per dir); existing calls without the flag behave exactly as before.

## 2026-08-26 - Physics: persist test_set + shared FFID split helper + backfill

- Context: `batch_evaluate.py --models physics` matched 0 experiments because `discover_results` requires `test_set/`, which `train_denoise_physics.py` never saved. The 3 `denoise_physics_base0822_level1.0_seed{42,43,44}` dirs were skipped.
- Change:
  - `utils/train_utils.py`: extracted `ffid_split_masks(per_shot_ffid, n_train, n_val, n_test) -> (train, val, test)` from `build_shot_split_loaders`; split semantics unchanged (unique FFID values sorted; first n_train / next n_val / last n_test unique values; all shots sharing a value stay together).
  - `scripts/ground_roll_attenuation/train/train_denoise_physics.py`: computes `test_mask` via the shared helper and persists `test_set/{input_shots,target_shots,ffid}.npy` on rank 0. `input_shots` = normalized noisy volume, `target_shots` = normalized ground-roll noise so batch_evaluate's `clean_ref = input - target` recovers the clean signal.
  - New `scripts/ground_roll_attenuation/backfill_physics_test_set.py`: regenerates missing `test_set/` for existing Physics result dirs. Reuses `_preprocess_shots` from the training script via importlib (no duplication) plus `ffid_split_masks`; caches the preprocessed volumes across dirs sharing the same SEG-Y pair (~8 GB, so one load serves all seeds); cross-checks `ffid.npy` against a sibling unet test set when present; `--force` overwrites an existing test_set.
- Data fact: the field SEG-Y (`dagang_noisy_padded201_1.0.sgy`, ~1699 shots) has FieldRecord header values that are NOT 1:1 with shots — only ~111 unique FFID values, so the `{89,11,11}` split's `test_mask` selects ~142 shots (all shots carrying the last 11 unique FFID values). The unet sibling test set holds 142 shots; the Physics backfill reproduces the same FFID ordering.
- Impact: `--models physics` now matches and can be evaluated on the 0822 tree. Future Physics runs auto-save `test_set/`.
