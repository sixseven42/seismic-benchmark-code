# Energy-Binned Weak Signal Evaluation (EB-WSE)

EB-WSE diagnoses how well a denoising model preserves **weak-amplitude signal** that global metrics such as MSE or overall SNR can hide. It splits the reference shot into energy percentile bins and reports per-bin normalized error (NE) and signal-to-noise ratio (SNR).

The current implementation lives in [`utils/eb_wse_metrics.py`](../../utils/eb_wse_metrics.py). The random-noise suppression inference scripts expose only the **NE** and **SNR** values through [`utils/inference_utils.py::compute_binned_metrics`](../../utils/inference_utils.py).

---

## What problem it solves

A global SNR improvement can be dominated by strong reflections while very weak events are still lost. EB-WSE isolates the low-energy samples, so a model that erases faint signals receives a low (bad) score in the bottom energy bins even if its global SNR looks good.

---

## Inputs

- `reference` — clean/target shot $r$, any shape (in inference it is `(n_traces, n_time)`).
- `prediction` — model output $p$, same shape as `reference`.
- `bins` — list of `(low_percentile, high_percentile)` pairs. Default:
  - `very_weak_5_20` : $(5, 20)$
  - `weak_20_40`     : $(20, 40)$
  - `medium_40_70`   : $(40, 70)$
  - `strong_70_100`  : $(70, 100)$
- `smooth_sigma` — Gaussian smoothing width used to build the energy map. Default `1.0`.
- `eps` — small constant to avoid division by zero. Default `1e-8`.

---

## Algorithm

1. **Energy map**

   $$
   E = \sqrt{\text{gaussian\_filter}(r^2, \; \sigma)}
   $$

   The Gaussian filter uses `scipy.ndimage.gaussian_filter`; if `scipy` is missing it falls back to a uniform average over a window of size roughly $2\sigma+1$.

2. **Valid samples**

   Flatten $r$, $p$, and $E$. Discard samples where $r_i = 0$ exactly, because they carry no signal to evaluate.

3. **Rank-based bins**

   Sort the remaining samples by their energy $E_i$ in ascending order. For a bin $(p_L, p_H)$ select the samples whose rank lies in

   $$
   \left[ N_{\text{valid}} \cdot \frac{p_L}{100}, \; N_{\text{valid}} \cdot \frac{p_H}{100} \right)
   $$

   where $N_{\text{valid}}$ is the number of non-zero reference samples. The highest percentile uses a closed upper bound so the bin contains exactly the requested fraction.

4. **Per-bin metrics**

   Let $\mathcal{B}$ be the set of samples in a bin and $N_{\mathcal{B}} = |\mathcal{B}|$.

   ### Normalized Error (NE)

   $$
   \text{NE}_{\mathcal{B}} =
   \frac{\sqrt{ \frac{1}{N_{\mathcal{B}}} \sum_{i \in \mathcal{B}} (p_i - r_i)^2 }}
        {\sqrt{ \frac{1}{N_{\mathcal{B}}} \sum_{i \in \mathcal{B}} r_i^2 } + \varepsilon}
   $$

   - $\text{NE} < 1$ : the error is smaller than the signal — good recovery.
   - $\text{NE} = 1$ : error and signal have comparable RMS amplitude.
   - $\text{NE} > 1$ : the error is larger than the signal itself — the model has destroyed the bin.

   ### Signal-to-Noise Ratio (SNR)

   $$
   \text{SNR}_{\mathcal{B}} = 10 \log_{10}
   \left(
       \frac{\sum_{i \in \mathcal{B}} r_i^2}
            {\sum_{i \in \mathcal{B}} (p_i - r_i)^2}
   \right)
   $$

   Edge cases:

   - If the residual energy is zero and the signal energy is positive, SNR is $+\infty$ (perfect reconstruction).
   - If the signal energy is zero, SNR is $-\infty$ (no signal in the bin).
   - In [`compute_binned_metrics`](../../utils/inference_utils.py) non-finite values are sanitized for JSON: `NaN → null`, `+∞ → 999.0`, `-∞ → -999.0`.

---

## Configuration

All parameters are controlled from the `inference.binned_metrics.eb_wse` block in the YAML config. If the block is missing, the defaults below are used, so older configs keep working.

```yaml
inference:
  binned_metrics:
    enabled: true
    eb_wse:
      enabled: true
      bins: [[5, 20], [20, 40], [40, 70], [70, 100]]
      smooth_sigma: 1.0
```

- `enabled` — set to `false` to skip EB-WSE entirely.
- `bins` — list of `(low_percentile, high_percentile)` pairs. Custom bins produce keys like `eb_wse_bin_10_30_ne` instead of the named defaults.
- `smooth_sigma` — Gaussian smoothing sigma for the energy map.

---

## Output keys in inference

The random-noise suppression inference scripts write mean EB-WSE values into `metrics_summary.json` with these keys:

```text
eb_wse_very_weak_5_20_ne
eb_wse_very_weak_5_20_snr
eb_wse_weak_20_40_ne
eb_wse_weak_20_40_snr
eb_wse_medium_40_70_ne
eb_wse_medium_40_70_snr
eb_wse_strong_70_100_ne
eb_wse_strong_70_100_snr
```

The `*_ne` keys use the NE formula above; the `*_snr` keys use the SNR formula above. Values are **means over all test shots**.

---

## How to interpret results

- Compare `noisy` vs `denoised` SNR in each bin. A useful model raises SNR in every bin, but the gain should be largest where the input SNR is lowest (the `very_weak` bin).
- If `denoised` SNR in `very_weak_5_20` is close to or lower than `noisy` SNR, the model is suppressing weak signals together with noise.
- The `delta` entry in `metrics_summary.json` (`denoised - noisy`) makes this comparison explicit: positive `snr` deltas and negative `ne` deltas indicate improvement.

---

## Limitations

- Samples where the reference is exactly zero are excluded, so EB-WSE does not measure performance in completely quiet regions.
- The energy map depends on `smooth_sigma`; the default `1.0` treats single high-amplitude spikes as part of the surrounding high-energy region rather than isolated outliers. Change `smooth_sigma` only when the physical scale of a "weak signal" is well understood.
- Percentile bins are rank-based on the reference only; the prediction does not influence bin membership.
