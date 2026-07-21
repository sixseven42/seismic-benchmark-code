# Frequency-Binned Fidelity and Recovery Evaluation (FB-FRE)

FB-FRE diagnoses how well a denoising model recovers signal **across the frequency spectrum**. It estimates the effective frequency band from the clean reference, splits that band into adaptive low/mid/high/very_high sub-bands, and reports per-band normalized error (NE) and signal-to-noise ratio (SNR). An energy-ratio diagnostic is also reported so each band can be interpreted relative to the total signal energy.

The implementation lives in [`utils/fb_fre_metrics.py`](../../utils/fb_fre_metrics.py). The random-noise suppression inference scripts expose only per-band **NE**, **SNR**, **energy ratio**, and **frequency range** through [`utils/inference_utils.py::compute_binned_metrics`](../../utils/inference_utils.py).

---

## What problem it solves

Denoising models often attenuate noise by suppressing high frequencies, but the same operation can remove legitimate high-frequency signal. FB-FRE makes frequency-specific loss visible by measuring reconstruction quality inside each frequency band separately.

---

## Inputs

- `reference` — clean/target shot $r$, any shape (in inference it is `(n_traces, n_time)`).
- `prediction` — model output $p$, same shape as `reference`.
- `dt` — time sampling interval in seconds (e.g. `0.008` for 8 ms SEG-Y data).
- `rel_threshold` — fraction of peak power used to define the effective band. Default `0.001` (0.1% of peak power).
- `bands` — list of `(name, (f_{\min}, f_{\max}))` tuples, or `"auto"` to derive them from the effective band. Inference always uses `"auto"`.
- `taper_width` — cosine taper width at band edges in Hz. Default `0.0` (rectangular passband).
- `eps` — small constant to avoid division by zero. Default `1e-8`.

---

## Algorithm

1. **Average power spectrum**

   For the reference volume $r$, compute the real FFT along the time axis:

   $$
   R(f) = \text{RFFT}(r, \; \text{axis}=\text{time})
   $$

   The frequency grid is

   $$
   f_k = \text{rfftfreq}(N_t, \; d=dt)
   $$

   where $N_t$ is the number of time samples. The average power spectrum across all non-time dimensions is

   $$
   P(f_k) = \frac{1}{N_{\text{avg}}} \sum_{\text{non-time axes}} |R(f_k)|^2
   $$

2. **Effective frequency band**

   With the default `"threshold"` method, the effective band is the contiguous region where the average power is at least `rel_threshold` of the peak:

   $$
   f_{\min} = \min \{ f_k \mid P(f_k) \ge 0.001 \cdot \max_f P(f) \}
   $$
   $$
   f_{\max} = \max \{ f_k \mid P(f_k) \ge 0.001 \cdot \max_f P(f) \}
   $$

   The bounds are clipped to `[0, f_{\text{Nyquist}}]` where

   $$
   f_{\text{Nyquist}} = \frac{1}{2 \, dt}
   $$

3. **Adaptive sub-bands**

   The effective band $[f_{\min}, f_{\max}]$ is split into four contiguous bands with relative widths

   $$
   (0.20, \; 0.30, \; 0.30, \; 0.20)
   $$

   named `low`, `mid`, `high`, and `very_high`. For a cumulative width $c$ after adding a band's ratio, the upper edge is

   $$
   f_{\text{edge}} = f_{\min} + (f_{\max} - f_{\min}) \cdot c
   $$

   The last band is forced to end exactly at $f_{\max}$ to avoid rounding drift.

4. **Band-pass filtering**

   For each band $(f_{\min}^{(b)}, f_{\max}^{(b)})$, build a mask $M_b(f)$ on the RFFT frequency grid. With the default `taper_width=0` the mask is rectangular:

   $$
   M_b(f) =
   \begin{cases}
   1 & f_{\min}^{(b)} \le f \le f_{\max}^{(b)} \\
   0 & \text{otherwise}
   \end{cases}
   $$

   The filtered reference and prediction for band $b$ are

   $$
   r_b = \text{IRFFT}\big( M_b \cdot \text{RFFT}(r) \big)
   $$
   $$
   p_b = \text{IRFFT}\big( M_b \cdot \text{RFFT}(p) \big)
   $$

   Both signals use **identical** filtering so the comparison is fair.

5. **Per-band metrics**

   Let the flattened band signals be $r_b$ and $p_b$.

   ### Normalized Error (NE)

   $$
   \text{NE}_b =
   \frac{\sqrt{\sum (p_b - r_b)^2}}
        {\sqrt{\sum r_b^2} + \varepsilon}
   $$

   - $\text{NE}_b \ll 1$ : the band is recovered well.
   - $\text{NE}_b \approx 1$ : the residual has the same energy as the band itself.
   - $\text{NE}_b > 1$ : the model has introduced more energy in that band than the original signal.

   ### Signal-to-Noise Ratio (SNR)

   $$
   \text{SNR}_b = 10 \log_{10}
   \left(
       \frac{\sum r_b^2}{\sum (p_b - r_b)^2 + \varepsilon}
   \right)
   $$

   If the reference band energy is zero, SNR is $-\infty$ (no signal in the band). In [`compute_binned_metrics`](../../utils/inference_utils.py) non-finite values are sanitized for JSON: `NaN → null`, `+∞ → 999.0`, `-∞ → -999.0`.

   ### Energy ratio

   $$
   \text{energy\_ratio}_b =
   \frac{\sum r_b^2}{\sum r^2 + \varepsilon}
   $$

   This shows what fraction of the total reference energy lives inside band $b$.

---

## Configuration

All parameters are controlled from the `inference.binned_metrics.fb_fre` block in the YAML config. If the block is missing, the defaults below are used, so older configs keep working.

```yaml
inference:
  binned_metrics:
    enabled: true
    fb_fre:
      enabled: true
      rel_threshold: 0.001
      band_ratios: [0.20, 0.30, 0.30, 0.20]
      band_names: ["low", "mid", "high", "very_high"]
      taper_width: 0.0
```

- `enabled` — set to `false` to skip FB-FRE entirely.
- `rel_threshold` — fraction of peak power used to define the effective frequency band.
- `band_ratios` — relative widths of the adaptive bands; must sum to `1.0`.
- `band_names` — name for each adaptive band.
- `taper_width` — cosine taper width in Hz at band edges; `0.0` gives a rectangular passband.

---

## Output keys in inference

The random-noise suppression inference scripts write mean FB-FRE values into `metrics_summary.json` with these keys (example for the `low` band):

```text
fb_fre_low_ne
fb_fre_low_snr
fb_fre_low_energy_ratio
fb_fre_low_frequency_range_hz
```

The same pattern is repeated for `mid`, `high`, and `very_high`. `frequency_range_hz` is a two-element list `[f_min, f_max]` in Hz. All other values are scalar means over the test shots.

---

## How to interpret results

- A useful model should raise SNR (and lower NE) in **all four** bands compared with the noisy input.
- If the `high` or `very_high` band shows a smaller SNR improvement than the `low` band, the model is likely over-smoothing and removing fine structure together with noise.
- `energy_ratio` tells you whether a band matters for this dataset. A band that carries 2% of the total energy can be allowed slightly worse recovery than a band that carries 40%.
- Compare `fb_fre_*_frequency_range_hz` across experiments only when `dt` is identical; the absolute Hz values depend directly on the sampling interval.

---

## Relationship to `utils/fb_fre_metrics.py`

The full module also exports `frequency_binned_fidelity_metrics`, which computes `BNE`, `BER`, and `BCC` per band. The inference wrapper deliberately reports only NE and SNR because:

- `BNE` is mathematically equivalent to the NE shown above.
- `BER` (band energy ratio) and `BCC` (band correlation coefficient) provide similar information but were not requested for the current random-noise suppression benchmark output.

If you need BER/BCC, call `frequency_binned_fidelity_metrics` directly.

---

## Limitations

- The effective band is data-dependent: different clean volumes can yield slightly different $f_{\min}$ and $f_{\max}$, and therefore different sub-band edges. Reported Hz ranges should always be shown alongside the metric values.
- Rectangular band-pass masks (`taper_width=0`) can cause ringing in the time domain. For sharper spectral separation use a positive `taper_width`, but note that this widens the effective band edges.
- FB-FRE assumes the time axis is the last axis. Shots are processed independently; no spatial (trace-to-trace) frequency information is used.
- The 0.1% threshold (`rel_threshold=0.001`) was chosen to capture the meaningful signal bandwidth while excluding numerical noise. Very low-amplitude tails below 0.1% of peak power are not evaluated.
