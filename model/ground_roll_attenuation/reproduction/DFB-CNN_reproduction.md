# DFB-CNN: Dual-Filter-Bank Convolutional Neural Network for Ground-Roll Attenuation

> Zhang, C., & van der Baan, M. (2022). *IEEE Trans. Geosci. Remote Sens., 60, 5907511.*

## 1. Core Idea

A dual-filter-bank CNN that separately processes **low-frequency** and **high-frequency** components of seismic data, using two CNNs with different kernel sizes and depths. The rationale: ground-roll is concentrated in low frequencies with broad spatial extent, while reflections span both low and high frequencies with narrower features.

## 2. Algorithm Pipeline

```
Input: Noisy shot gather y = s + n (x–t domain)
                                    │
                    ┌───────────────┴───────────────┐
                    │  Low-pass filter (0 ~ L Hz)    │
                    │  L ∈ [10, 20] Hz               │
                    └───────────────┬───────────────┘
                    │                               │
              y_l (low-freq)                  y_h = y - y_l (high-freq)
                    │                               │
              R{·} (RT transform)             R{·} (RT transform)
                    │                               │
              CNN1 (5×5 kernel,             CNN2 (3×3 kernel,
                    9 layers, 100 feat)           5 layers, 64 feat)
                    │                               │
              F1(R{y_l}) = noise_pred      F2(R{y_h}) = noise_pred
                    │                               │
        ŝ_l = R⁻¹{R{y_l} - F1(...)}  ŝ_h = R⁻¹{R{y_h} - F2(...)}
                    │                               │
                    └───────────────┬───────────────┘
                                    │
                          ŝ = ŝ_l + ŝ_h   (denoised output)
```

### 2.1 Frequency Separation

- A low-pass filter with cutoff `L ∈ [10, 20] Hz` splits input into:
  - `y_l` = low-frequency component (contains most ground-roll)
  - `y_h` = y - y_l (high-frequency component)
- `L` is **randomly sampled** during training → data augmentation, no fixed threshold

### 2.2 Radial Trace (RT) Transform

- Maps `(offset x, time t)` → `(apparent velocity v, shifted time t')`
- `t' = t - t0`, `v = (x - x0) / (t - t0)`
- Origin `(x0, t0)` at source point of shot gather
- **Effect**: Ground-roll (low apparent velocity, passes through origin) compacts near center of RT domain; reflections (high velocity, not through origin) spread out
- **Benefit**: Emphasizes velocity/spatial differences between signal and noise; easier for CNN to separate

### 2.3 Dual CNN Architecture (DnCNN-style residual)

Both CNNs predict **noise** (residual learning: `x - net(x)`), not signal. Architecture per CNN:

```
Input → Conv(k×k, n_feat) → ReLU
      → [Conv(k×k, n_feat) → BN → ReLU] × (depth-2)
      → Conv(k×k, 1)
Output = Input - net(Input)
```

|                    | CNN1 (low-freq) | CNN2 (high-freq) |
|--------------------|:---------------:|:----------------:|
| Kernel size        | 5 × 5           | 3 × 3            |
| Depth (total conv) | 9               | 5                |
| Feature maps       | 100             | 64               |
| Parameters         | ~1.75 M         | ~0.1 M           |

### 2.4 Loss Function

MSE between predicted noise and true noise in RT domain:
- L1 = (1/N) Σ ||F1(R{y_l}) - R{n_l}||²_F
- L2 = (1/N) Σ ||F2(R{y_h}) - R{n_h}||²_F

## 3. Training Strategy

| Parameter    | Value                        |
|-------------|------------------------------|
| Optimizer    | Adam (default β₁=0.9, β₂=0.999) |
| Learning rate| Exponential decay 1e-3 → 1e-5 |
| Epochs       | 50                           |
| Batch size   | 32                           |
| Patch size   | 80 × 80 (in RT domain)       |
| Patch stride | 36                           |
| Training data| 64,960 patches per branch    |

### Training Data Construction

1. **Synthetic reflections**: 100 clean shot gathers (128 traces × 1500 samples), Ricker wavelets, 6–20 hyperbolic events each
2. **Field ground-roll**: Extracted from real land data (various geometries, irregular shapes)
3. **Noise injection**: White Gaussian noise (σ ∈ [0, 0.2]) for robustness
4. Mixed: synthetic signal + field noise + random noise

## 4. Reproduction Adaptations

### Simplifications vs. Paper

1. **RT transform on patches**: Paper does frequency split → RT → patchify. We do patchify → RT inside model to fit existing pipeline.
2. **Low-pass filter**: Paper uses frequency-domain filter. We use spatial Gaussian blur (kernel size ~cutoff-equivalent), differentiable on GPU.
3. **RT origin**: Paper uses shot source point. For patches, origin is at bottom-center of each patch.
4. **Training data**: Reuse existing paired SEG-Y dataset (noisy/noise volumes) instead of building synthetic + field mixing.

### RT Transform Implementation

Implemented via `torch.nn.functional.grid_sample`:
- **Forward**: For each (v, t') grid point, compute (x, t) = (x0 + v·(t - t0), t) in input space, bilinear sample
- **Inverse**: For each (x, t) grid point, compute (v, t') = ((x - x0)/(t - t0 + ε), t - t0) in RT space, bilinear sample
- Velocity range: [-v_max, v_max] with configurable number of velocity bins
- Fully differentiable, GPU-compatible

### Frequency Split Implementation

- Spatial Gaussian low-pass filter (σ controlled by equivalent cutoff frequency)
- `y_l = GaussianBlur(y)`, `y_h = y - y_l`
- Differentiable, works on patches directly

## 5. Key Parameters

| Parameter          | Default    | Description                              |
|--------------------|------------|------------------------------------------|
| `low_cutoff_hz`   | 15.0       | Cutoff freq for low-pass (Hz); randomized ±5 during training |
| `v_max`           | 3200.0     | Max velocity in RT domain (m/s)          |
| `n_velocities`    | 128        | Number of velocity bins in RT domain     |
| `cnn1_kernel_size`| 5          | CNN1 (low-freq) kernel size              |
| `cnn1_depth`      | 9          | CNN1 conv layers count                   |
| `cnn1_base_channels`| 100      | CNN1 feature maps per layer              |
| `cnn2_kernel_size`| 3          | CNN2 (high-freq) kernel size             |
| `cnn2_depth`      | 5          | CNN2 conv layers count                   |
| `cnn2_base_channels`| 64       | CNN2 feature maps per layer              |
