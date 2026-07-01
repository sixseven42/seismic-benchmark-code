# Physics-Constrained DNN for Nonstationary Coherent Noise Suppression

> Liu, J., Hu, T., Wang, C., Li, X., Zeng, Q., & Liu, L. (2025). *IEEE Trans. Geosci. Remote Sens., 63, 5903010.*

## 1. Core Idea

A deep neural network that jointly uses **data-driven labels** and **multimodal physical information constraints** to suppress near-surface-related nonstationary coherent noise (surface waves, single-frequency noise, linear noise). The physical constraints limit the solution space, reducing underfitting risk and enabling effective training under small-sample conditions.

## 2. Algorithm Pipeline

```
Input: Noisy prestack seismic data U0(t, x) (x–t domain)
              │
              ▼
    ┌─────────────────────────┐
    │  SDPAF preprocessing    │  ← nonlinear activation: amplifies
    │  (optional, before net) │    weak signals, preserves strong
    └─────────────────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │  FC1: 1×1 Conv          │  ← input (1 ch) + grid (2 ch) → C feature channels
    │  U1 = FC1(U0 + G)       │
    └─────────────────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │  MPIC Block × N         │  ← cascaded physical-constraint blocks
    │  ┌───────────────────┐  │
    │  │ Spatial modality  │──┤  T⁻¹(γ_T ⊙ Conv(T(x)))
    │  │ Frequency modality│──┤  F⁻¹(γ_F ⊙ Conv_freq(F(x)))
    │  │ Hilbert modality  │──┤  H⁻¹(γ_H ⊙ Conv_hilb(H(x)))
    │  │     + BN + ReLU   │  │
    │  │     + residual     │  │
    │  └───────────────────┘  │
    └─────────────────────────┘
              │
              ▼
    ┌─────────────────────────┐
    │  FCn: 1×1 Conv          │  ← C feature channels → output (1 ch)
    │  Un = FCn(U_{n-1})      │
    └─────────────────────────┘
              │
              ▼
Output: Denoised data Ûn(t, x)  (or predicted noise for residual learning)
```

### 2.1 SDPAF (Seismic Data Preprocessing Activation Function)

Addresses the amplitude disparity problem: surface waves have much higher amplitude than deep reflections. Small-amplitude signals contribute negligibly to gradients ("negative performance") during training.

**Design**: Piecewise smooth function with three regimes:
- **Low |X|** → nonlinear amplification (boost weak signals)
- **Medium |X|** → quasi-linear pass-through
- **High |X|** → near-linear preservation (keep strong amplitudes)

Parameters: `α ∈ [0.001, 0.1]` (nonlinear curvature), `k ∈ [1, 10]` (linear-regime width). The function is smooth and has smooth gradients everywhere, ensuring stable backpropagation.

Applied **before** training as a preprocessing step; the inverse SDPAF recovers true amplitudes for output.

### 2.2 MPIC Block (Multimodality Physical Information Constraint Block)

Each block processes a multi-channel feature matrix through parallel modality branches:

```
U_l = ReLU(BN( Σ_i Σ_j M_j(w_ij, U_{l-1}(i)) + b_ij ))

where M_j ∈ {M_spatial, M_frequency, M_hilbert}
```

#### Spatial modality M_T
```
M_T(w, U) = T⁻¹( γ_T ⊙ w * T(U) )
```
- `T, T⁻¹`: identity mapping (operates in native x–t domain)
- `γ_T`: spatial-domain constraint parameter (e.g., weighting for known noise regions)
- `w`: standard Conv2d kernel

#### Frequency modality M_F
```
M_F(w, U) = F⁻¹( γ_F(f) ⊙ w_freq * F(U) )
```
- `F, F⁻¹`: 2D real FFT / inverse FFT
- `γ_F(f)`: frequency-dependent piecewise mask — **0 in noise frequency range** (ground-roll: 0–15 Hz), **0~1 in reflection frequency range**
- `w_freq`: frequency-domain weight (complex conv via real/imag split)

#### Hilbert modality M_H
```
M_H(w, U) = H⁻¹( γ_H ⊙ w_hilb * H(U) )
```
- `H, H⁻¹`: Hilbert (analytic signal) forward / inverse transform
- `γ_H`: envelope-domain constraint
- `w_hilb`: weight in the analytic domain

**Residual connection**: Each MPIC Block includes a short-circuit (inspired by ResNet) to prevent vanishing gradients with increasing depth.

### 2.3 Loss Function

L2-norm relative error:
```
LPLoss(Ûn, Un_label) = ||Ûn - Un_label||_F / ||Un_label||_F
```

where `||·||_F` is the Frobenius norm over a training batch.

## 3. Training Strategy

| Parameter        | Value                        |
|-----------------|------------------------------|
| Optimizer        | Adam                         |
| Training samples  | 50 shots (small-sample regime) |
| Batch training   | mini-batch SGD               |
| Epochs           | 300 (loss stabilizes)        |
| SDPAF α          | 0.001 – 0.1 (default 0.01)  |
| SDPAF k          | 1 – 10 (default 5)           |
| Feature channels | 32–64                        |

### Key advantage over data-driven methods

Physical constraints limit the solution space → the optimizer avoids local optima (underfitting). The paper shows that with the same architecture, the physically constrained network converges to a lower loss than a pure data-driven network.

## 4. Reproduction Adaptations

### Simplifications vs. Paper

1. **SDPAF**: The full piecewise formula is approximated with a simpler, smooth `tanh`-based nonlinearity that shares the same qualitative behavior (boost weak signals, preserve strong ones, smooth gradients).
2. **Frequency constraint**: Instead of a piecewise γ_F(f), we use a learnable smooth frequency mask parameterized by a sigmoid of radial frequency: `mask(r) = σ((r - f_center) / sigma)` where `r = √(f_x² + f_y²)`.
3. **Hilbert constraint**: Implemented via analytic signal construction in FFT domain (double positive frequencies, zero negative), with learnable per-channel scaling.
4. **Position encoding**: The grid `G(c0, t, x)` is implemented as normalized 2D coordinate channels concatenated with the input.
5. **Noise prediction**: The model can output either denoised signal or predicted noise. For compatibility with the existing pipeline, it predicts noise (residual learning).

### SDPAF Approximation

The paper's SDPAF is complex with three piecewise branches. We approximate it with:
```
SDPAF(x) = x + β * tanh(γ * x)
```
where `β > 0` controls the amplification boost for small signals and `γ > 0` controls the transition width. This shares the key properties:
- Small |x|: `≈ x·(1 + βγ)` → amplified
- Large |x|: `≈ x ± β` → near-linear
- Smooth and differentiable everywhere

### Frequency Mask

Learnable soft sigmoid gate in the 2D frequency domain:
```
mask(fx, fy) = sigmoid((√(fx² + fy²) - f_center) / sigma)
```
- `f_center` and `sigma` are learnable per modality channel
- Initialized to suppress low frequencies (f_center ≈ 0.1×f_nyquist)

## 5. Key Parameters

| Parameter             | Default | Description                                    |
|-----------------------|---------|------------------------------------------------|
| `n_channels`          | 32      | Feature channels after first FC layer          |
| `n_mpic_blocks`       | 4       | Number of cascaded MPIC Blocks                 |
| `sdpaF_alpha`         | 0.01    | SDPAF nonlinear curvature (≈ β)               |
| `sdpaF_gamma`         | 5.0     | SDPAF transition sharpness (≈ γ)              |
| `use_spatial`         | true    | Enable spatial-domain modality in MPIC         |
| `use_frequency`       | true    | Enable frequency-domain modality in MPIC       |
| `use_hilbert`         | true    | Enable Hilbert-domain modality in MPIC         |
| `freq_init_center`    | 0.15    | Initial frequency mask center (fraction of Nyquist) |
| `freq_init_sigma`     | 0.05    | Initial frequency mask transition width        |
