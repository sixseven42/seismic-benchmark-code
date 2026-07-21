# 频率分箱保真度与恢复评估（FB-FRE）

FB-FRE 用于诊断去噪模型在**不同频率段**上的恢复质量。它从干净参考数据估计有效频带，将有效频带自适应划分为低、中、高、甚高四个子带，然后分别计算每个子带的归一化误差（NE）、信噪比（SNR）以及该频段能量占总能量的比例。

实现位于 [`utils/fb_fre_metrics.py`](../../utils/fb_fre_metrics.py)。随机噪声压制推理脚本通过 [`utils/inference_utils.py::compute_binned_metrics`](../../utils/inference_utils.py) 只暴露每个频段的 **NE、SNR、能量占比、频率范围**。

---

## 解决的问题

去噪模型往往通过压制高频来抑制噪声，但这同时也会损失合理的高频信号。FB-FRE 将频谱分段评估，使频率相关的信号损失变得可见。

---

## 输入

- `reference` — 干净/目标炮记录 $r$，任意形状（推理中通常为 `(n_traces, n_time)`）。
- `prediction` — 模型输出 $p$，与 `reference` 同形状。
- `dt` — 时间采样间隔，单位为秒（例如 SEG-Y 数据为 8 ms 时 `dt=0.008`）。
- `rel_threshold` — 定义有效频带的峰值功率比例，默认 `0.001`（0.1% 峰值功率）。
- `bands` — 频段列表 `(name, (f_min, f_max))`，或 `"auto"` 由有效频带自动生成。推理固定使用 `"auto"`。
- `taper_width` — 频段边界的余弦过渡宽度，单位为 Hz。默认 `0.0`（矩形通带）。
- `eps` — 防止除零的小常数，默认 `1e-8`。

---

## 计算流程

1. **平均功率谱**

   对参考数据 $r$ 沿时间轴做实数 FFT：

   $$
   R(f) = \text{RFFT}(r, \; \text{axis}=\text{time})
   $$

   频率网格为

   $$
   f_k = \text{rfftfreq}(N_t, \; d=dt)
   $$

   其中 $N_t$ 为时间采样点数。对所有非时间维度平均后得到功率谱：

   $$
   P(f_k) = \frac{1}{N_{\text{avg}}} \sum_{\text{非时间轴}} |R(f_k)|^2
   $$

2. **有效频带估计**

   默认 `"threshold"` 方法下，有效频带是平均功率不低于峰值功率 0.1% 的连续区域：

   $$
   f_{\min} = \min \{ f_k \mid P(f_k) \ge 0.001 \cdot \max_f P(f) \}
   $$
   $$
   f_{\max} = \max \{ f_k \mid P(f_k) \ge 0.001 \cdot \max_f P(f) \}
   $$

   边界被裁剪到 `[0, f_\text{Nyquist}]`，其中

   $$
   f_{\text{Nyquist}} = \frac{1}{2 \, dt}
   $$

3. **自适应子带划分**

   有效频带 $[f_{\min}, f_{\max}]$ 按相对宽度

   $$
   (0.20, \; 0.30, \; 0.30, \; 0.20)
   $$

   划分为 `low`、`mid`、`high`、`very_high` 四个连续频段。第 $b$ 个频段的累积宽度为 $c$ 时，其上边界为

   $$
   f_{\text{edge}} = f_{\min} + (f_{\max} - f_{\min}) \cdot c
   $$

   最后一个频段强制结束在 $f_{\max}$，避免浮点漂移。

4. **带通滤波**

   对每个频段 $(f_{\min}^{(b)}, f_{\max}^{(b)})$，在 RFFT 频率网格上构造掩码 $M_b(f)$。默认 `taper_width=0` 时为矩形掩码：

   $$
   M_b(f) =
   \begin{cases}
   1 & f_{\min}^{(b)} \le f \le f_{\max}^{(b)} \\
   0 & \text{其他}
   \end{cases}
   $$

   对参考和预测做同样的滤波：

   $$
   r_b = \text{IRFFT}\big( M_b \cdot \text{RFFT}(r) \big)
   $$
   $$
   p_b = \text{IRFFT}\big( M_b \cdot \text{RFFT}(p) \big)
   $$

   两者使用完全相同的滤波器，保证对比公平。

5. **逐频段指标**

   设 $r_b$、$p_b$ 为滤波后展平的频段信号。

   ### 归一化误差（NE）

   $$
   \text{NE}_b =
   \frac{\sqrt{\sum (p_b - r_b)^2}}
        {\sqrt{\sum r_b^2} + \varepsilon}
   $$

   - $\text{NE}_b \ll 1$：该频段恢复较好。
   - $\text{NE}_b \approx 1$：残差能量与该频段信号能量相当。
   - $\text{NE}_b > 1$：模型在该频段引入的能量超过原始信号。

   ### 信噪比（SNR）

   $$
   \text{SNR}_b = 10 \log_{10}
   \left(
       \frac{\sum r_b^2}{\sum (p_b - r_b)^2 + \varepsilon}
   \right)
   $$

   如果参考频段能量为 0，SNR 为 $-\infty$（该频段无信号）。在 [`compute_binned_metrics`](../../utils/inference_utils.py) 中，非有限值会被处理为 JSON 安全值：`NaN → null`，`+∞ → 999.0`，`-∞ → -999.0`。

   ### 能量占比

   $$
   \text{energy\_ratio}_b =
   \frac{\sum r_b^2}{\sum r^2 + \varepsilon}
   $$

   表示频段 $b$ 的能量占参考总能量的比例。

---

## 配置方法

所有参数都在 YAML 配置的 `inference.binned_metrics.fb_fre` 块中控制。如果该块缺失，则使用以下默认值，旧配置仍然兼容。

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

- `enabled` — 设为 `false` 可跳过 FB-FRE 计算。
- `rel_threshold` — 定义有效频带的峰值功率比例。
- `band_ratios` — 自适应子带的相对宽度，必须和为 `1.0`。
- `band_names` — 每个自适应子带的名称。
- `taper_width` — 频段边界的余弦过渡宽度（Hz）；`0.0` 表示矩形通带。

---

## 推理输出字段

随机噪声压制推理脚本会把均值写入 `metrics_summary.json`，以 `low` 频段为例：

```text
fb_fre_low_ne
fb_fre_low_snr
fb_fre_low_energy_ratio
fb_fre_low_frequency_range_hz
```

`mid`、`high`、`very_high` 三个频段字段名模式相同。`frequency_range_hz` 是两元素列表 `[f_min, f_max]`，单位为 Hz；其余为对所有测试炮取平均后的标量。

---

## 结果解读

- 有效的模型应在 **全部四个** 频段都提升 SNR（并降低 NE）。
- 如果 `high` 或 `very_high` 频段的 SNR 提升明显小于 `low` 频段，说明模型可能存在过度平滑，把细节结构与噪声一起去除了。
- `energy_ratio` 表示该频段在数据中的重要性。仅占 2% 总能量的频段允许恢复稍差；占 40% 总能量的频段则应重点关注。
- 比较不同实验的 `fb_fre_*_frequency_range_hz` 时，必须保证 `dt` 相同；Hz 绝对值直接依赖于采样间隔。

---

## 与 `utils/fb_fre_metrics.py` 的关系

完整模块还提供了 `frequency_binned_fidelity_metrics`，可计算每个频段的 `BNE`、`BER`、`BCC`。推理包装器只输出 NE 和 SNR，原因是：

- `BNE` 与上述 NE 数学等价。
- `BER`（频段能量比）和 `BCC`（频段相关系数）与 NE/SNR 信息重叠，当前随机噪声压制基准输出未包含它们。

如需 BER/BCC，可直接调用 `frequency_binned_fidelity_metrics`。

---

## 局限性

- 有效频带是数据相关的：不同的干净数据会给出略有不同的 $f_{\min}$、$f_{\max}$ 以及子带边界。报告分数时必须同时给出 Hz 范围。
- 默认使用矩形带通掩码（`taper_width=0`），可能在时间域产生吉布斯现象/振铃。若需更平滑的频谱分离，可增大 `taper_width`，但这会加宽有效边界。
- FB-FRE 假设时间轴是最后一维。每炮独立处理，不使用道间（空间）频率信息。
- 0.1% 阈值用于在有效信号带宽与数值噪声之间取舍；低于 0.1% 峰值功率的极低能量尾部不被评估。
