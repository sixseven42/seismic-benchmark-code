# 能量分箱弱信号评估（EB-WSE）

EB-WSE 用于诊断去噪模型对**弱振幅信号**的保留能力。全局指标（如 MSE、整体 SNR）容易被强反射能量主导，从而掩盖弱信号的丢失。EB-WSE 将参考炮记录按能量百分位分成若干区间，分别计算每个区间的归一化误差（NE）和信噪比（SNR）。

当前实现位于 [`utils/eb_wse_metrics.py`](../../utils/eb_wse_metrics.py)。随机噪声压制推理脚本通过 [`utils/inference_utils.py::compute_binned_metrics`](../../utils/inference_utils.py) 只暴露每个能量箱的 **NE** 和 **SNR**。

---

## 解决的问题

全局 SNR 的提升可能主要来自强反射，而非常微弱的同相轴可能已经被抹去。EB-WSE 把低能量样本单独拿出来评估，因此即使全局 SNR 看起来很好，只要模型抹除了弱信号，底层能量箱的分数就会显著下降。

---

## 输入

- `reference` — 干净/目标炮记录 $r$，任意形状（推理中通常为 `(n_traces, n_time)`）。
- `prediction` — 模型输出 $p$，与 `reference` 同形状。
- `bins` — 能量百分位区间列表，每个元素为 `(low_percentile, high_percentile)`。默认值：
  - `very_weak_5_20` : $(5, 20)$
  - `weak_20_40`     : $(20, 40)$
  - `medium_40_70`   : $(40, 70)$
  - `strong_70_100`  : $(70, 100)$
- `smooth_sigma` — 构建能量图时使用的高斯平滑宽度，默认 `1.0`。
- `eps` — 防止除零的小常数，默认 `1e-8`。

---

## 计算流程

1. **能量图**

   $$
   E = \sqrt{\text{gaussian\_filter}(r^2, \; \sigma)}
   $$

   高斯滤波使用 `scipy.ndimage.gaussian_filter`；如果缺少 `scipy`，则回退到大小约为 $2\sigma+1$ 的均匀平均。

2. **有效样本**

   将 $r$、$p$、$E$ 展平，并剔除 $r_i = 0$ 的样本（这些位置没有信号可评估）。

3. **按能量排序分箱**

   将剩余样本按能量 $E_i$ 升序排列。对于区间 $(p_L, p_H)$，选取排名落在

   $$
   \left[ N_{\text{valid}} \cdot \frac{p_L}{100}, \; N_{\text{valid}} \cdot \frac{p_H}{100} \right)
   $$

   内的样本，其中 $N_{\text{valid}}$ 为非零参考样本总数。最高百分位区间的上界闭合，以保证区间比例准确。

4. **逐箱指标**

   设 $\mathcal{B}$ 为某个箱内的样本集合，$N_{\mathcal{B}} = |\mathcal{B}|$。

   ### 归一化误差（NE）

   $$
   \text{NE}_{\mathcal{B}} =
   \frac{\sqrt{ \frac{1}{N_{\mathcal{B}}} \sum_{i \in \mathcal{B}} (p_i - r_i)^2 }}
        {\sqrt{ \frac{1}{N_{\mathcal{B}}} \sum_{i \in \mathcal{B}} r_i^2 } + \varepsilon}
   $$

   - $\text{NE} < 1$：误差小于信号，恢复较好。
   - $\text{NE} = 1$：误差与信号 RMS 幅度相当。
   - $\text{NE} > 1$：误差大于信号本身，说明该能量箱内的信号已被破坏。

   ### 信噪比（SNR）

   $$
   \text{SNR}_{\mathcal{B}} = 10 \log_{10}
   \left(
       \frac{\sum_{i \in \mathcal{B}} r_i^2}
            {\sum_{i \in \mathcal{B}} (p_i - r_i)^2}
   \right)
   $$

   边界情况：

   - 残差能量为 0 且信号能量为正时，SNR 为 $+\infty$（完美重建）。
   - 信号能量为 0 时，SNR 为 $-\infty$（该箱内无信号）。
   - 在 [`compute_binned_metrics`](../../utils/inference_utils.py) 中，非有限值会被处理为 JSON 安全值：`NaN → null`，`+∞ → 999.0`，`-∞ → -999.0`。

---

## 配置方法

所有参数都在 YAML 配置的 `inference.binned_metrics.eb_wse` 块中控制。如果该块缺失，则使用以下默认值，旧配置仍然兼容。

```yaml
inference:
  binned_metrics:
    enabled: true
    eb_wse:
      enabled: true
      bins: [[5, 20], [20, 40], [40, 70], [70, 100]]
      smooth_sigma: 1.0
```

- `enabled` — 设为 `false` 可跳过 EB-WSE 计算。
- `bins` — `(低百分位, 高百分位)` 区间列表。自定义区间会生成类似 `eb_wse_bin_10_30_ne` 的字段名，而非默认命名。
- `smooth_sigma` — 能量图高斯平滑的 sigma。

---

## 推理输出字段

随机噪声压制推理脚本会把每个能量箱的均值写入 `metrics_summary.json`，字段名如下：

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

` *_ne` 对应上面的 NE 公式，` *_snr` 对应 SNR 公式。所有值都是对所有测试炮取平均后的结果。

---

## 结果解读

- 对比每个箱中 `noisy` 与 `denoised` 的 SNR。有效的模型应在所有箱都提升 SNR，且提升幅度在输入 SNR 最低的 `very_weak` 箱最明显。
- 如果 `denoised` 在 `very_weak_5_20` 箱的 SNR 接近甚至低于 `noisy` 的 SNR，说明模型把弱信号连同噪声一起去掉了。
- `metrics_summary.json` 中的 `delta`（`denoised - noisy`）可以直接看出改善：SNR 的 `delta` 为正、NE 的 `delta` 为负表示更好。

---

## 局限性

- 参考值为严格 0 的样本被排除，因此 EB-WSE 不评估完全安静区域的恢复情况。
- 能量图依赖 `smooth_sigma`；默认 `1.0` 会把孤立的强振幅尖峰视为周围高能量区域的一部分。只有在对“弱信号”的物理尺度有明确认识时才应调整该参数。
- 分箱完全基于参考数据，预测结果不影响样本属于哪个箱。
