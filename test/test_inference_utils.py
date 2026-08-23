from __future__ import annotations

import unittest

import numpy as np

from utils.inference_utils import (
    align_max_abs_shot_to_global,
    compute_binned_metrics,
    compute_pooled_binned_metrics,
)


class ComputePooledBinnedMetricsTest(unittest.TestCase):
    def test_matches_explicit_reshape_for_eb_and_fb_metrics(self) -> None:
        n_time = 64
        dt = 1.0 / 64.0
        time = np.arange(n_time) * dt
        base = np.sin(2.0 * np.pi * 4.0 * time) + np.sin(
            2.0 * np.pi * 8.0 * time
        )
        target = np.stack([base[None, :], (10.0 * base)[None, :]])
        pred = target.copy()
        pred[0] += 0.1 * base
        pred[1] += 10.0 * base
        kwargs = {
            "eb_bins": ((0, 100),),
            "eb_smooth_sigma": 0.0,
            "fb_band_ratios": (1.0,),
            "fb_band_names": ("all",),
        }

        pooled = compute_pooled_binned_metrics(pred, target, dt=dt, **kwargs)
        expected = compute_binned_metrics(
            pred.reshape(1, -1, n_time),
            target.reshape(1, -1, n_time),
            dt=dt,
            **kwargs,
        )
        per_shot = compute_binned_metrics(pred, target, dt=dt, **kwargs)

        self.assertEqual(pooled, expected)
        for prefix in ("eb_wse_bin_0_100", "fb_fre_all"):
            self.assertEqual(pooled[f"{prefix}_ne"], 0.995087)
            self.assertEqual(pooled[f"{prefix}_snr"], 0.042779)
            self.assertEqual(per_shot[f"{prefix}_ne"], 0.55)
            self.assertEqual(per_shot[f"{prefix}_snr"], 10.0)

    def test_rejects_shape_mismatch_before_reshape(self) -> None:
        pred = np.zeros((2, 3, 4), dtype=np.float32)
        target = np.zeros((1, 6, 4), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            compute_pooled_binned_metrics(pred, target, dt=0.004)


class AlignMaxAbsShotToGlobalTest(unittest.TestCase):
    def test_recovers_per_shot_scales_without_mutating_inputs(self) -> None:
        input_shots = np.array(
            [
                [[1.0, -0.5, 0.25], [0.2, -0.1, 0.4]],
                [[-1.0, 0.4, 0.2], [0.5, -0.25, 0.1]],
            ],
            dtype=np.float32,
        )
        clean = input_shots * 0.6
        pred = clean + np.array([0.05, -0.1], dtype=np.float32)[:, None, None]
        expected_scales = np.array([0.25, 0.8], dtype=np.float64)
        global_input = input_shots * expected_scales.astype(np.float32)[:, None, None]
        originals = tuple(array.copy() for array in (input_shots, clean, pred, global_input))

        aligned_input, aligned_clean, aligned_pred, scales = (
            align_max_abs_shot_to_global(
                input_shots, clean, pred, global_input
            )
        )

        np.testing.assert_allclose(scales, expected_scales, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(aligned_input, global_input)
        np.testing.assert_allclose(
            aligned_clean, clean * expected_scales[:, None, None]
        )
        np.testing.assert_allclose(
            aligned_pred, pred * expected_scales[:, None, None]
        )
        for actual, original in zip(
            (input_shots, clean, pred, global_input), originals
        ):
            np.testing.assert_array_equal(actual, original)

    def test_rejects_unrelated_reference_and_invalid_inputs(self) -> None:
        input_shots = np.ones((2, 2, 3), dtype=np.float32)
        clean = input_shots * 0.5
        pred = input_shots * 0.4
        unrelated = input_shots.copy()
        unrelated[0, 0, 0] += 0.05

        with self.assertRaisesRegex(ValueError, "not proportional"):
            align_max_abs_shot_to_global(
                input_shots, clean, pred, unrelated
            )
        with self.assertRaisesRegex(ValueError, "zero-energy"):
            align_max_abs_shot_to_global(
                np.zeros_like(input_shots), clean, pred, input_shots
            )
        bad_reference = input_shots.copy()
        bad_reference[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            align_max_abs_shot_to_global(
                input_shots, clean, pred, bad_reference
            )


if __name__ == "__main__":
    unittest.main()
