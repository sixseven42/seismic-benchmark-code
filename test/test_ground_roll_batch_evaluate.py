from __future__ import annotations

import unittest

from scripts.ground_roll_attenuation.batch_evaluate import (
    _physics_alignment_required,
)


class PhysicsAlignmentScopeTest(unittest.TestCase):
    def test_only_legacy_physics_unet_requires_alignment(self) -> None:
        shot_cfg = {
            "preprocess": {
                "normalize_mode": "max_abs",
                "normalize_scope": "shot",
            }
        }
        global_cfg = {
            "preprocess": {
                "normalize_mode": "max_abs",
                "normalize_scope": "global",
            }
        }
        skipped_cfg = {
            "preprocess": {
                "normalize_mode": "max_abs",
                "normalize_scope": "shot",
                "skip": ["normalize"],
            }
        }

        self.assertTrue(_physics_alignment_required("physics_unet", shot_cfg))
        self.assertFalse(_physics_alignment_required("physics_unet", global_cfg))
        self.assertFalse(_physics_alignment_required("physics_unet", skipped_cfg))
        for model_type in ("unet", "res_unet", "ddpm_unet", "physics_dnn"):
            self.assertFalse(_physics_alignment_required(model_type, shot_cfg))


if __name__ == "__main__":
    unittest.main()
