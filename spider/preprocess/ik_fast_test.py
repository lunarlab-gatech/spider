# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Checks for the ik_fast contact target blend."""

import unittest

import numpy as np

from spider.preprocess.ik_fast import contact_blend_weights


class ContactBlendTest(unittest.TestCase):
    """The blend must not jump the fingertip target at a contact switch."""

    def test_blend_weight_ramps_over_each_span(self):
        """Weights step by 1 / (ramp_frames + 1) at each span end, 0 outside."""
        # Covers: a weight that jumps at a span end swaps the target by 1-3 cm in one frame
        # Owner: ik_fast contact_blend_weights (pure)
        cases = {
            "long span reaches 1": (
                [0, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1 / 3, 2 / 3, 1, 1, 1, 2 / 3, 1 / 3, 0],
            ),
            "short span peaks below 1": ([0, 1, 1, 1, 0], [0, 1 / 3, 2 / 3, 1 / 3, 0]),
            "spans at the clip ends": ([1, 1, 0, 1], [1 / 3, 1 / 3, 0, 1 / 3]),
        }
        for name, (contact, expected) in cases.items():
            with self.subTest(name):
                weights = contact_blend_weights(np.array(contact, float)[:, None], 2)
                np.testing.assert_allclose(weights[:, 0], expected, err_msg=name)


if __name__ == "__main__":
    unittest.main()
