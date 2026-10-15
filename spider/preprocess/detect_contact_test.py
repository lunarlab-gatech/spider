# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the license in the repository root.

"""Checks for the detect_contact surface normal."""

import unittest

import numpy as np

from spider.preprocess.detect_contact import outward_normal


class OutwardNormalTest(unittest.TestCase):
    """The contact normal must point out of the object."""

    def test_normal_points_out_of_the_object(self):
        """The fromto difference flips when the fingertip penetrates."""
        # Covers: an inward normal puts the IK fingertip target inside the object
        # Owner: detect_contact outward_normal (pure)
        # fromto is (fingertip point, object point) as mj_geomDistance(tip, object)
        # returns it; the object surface faces +z or +x at the object point.
        cases = {
            "apart": ([0, 0, 0.07, 0, 0, 0.05], 0.02, [0, 0, 1]),
            "penetrating": ([0, 0, 0.045, 0, 0, 0.05], -0.005, [0, 0, 1]),
            "penetrating side": ([0.048, 0, 0, 0.05, 0, 0], -0.002, [1, 0, 0]),
            "touching": ([0, 0, 0.05, 0, 0, 0.05], 0.0, [0, 0, 0]),
        }
        for name, (fromto, dist, expected) in cases.items():
            with self.subTest(name):
                np.testing.assert_allclose(
                    outward_normal(np.array(fromto, float), dist),
                    expected,
                    err_msg=name,
                )


if __name__ == "__main__":
    unittest.main()
