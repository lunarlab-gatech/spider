"""Checks on the hinge-angle conversion that shape_wrist hands to MJWP."""

import unittest

import numpy as np

from spider.preprocess.shape_wrist import euler_chain_to_matrix, matrix_to_euler_chain


class EulerChainTest(unittest.TestCase):
    """matrix_to_euler_chain keeps the hinge branch of the reference."""

    def test_rotations_return_to_the_reference_branch_and_wrap(self):
        """Angles from a rotation land back on the reference angles."""
        # Covers: MJWP tracks hinge angles, so a branch flip or 2 pi wrap spins the hand (a void run
        # on 2026-10-22: the IK reference sat at rot_y = 2.82 rad, past scipy's [-pi/2, pi/2] branch)
        # Owner: shape_wrist.matrix_to_euler_chain (pure)
        cases = {
            "rot_y past pi/2": [-0.539, 2.822, -0.716],
            "rot_x and rot_z past pi": [-3.941, 0.185, 4.300],
            "inside scipy's branch": [0.3, -0.4, 1.2],
        }
        for name, ref in cases.items():
            ref = np.array([ref])
            got = matrix_to_euler_chain(euler_chain_to_matrix(ref), ref)
            np.testing.assert_allclose(got, ref, atol=1e-9, err_msg=name)


if __name__ == "__main__":
    unittest.main()
