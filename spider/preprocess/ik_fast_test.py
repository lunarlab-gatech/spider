"""Checks on the IK limits and tasks of ik_fast."""

import unittest

import mink
import mujoco
import numpy as np

from spider.preprocess.ik_fast import CollisionDisplacementLimit

# A 5 mm sphere on an x slide, its surface 10 mm from a box face at x = 0. The
# slide sits on a z slide, as a finger sits on the wrist joints: mink 1.1.0
# skips a pair whose bodies hang from the same parent.
_SLIDER = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="0.05 0 0" size="0.05 0.05 0.05"/>
    <body name="base">
      <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body name="tip" pos="-0.015 0 0">
        <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
        <geom name="tip" type="sphere" size="0.005"/>
        <site name="tip"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


class CollisionLimitTest(unittest.TestCase):
    def test_step_closes_at_most_gain_times_the_gap(self):
        # Covers: at sim_dt 0.001 mink 1.1.0's limit let a fingertip cross its whole gap into the object in one step
        # Owner: ik_fast collision limit (pure)
        model = mujoco.MjModel.from_xml_string(_SLIDER)
        configuration = mink.Configuration(model)
        task = mink.FrameTask("tip", "site", position_cost=1.0, orientation_cost=0.0)
        task.set_target(mink.SE3.from_translation(np.array([0.05, 0.0, 0.0])))  # 50 mm into the box
        limit = CollisionDisplacementLimit(
            model,
            [(["tip"], ["wall"])],
            gain=0.85,
            minimum_distance_from_collisions=0.0,
            collision_detection_distance=0.02,
        )
        for dt in (0.001, 0.02):
            configuration.update(np.zeros(2))
            v = mink.solve_ik(configuration, [task], dt, "daqp", limits=[limit])
            # Gap 10 mm, gain 0.85: the step closes 8.5 mm whatever dt is.
            np.testing.assert_allclose(v * dt, [0.0, 0.0085], atol=1e-6, err_msg=f"dt {dt}")


if __name__ == "__main__":
    unittest.main()
