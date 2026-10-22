"""Checks on the mink limits that ik_fast relies on."""

import unittest

import mink
import mujoco
import numpy as np

# A 5 mm sphere on an x slide, its surface 10 mm from a box face at x = 0 and
# from a floor plane at x = 0 that faces -x. The slide sits on a z slide, as a
# finger sits on the wrist joints; the floor sits in the world body, as the
# scene floor does.
_SLIDER = """
<mujoco>
  <worldbody>
    <geom name="wall" type="box" pos="0.05 0 0" size="0.05 0.05 0.05"/>
    <geom name="floor" type="plane" size="0 0 0.05" zaxis="-1 0 0"/>
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
    """mink.CollisionAvoidanceLimit as ik_fast builds it."""

    def test_step_closes_at_most_gain_times_the_gap(self):
        """One IK step closes gain times the gap, at any dt, against an object or the floor."""
        # Covers: mink 1.1.0 to 1.3.0 bound the step by gain * gap / dt, so at sim_dt 0.001 a fingertip crossed its whole gap into the object in one step; the same bound holds against a plane in the world body (the floor)
        # Owner: the mink pin in pyproject.toml (ik_fast collision limit)
        model = mujoco.MjModel.from_xml_string(_SLIDER)
        configuration = mink.Configuration(model)
        task = mink.FrameTask("tip", "site", position_cost=1.0, orientation_cost=0.0)
        task.set_target(mink.SE3.from_translation(np.array([0.05, 0.0, 0.0])))  # 50 mm into the box
        for other in ("wall", "floor"):
            limit = mink.CollisionAvoidanceLimit(
                model,
                [(["tip"], [other])],
                gain=0.85,
                minimum_distance_from_collisions=0.0,
                collision_detection_distance=0.02,
            )
            for dt in (0.001, 0.02):
                configuration.update(np.zeros(2))
                v = mink.solve_ik(configuration, [task], dt, "daqp", limits=[limit])
                # Gap 10 mm, gain 0.85: the step closes 8.5 mm whatever dt is.
                np.testing.assert_allclose(v * dt, [0.0, 0.0085], atol=1e-6, err_msg=f"{other}, dt {dt}")


if __name__ == "__main__":
    unittest.main()
