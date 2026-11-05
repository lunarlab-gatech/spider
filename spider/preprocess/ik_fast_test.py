"""Checks on the mink limits that ik_fast relies on, and on the side planes it builds on them."""

import unittest

import mink
import mujoco
import numpy as np

from spider.preprocess.ik_fast import PenetrationTask, SidePlanes

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


# A 4 mm plate, 100 mm wide, and a 5 mm fingertip sphere on x and z slides.
_PLATE = """
<mujoco>
  <worldbody>
    <geom name="plate" type="box" size="0.05 0.05 0.002"/>
    <body name="tip">
      <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
      <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <geom name="tip" type="sphere" size="0.005"/>
      <site name="tip"/>
    </body>
  </worldbody>
</mujoco>
"""


class SidePlanesTest(unittest.TestCase):
    """SidePlanes with the limit and push-out task that ik_fast builds on them."""

    def test_fingertip_goes_around_the_edge_to_the_human_side(self):
        """A fingertip held high goes under a thin plate when the human fingertip does."""
        # Covers: the collision limit keeps a fingertip on the side of a thin part it reaches first; with the wrist
        # lifted by the hand-floor pairs, the right index and middle of recording 3 stayed over the stage glass
        # while the human fingers were under it. A plane facing away from the human fingertip puts them over too.
        # Owner: ik_fast.SidePlanes (synthetic model, mink)
        spec = mujoco.MjSpec.from_string(_PLATE)
        SidePlanes.add_geoms(spec, 1)
        model = spec.compile()
        plate, tip = model.geom("plate").id, model.geom("tip").id
        for name, with_planes, under in (("collision limit only", False, False), ("side planes", True, True)):
            configuration = mink.Configuration(model)
            configuration.update(np.array([-0.08, 0.012]))
            sides = SidePlanes(model, [tip], [plate])
            limits = [mink.CollisionAvoidanceLimit(model, [(["tip"], ["plate"])], minimum_distance_from_collisions=0.0,
                                                   collision_detection_distance=0.02)]
            if with_planes:
                limits.append(mink.CollisionAvoidanceLimit(model, sides.pairs, minimum_distance_from_collisions=0.0,
                                                           collision_detection_distance=0.02))
            tasks = [PenetrationTask(limit, 0.0) for limit in limits]
            # The human fingertip passes the edge 8 mm under the plate mid-plane; a second target 30 mm over it
            # stands in for the lifted wrist and holds the robot fingertip 11 mm over the mid-plane.
            human, lifted = (mink.FrameTask("tip", "site", position_cost=1.0, orientation_cost=0.0) for _ in range(2))
            tasks += [human, lifted]
            for x in np.linspace(-0.08, -0.02, 60):
                human.set_target(mink.SE3.from_translation(np.array([x, 0.0, -0.008])))
                lifted.set_target(mink.SE3.from_translation(np.array([x, 0.0, 0.03])))
                if with_planes:
                    sides.update(model, configuration.data, np.array([[x, 0.0, -0.008]]))
                    configuration.update()
                for _ in range(10):
                    v = mink.solve_ik(configuration, tasks, 0.001, "daqp", damping=1e-5, limits=limits)
                    configuration.integrate_inplace(v, 0.001)
            x, z = configuration.q
            self.assertAlmostEqual(x, -0.02, delta=0.003, msg=name)
            # Under: the sphere below the plate bottom (z -0.002); over: above its top (z 0.002).
            self.assertTrue(z < -0.007 + 1e-3 if under else z > 0.007 - 1e-3, f"{name}: z {z:.4f}")
            dist = mujoco.mj_geomDistance(model, configuration.data, tip, plate, 0.02, np.zeros(6))
            self.assertGreater(dist, -1e-3, f"{name}: tip {dist * 1e3:.1f} mm inside the plate")


if __name__ == "__main__":
    unittest.main()
