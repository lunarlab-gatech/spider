"""Checks on the mink limits that ik_fast relies on, on the side planes it builds on them, and on its frame-loop step."""

import unittest

import mink
import mujoco
import numpy as np

from spider.preprocess.ik_fast import (
    PenetrationTask,
    SidePlanes,
    hold_sides,
    improving_scale,
    step_speeds,
)

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


# A hinge and a slide, as a hand has, and a free body, as an object is.
_JOINTS = """
<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="turn" type="hinge" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <body name="tip" pos="0.1 0 0">
        <joint name="reach" type="slide" axis="1 0 0" range="-1 1"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <site name="tip"/>
      </body>
    </body>
    <body name="object"><freejoint name="object"/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body>
  </worldbody>
</mujoco>
"""


class FrameStepTest(unittest.TestCase):
    """The step of the IK frame loop: its largest joint change, and the part of it that is taken."""

    def test_step_moves_a_hinge_and_a_slide_by_at_most_their_bound(self):
        """One IK step toward a far target turns a hinge 0.29 deg and moves a slide 1 mm, at any dt."""
        # Covers: step_speeds naming no joint (joint types read as numpy ints matched no enum key) or giving a
        # speed that mink scales with dt, so the frame loop takes whole steps again: at 5 to 10 deg of wrist turn
        # per step the collision row is 4.4 mm wrong and fingertips of airknife recording 3 went through the glass
        # Owner: ik_fast.step_speeds with mink.VelocityLimit (synthetic model)
        model = mujoco.MjModel.from_xml_string(_JOINTS)
        configuration = mink.Configuration(model)
        task = mink.FrameTask("tip", "site", position_cost=1.0, orientation_cost=0.0)
        task.set_target(mink.SE3.from_translation(np.array([0.5, 0.5, 0.0])))  # 45 deg round and 0.6 m out
        for dt in (0.0005, 0.02):
            configuration.update(model.qpos0)
            limit = mink.VelocityLimit(model, step_speeds(model, dt))
            step = mink.solve_ik(configuration, [task], dt, "daqp", limits=[limit]) * dt
            np.testing.assert_allclose(step[:2], [np.radians(0.29), 0.001], rtol=1e-6, err_msg=f"dt {dt}")

    def test_improving_scale_is_the_largest_part_of_the_step_that_lowers_the_cost(self):
        """The step is halved until the cost falls, down to MIN_STEP_SCALE, and dropped when no part lowers it."""
        # Covers: improving_scale giving a part of the step that does not lower the cost (the solver then moves
        # back on the next step and the frame does not settle), or a part shorter than 1/16
        # Owner: ik_fast.improving_scale (pure)
        cases = {  # cost after the part `s` of a step whose cost before is cost(0); the largest part that lowers it
            "whole step lowers the cost": (lambda s: 1.0 - s, 1.0),
            "minimum at 0.3 of the step": (lambda s: (s - 0.3) ** 2, 0.5),  # 0.49 at 1, 0.04 at 1/2, against 0.09
            "minimum at 0.04 of the step": (lambda s: (s - 0.04) ** 2, 1 / 16),  # first under 0.0016 at 1/16
            "minimum at 0.02 of the step": (lambda s: (s - 0.02) ** 2, 0.0),  # 0.0018 at 1/16, against 0.0004
            "no part lowers the cost": (lambda s: 1.0 + s, 0.0),
            "equal cost is no gain": (lambda s: 1.0, 0.0),
        }
        for name, (cost, want) in cases.items():
            self.assertEqual(improving_scale(cost(0.0), cost), want, name)


class HoldSidesTest(unittest.TestCase):
    """hold_sides on the planes of one fingertip."""

    def test_the_side_with_the_most_frames_holds_over_each_near_span(self):
        """Frames of the other side, and frames inside the part, take the plane of the nearest frame of the held side."""
        # Covers: the measured left index of airknife recording 3 reads over the glass for 26 frames of a grasp
        # from under it; with a plane per frame the IK fingertip went through the glass and the left hand clamped
        # it. Also: the first side held when most frames have the other, the turn of the plane at an edge taken
        # for a change of side, a span decided by another span, and one span across two bodies
        # Owner: ik_fast.hold_sides (pure)
        down, up, out = [0.0, 0.0, -1.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]
        slant = [-np.sqrt(0.5), 0.0, -np.sqrt(0.5)]  # between out and down: the plane turns about an edge
        rise = [-np.sqrt(0.5), 0.0, np.sqrt(0.5)]  # between up and out
        none = [0.0, 0.0, 0.0]
        cases = {  # per frame: (near, body or None for no plane, normal); then the frame whose plane each frame gets
            "one frame reads through": ([(1, 1, down), (1, 1, down), (1, 1, up), (1, 1, down), (1, 1, down)], [0, 1, 1, 3, 4]),
            "frames inside the part": ([(1, 1, down), (1, 1, down), (1, None, none), (1, None, none), (1, 1, down)], [0, 1, 1, 4, 4]),
            "most frames on the second side": ([(1, 1, up), (1, 1, down), (1, 1, down), (1, 1, down)], [1, 1, 2, 3]),
            "turn about an edge": ([(1, 1, out), (1, 1, slant), (1, 1, down), (1, 1, down)], [0, 1, 2, 3]),
            "half turn in 45 deg steps": ([(1, 1, up), (1, 1, rise), (1, 1, out), (1, 1, slant), (1, 1, down)], [0, 1, 2, 3, 4]),
            "tie: the first side": ([(1, 1, down), (1, 1, down), (1, 1, up), (1, 1, up)], [0, 1, 1, 1]),
            "two spans": ([(1, 1, down), (1, 1, down), (0, None, none), (1, 1, up), (1, 1, up), (1, 1, down)], [0, 1, None, 3, 4, 4]),
            "two bodies": ([(1, 1, down), (1, 1, down), (1, 2, up), (1, 2, up)], [0, 1, 2, 3]),
        }
        for name, (frames, sources) in cases.items():
            near = np.array([f[0] for f in frames], dtype=bool)
            active = np.array([f[1] is not None for f in frames])
            body = np.array([f[1] or 0 for f in frames])
            point = np.array([[float(t), 0.0, 0.0] if a else none for t, a in enumerate(active)])  # the frame it is from
            normal = np.array([f[2] for f in frames])
            held = hold_sides(near, active, body, point, normal)
            want = [t for t in sources if t is not None]
            np.testing.assert_array_equal(held[0], [t is not None for t in sources], err_msg=name)
            np.testing.assert_array_equal(held[2][held[0], 0], want, err_msg=name)
            np.testing.assert_array_equal(held[1][held[0]], body[want], err_msg=name)
            np.testing.assert_array_equal(held[3][held[0]], normal[want], err_msg=name)
            self.assertEqual(held[4], sum(t is not None and t != i for i, t in enumerate(sources)), name)


# A 4 mm plate, 100 mm wide, that moves, and a fingertip capsule.
_FREE_PLATE = """
<mujoco>
  <worldbody>
    <body name="stage"><freejoint/><geom name="plate" type="box" size="0.05 0.05 0.002"/></body>
    <body name="tip" mocap="true"><geom name="tip" type="capsule" size="0.005 0.01"/></body>
  </worldbody>
</mujoco>
"""

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
            path = np.linspace(-0.08, -0.02, 60)
            if with_planes:
                human_tips = np.stack([path, np.zeros(60), np.full(60, -0.008)], axis=1)[:, None]
                sides.plan(model, configuration.data, human_tips, lambda t: None)
            for t, x in enumerate(path):
                human.set_target(mink.SE3.from_translation(np.array([x, 0.0, -0.008])))
                lifted.set_target(mink.SE3.from_translation(np.array([x, 0.0, 0.03])))
                if with_planes:
                    sides.place(model, configuration.data, t)
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

    def test_plan_reads_each_frame_at_the_object_pose_of_that_frame(self):
        """A human fingertip that moves with the plate has the same plane in the plate frame on every frame."""
        # Covers: plan reading every frame at one object pose, so on a layout that moves the stage the plane of a
        # frame is lost or saved off the glass, and IK and the MJWP side reward hold the fingertip at that plane
        # Owner: ik_fast.SidePlanes.plan and place (synthetic model)
        spec = mujoco.MjSpec.from_string(_FREE_PLATE)
        SidePlanes.add_geoms(spec, 1)
        model = spec.compile()
        data = mujoco.MjData(model)
        sides = SidePlanes(model, [model.geom("tip").id], [model.geom("plate").id])
        quat = np.array([0.9, 0.2, -0.3, 0.25]) / np.linalg.norm([0.9, 0.2, -0.3, 0.25])
        poses = np.array([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.1, -0.2, 0.3, *quat]])  # frame 0, frame 1
        turned = np.zeros(9)
        mujoco.mju_quat2Mat(turned, quat)
        rotations = [np.eye(3), turned.reshape(3, 3)]
        local = np.array([0.04, 0.0, -0.012])  # 10 mm under the plate bottom, 10 mm from its edge
        human_tips = np.array([p[:3] + r @ local for p, r in zip(poses, rotations, strict=True)])[:, None]

        def set_frame(t):
            data.qpos[:] = poses[t]

        self.assertEqual(sides.plan(model, data, human_tips, set_frame), [0])
        plane_mocap = model.body_mocapid[model.geom_bodyid[model.geom("side_plane_0").id]]
        for t, rotation in enumerate(rotations):
            set_frame(t)
            sides.place(model, data, t)
            self.assertTrue(sides.active[0], f"frame {t}")
            np.testing.assert_allclose(sides.point[0], [0.04, 0.0, -0.002], atol=1e-9, err_msg=f"frame {t}")
            np.testing.assert_allclose(sides.normal[0], [0.0, 0.0, -1.0], atol=1e-9, err_msg=f"frame {t}")
            # In the world the plane is on the plate bottom, 10 mm from the human fingertip along the plate normal.
            want = human_tips[t, 0] + 0.01 * rotation[:, 2]
            np.testing.assert_allclose(data.mocap_pos[plane_mocap], want, atol=1e-9, err_msg=f"frame {t}, world")


if __name__ == "__main__":
    unittest.main()
