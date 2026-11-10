# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import types
import unittest

import mujoco
import numpy as np
import torch


def test_imports():
    import spider
    from spider.simulators import mjwp
    assert isinstance(spider.ROOT, str)
    assert isinstance(mjwp, types.ModuleType)


# A slide-jointed box above a free box resting on the floor, so the state has
# contacts, a free joint and a mocap body.
_SCENE = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="hand" pos="0 0 0.3">
      <joint name="hx" type="slide" axis="1 0 0"/>
      <geom name="palm" type="box" size="0.03 0.03 0.03"/>
    </body>
    <body name="right_object" pos="0 0 0.019">
      <freejoint/>
      <geom name="obj" type="box" size="0.02 0.02 0.02"/>
    </body>
    <body name="mocap" mocap="true" pos="0.2 0 0.2">
      <geom type="sphere" size="0.01" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator><position joint="hx" kp="10"/></actuator>
</mujoco>
"""


@unittest.skipUnless(torch.cuda.is_available(), "MJWP needs CUDA")
class StateCopyTest(unittest.TestCase):
    """Copies of MJWP state between MuJoCo, torch and Warp."""

    def test_sync_and_copy_keep_vector_and_matrix_fields(self):
        """sync_env_mujoco and copy_sample_state write typed fields to every world."""
        # Covers: wp.copy rejects a float32 tensor into a vec3, quat, mat33 or spatial_vector array, so a local retry (sync_env_mujoco) or a resample (copy_sample_state) stopped MJWP
        # Owner: mjwp state copies (GPU, one small model)
        import warp as wp

        from spider.config import Config
        from spider.simulators import mjwp

        with tempfile.NamedTemporaryFile("w", suffix=".xml") as f:
            f.write(_SCENE)
            f.flush()
            config = Config(model_path=f.name, num_samples=3, device="cuda:0", viewer="")
            config.embodiment_type = "right"
            config.nq, config.nv, config.nu, config.nq_obj = 8, 7, 1, 7
            qpos = torch.zeros(2, 8, device="cuda:0")
            qpos[:, 3], qpos[:, 4] = 0.019, 1.0
            zeros = torch.zeros(2, 7, device="cuda:0")
            env = mjwp.setup_env(config, (qpos, zeros, zeros[:, :1], None, None))

        mj_data = env.data_cpu
        mujoco.mj_forward(env.model_cpu, mj_data)
        mj_data.xfrc_applied[2] = [1, 2, 3, 4, 5, 6]
        mjwp.sync_env_mujoco(config, env, mj_data)
        fields = ["qpos", "xfrc_applied", "mocap_quat", "xmat", "geom_xmat", "energy"]
        for name in fields:
            got = wp.to_torch(getattr(env.data_wp, name)).cpu().numpy()
            want = np.asarray(getattr(mj_data, name), dtype=np.float32)
            for world in range(config.num_samples):
                np.testing.assert_allclose(got[world].reshape(want.shape), want, atol=1e-6, err_msg=name)

        xpos = wp.to_torch(env.data_wp.xpos)
        xpos[0] += 1.0
        mjwp.copy_sample_state(config, env, torch.tensor([0]), torch.tensor([2]))
        np.testing.assert_allclose(xpos[2].cpu().numpy(), xpos[0].cpu().numpy())


def test_robust_plan_settings_reject_partial_values():
    """check_robust_plan rejects ctrl_limit_* settings that would do nothing."""
    # Covers: actuators with a zero cost, a margin and a cost without actuators
    # Owner: config.check_robust_plan (pure)
    import pytest

    from spider.config import Config, check_robust_plan

    check_robust_plan(Config())
    check_robust_plan(Config(ctrl_limit_actuators=["a"], ctrl_limit_margin=0.1, ctrl_limit_rew_scale=0.1))
    for bad in (
        {"ctrl_limit_actuators": ["a"], "ctrl_limit_margin": 0.1},
        {"ctrl_limit_margin": 0.1, "ctrl_limit_rew_scale": 0.1},
    ):
        with pytest.raises(ValueError, match="set "):
            check_robust_plan(Config(**bad))


@unittest.skipUnless(torch.cuda.is_available(), "MJWP needs CUDA")
class RobustPlanTest(unittest.TestCase):
    """The control limit cost on MJWP worlds."""

    def test_ctrl_limit_cost_charges_the_distance_inside_the_margin(self):
        """get_reward charges controls closer than the margin to ctrlrange."""
        # Covers: a control limit cost with the wrong sign, on clamped instead of sampled controls, or outside the margin
        # Owner: config.resolve_ctrl_limits and mjwp.get_reward (GPU, one small model)
        from spider.config import Config, resolve_ctrl_limits
        from spider.simulators import mjwp

        scene = _SCENE.replace('kp="10"/>', 'kp="10" name="act" ctrlrange="-0.1 0.1"/>')
        with tempfile.NamedTemporaryFile("w", suffix=".xml") as f:
            f.write(scene)
            f.flush()
            config = Config(model_path=f.name, num_samples=2, device="cuda:0", viewer="",
                            ctrl_limit_actuators=["act"], ctrl_limit_margin=0.05, ctrl_limit_rew_scale=2.0)
            config.embodiment_type = "right"
            config.nq, config.nv, config.nu, config.nq_obj = 8, 7, 1, 7
            qpos = torch.zeros(2, 8, device="cuda:0")
            qpos[:, 3], qpos[:, 4] = 0.019, 1.0
            zeros = torch.zeros(2, 7, device="cuda:0")
            env = mjwp.setup_env(config, (qpos, zeros, zeros[:, :1], None, None))
        resolve_ctrl_limits(config, env.model_cpu)
        ref = (torch.zeros(8, device="cuda:0"), torch.zeros(7, device="cuda:0"), None, None, None)
        # Room to the limit: 0.1 (none charged), 0.02 (0.03 short), -0.1 past it (0.15 short).
        for ctrl, cost in ((0.0, 0.0), (0.08, 0.06), (0.2, 0.3)):
            mjwp.step_env(config, env, torch.full((2, 1), ctrl, device="cuda:0"))
            _, info = mjwp.get_reward(config, env, ref)
            np.testing.assert_allclose(info["ctrl_limit_rew"].cpu().numpy(), -cost, atol=1e-6)


# A 4 mm plate on a free body and a capsule fingertip (5 mm radius, 10 mm half length) on a mocap body.
_STAGE = """
<mujoco>
  <worldbody>
    <body name="stage"><freejoint/><geom name="plate" type="box" size="0.05 0.05 0.002"/></body>
    <body name="tip" mocap="true"><geom name="tip" type="capsule" size="0.005 0.01"/></body>
  </worldbody>
</mujoco>
"""


def test_side_plane_distance_rebuilds_the_ik_side_plane_on_the_moving_body():
    """side_plane_distance gives MuJoCo's signed distance to the IK side plane, wherever the body goes."""
    # Covers: the MJWP side reward rebuilding the plane of ik_fast.SidePlanes.reference_planes in the wrong frame
    # (rotation transposed, plane left in the world while the stage moves) or with the wrong capsule term, so it charges
    # fingers on the human side of the glass or misses fingers on top of it; and charging a plan where IK itself left
    # the finger past the plane, so MJWP leaves its reference
    # Owner: mjwp.side_plane_distance with ik_fast.SidePlanes (CPU, synthetic model)
    from spider.preprocess.ik_fast import SidePlanes
    from spider.simulators.mjwp import side_plane_distance

    spec = mujoco.MjSpec.from_string(_STAGE)
    SidePlanes.add_geoms(spec, 1)
    model = spec.compile()
    data = mujoco.MjData(model)
    tip, plane = model.geom("tip").id, model.geom("side_plane_0").id
    stage, tip_mocap = model.body("stage").id, model.body("tip").mocapid[0]

    def place(pose, local_pos, local_quat):  # stage pose (x, y, z, qw, qx, qy, qz), tip pose in the stage frame
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        data.mocap_pos[tip_mocap] = pose[:3] + data.xmat[stage].reshape(3, 3) @ local_pos
        mujoco.mju_mulQuat(data.mocap_quat[tip_mocap], pose[3:], local_quat)
        mujoco.mj_kinematics(model, data)

    def distance(planes):
        arrays = [torch.from_numpy(a.reshape(1, *a.shape)) for a in (data.geom_xpos, data.geom_xmat, data.xpos, data.xmat)]
        return side_plane_distance(model, [tip], *arrays, planes).item()

    def solved(local_pos, local_quat):
        """reference_planes of a frame that IK solved with the tip at this pose, packed as io.load_side_planes."""
        place(turned, np.array(local_pos), local_quat)
        active, body, point, normal = sides.reference_planes(model, data)
        return torch.tensor([[active[0], body[0], *point[0], *normal[0]]], dtype=torch.float64)

    turned = np.array([0.1, -0.2, 0.3, *(np.array([0.9, 0.2, -0.3, 0.25]) / np.linalg.norm([0.9, 0.2, -0.3, 0.25]))])
    moved = np.array([0.4, 0.1, 0.05, *(np.array([0.3, 0.8, 0.1, -0.5]) / np.linalg.norm([0.3, 0.8, 0.1, -0.5]))])
    upright, tilted, flat = [1.0, 0, 0, 0], [0.924, 0.383, 0, 0], [0.707, 0, 0.707, 0]
    # The human fingertip is 10 mm under the plate near its edge: the plane is on the plate bottom, facing down.
    place(turned, np.zeros(3), upright)
    sides = SidePlanes(model, [tip], [model.geom("plate").id])
    sides.update(model, data, (turned[:3] + data.xmat[stage].reshape(3, 3) @ [0.04, 0.0, -0.012])[None])
    planes = solved([0.02, 0.01, -0.03], upright)  # IK left the tip under the plate
    cases = {  # tip pose in the stage frame; distance to the IK plane geom by MuJoCo's capsule-plane collision
        "under, upright": ([0.02, 0.01, -0.03], upright),
        "under, tilted, 2 mm in": ([0.03, 0.0, -0.012], tilted),
        "on top, flat": ([0.0, 0.0, 0.008], flat),
    }
    for name, (local_pos, local_quat) in cases.items():
        place(turned, np.array(local_pos), local_quat)
        want = mujoco.mj_geomDistance(model, data, tip, plane, 1.0, np.zeros(6))
        np.testing.assert_allclose(distance(planes), want, atol=1e-9, err_msg=name)
        place(moved, np.array(local_pos), local_quat)  # the IK plane geom stays; the side plane moves with the stage
        np.testing.assert_allclose(distance(planes), want, atol=1e-9, err_msg=f"{name}, stage moved")
    assert distance(planes) < -0.01  # on top of the plate: past the plane by the plate and the capsule radius
    # IK left the tip on top, 15 mm past the plane: the plane moves 15 mm up (to z 0.013), and a plan pays only for going
    # further; under the plate it pays nothing.
    planes = solved([0.0, 0.0, 0.008], flat)
    for local_z, want in ((0.008, 0.0), (0.013, -0.005), (-0.03, 0.038)):
        place(moved, np.array([0.0, 0.0, local_z]), flat)
        np.testing.assert_allclose(distance(planes), want, atol=1e-9, err_msg=f"relaxed plane, tip at z {local_z}")
