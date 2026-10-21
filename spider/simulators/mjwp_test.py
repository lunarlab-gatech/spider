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


def test_inflate_hull_moves_every_face_out_by_the_distance():
    """inflate_hull grows a box by the distance in every direction."""
    # Covers: an inflation that shrinks the hull, misses a direction, or grows it by more than the distance
    # Owner: mjwp.inflate_hull (pure geometry)
    from spider.simulators.mjwp import inflate_hull

    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 2) for z in (0, 3)], dtype=float)
    grown = inflate_hull(cube, 0.01)
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(1000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # The support function along every direction grows by 0.98 to 1.0 times the distance.
    growth = (grown @ directions.T).max(0) - (cube @ directions.T).max(0)
    assert growth.min() >= 0.98 * 0.01 and growth.max() <= 0.01 + 1e-12


@unittest.skipUnless(torch.cuda.is_available(), "MJWP needs CUDA")
class RobustPlanTest(unittest.TestCase):
    """The control limit cost on MJWP worlds."""

    def _env(self, ctrlrange=""):
        """Two MJWP worlds of _SCENE, with an optional ctrlrange on the actuator."""
        from spider.config import Config
        from spider.simulators import mjwp

        scene = _SCENE.replace('kp="10"/>', f'kp="10" {ctrlrange}/>')
        with tempfile.NamedTemporaryFile("w", suffix=".xml") as f:
            f.write(scene)
            f.flush()
            config = Config(model_path=f.name, num_samples=2, device="cuda:0", viewer="")
            config.embodiment_type = "right"
            config.nq, config.nv, config.nu, config.nq_obj = 8, 7, 1, 7
            qpos = torch.zeros(2, 8, device="cuda:0")
            qpos[:, 3], qpos[:, 4] = 0.019, 1.0
            zeros = torch.zeros(2, 7, device="cuda:0")
            env = mjwp.setup_env(config, (qpos, zeros, zeros[:, :1], None, None))
            if ctrlrange:
                config.ctrl_limit_ids = [0]
                config.ctrl_limit_low, config.ctrl_limit_high = [-0.1], [0.1]
        return config, env, mjwp

    def test_ctrl_limit_cost_charges_the_distance_inside_the_margin(self):
        """get_reward charges controls closer than the margin to ctrlrange."""
        # Covers: a control limit cost with the wrong sign, on clamped instead of sampled controls, or outside the margin
        # Owner: mjwp.get_reward (GPU, one small model)
        import warp as wp

        config, env, mjwp = self._env('ctrlrange="-0.1 0.1"')
        config.ctrl_limit_margin, config.ctrl_limit_rew_scale = 0.05, 2.0
        ref = (torch.zeros(8, device="cuda:0"), torch.zeros(7, device="cuda:0"), None, None, None)
        # Room to the limit: 0.1 (none charged), 0.02 (0.03 short), -0.1 past it (0.15 short).
        for ctrl, cost in ((0.0, 0.0), (0.08, 0.06), (0.2, 0.3)):
            wp.to_torch(env.data_wp.ctrl)[:] = ctrl
            _, info = mjwp.get_reward(config, env, ref)
            np.testing.assert_allclose(info["ctrl_limit_rew"].cpu().numpy(), -cost, atol=1e-6)
