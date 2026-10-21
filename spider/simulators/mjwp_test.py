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


def test_optimize_once_keeps_the_first_group_and_scores_the_worst():
    """optimize_once scores the worst group and records only the first."""
    # Covers: an object_offsets group reaching the kept plan (recorded states, snapshot) or a score other than the worst group
    # Owner: sampling_fast.optimize_once with a stub rollout (CPU)
    from unittest import mock

    from spider.config import Config
    from spider.optimizers import sampling_fast

    config = Config(num_samples=3, device="cpu", viewer="", use_torch_compile=False)
    group_rews = [torch.tensor([1.0, 5.0, 3.0]), torch.tensor([2.0, 0.0, 2.5])]
    recorded = []

    def rollout(config, env, ctrls, ref_slice, env_param, record_states=False):
        group = env_param["group"]
        recorded.append(record_states)
        info = {"trace": torch.zeros(3, 1, 1, 3)}
        if record_states:
            info["recorded_qpos"] = 10.0 * group + torch.arange(3.0)[:, None, None]
            info["recorded_qvel"] = info["recorded_qpos"]
            info["state_snapshot"] = f"group {group}"
        return ctrls, group_rews[group], torch.zeros(3, dtype=torch.bool), info

    optimize_once = sampling_fast.make_optimize_once_fn_fast(rollout)
    with mock.patch.object(sampling_fast, "sample_ctrls", lambda c, ctrls, p: ctrls.repeat(3, 1, 1)):
        _, _, info = optimize_once(
            config, None, torch.zeros(2, 1), (), [{"group": 0}, {"group": 1}], None, select_best=True
        )
    # Worst per sample: [1, 0, 2.5], so sample 2 wins, though group 0 alone picks sample 1.
    assert info["best_idx"] == 2
    assert recorded == [True, False]
    assert info["recorded_qpos"].item() == 2.0 and info["state_snapshot"] == "group 0"


@unittest.skipUnless(torch.cuda.is_available(), "MJWP needs CUDA")
class RobustPlanTest(unittest.TestCase):
    """Object offsets and the control limit cost on MJWP worlds."""

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

    def test_object_offset_moves_and_turns_the_free_object_only(self):
        """object_offset moves and turns the free object about world z."""
        # Covers: object_offsets writing the wrong qpos (hand joints, quaternion order) or a yaw about the wrong axis
        # Owner: mjwp.load_env_params (GPU, one small model)
        import warp as wp

        config, env, mjwp = self._env()
        qpos = wp.to_torch(env.data_wp.qpos)
        h = np.sqrt(0.5)
        qpos[:, 4:8] = torch.tensor([h, h, 0.0, 0.0])  # object turned 90 deg about world x
        before = qpos.cpu().numpy()
        mjwp.load_env_params(config, env, {"object_offset": (0.01, -0.02, np.pi / 2)})
        after = wp.to_torch(env.data_wp.qpos).cpu().numpy()
        want = before.copy()
        want[:, 1] += 0.01
        want[:, 2] -= 0.02
        # 90 deg about world z after 90 deg about x: (h, 0, 0, h) * (h, h, 0, 0).
        want[:, 4:8] = [0.5, 0.5, 0.5, 0.5]
        np.testing.assert_allclose(after, want, atol=1e-6)

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
