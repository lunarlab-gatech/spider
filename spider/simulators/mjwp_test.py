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


def test_inflated_model_grows_only_paired_meshes_of_the_body(tmp_path):
    """_inflated_model grows the paired mesh of a body by the distance, once, at its scale."""
    # Covers: a wrong meshdir, a scale applied twice or not at all, an unpaired geom grown, a shared mesh or a moving body grown silently
    # Owner: mjwp._inflated_model (CPU, one small scene)
    import pytest

    from spider.config import Config
    from spider.simulators.mjwp import _inflated_model

    (tmp_path / "assets").mkdir()
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    faces = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
    (tmp_path / "assets" / "box.obj").write_text(
        "".join(f"v {x} {y} {z}\n" for x, y, z in corners)
        + "".join("f " + " ".join(map(str, f)) + "\n" for f in faces)
    )
    scene = """
<mujoco>
  <compiler meshdir="assets"/>
  <asset>{meshes}</asset>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="funnel">{geoms}</body>
  </worldbody>
  <contact><pair geom1="paired" geom2="floor"/></contact>
</mujoco>
"""
    meshes = '<mesh name="a" file="box.obj" scale="0.02 0.02 0.02"/>'
    meshes += '<mesh name="b" file="box.obj" scale="0.02 0.02 0.02"/>'
    geoms = '<geom name="paired" type="mesh" mesh="a"/><geom name="free" type="mesh" mesh="b"/>'
    path = tmp_path / "scene.xml"
    path.write_text(scene.format(meshes=meshes, geoms=geoms))
    config = Config(model_path=str(path), inflate_bodies=["funnel"], inflate_distance=0.003)

    model = _inflated_model(config)

    def extent(name):
        mesh = model.mesh(name)
        vert = model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + mesh.vertnum[0]]
        return vert.max(0) - vert.min(0)

    np.testing.assert_allclose(extent("a"), 0.02 + 2 * 0.003, atol=2 * 0.02 * 0.003)
    np.testing.assert_allclose(extent("b"), 0.02, atol=1e-9)

    path.write_text(scene.format(meshes=meshes, geoms=geoms + '<geom type="mesh" mesh="a"/>'))
    with pytest.raises(ValueError, match="used by 2 geoms"):
        _inflated_model(config)

    path.write_text(scene.format(meshes=meshes, geoms='<freejoint/>' + geoms))
    with pytest.raises(ValueError, match="has joints"):
        _inflated_model(config)


def test_robust_plan_settings_reject_partial_values():
    """check_robust_plan rejects inflate_* and ctrl_limit_* settings that would do nothing."""
    # Covers: a distance without bodies, bodies without a distance, actuators with a zero cost
    # Owner: config.check_robust_plan (pure)
    import pytest

    from spider.config import Config, check_robust_plan

    check_robust_plan(Config())
    check_robust_plan(Config(inflate_bodies=["b"], inflate_distance=0.003,
                             ctrl_limit_actuators=["a"], ctrl_limit_margin=0.1, ctrl_limit_rew_scale=0.1))
    for bad in (
        {"inflate_distance": 0.003},
        {"inflate_bodies": ["b"]},
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
