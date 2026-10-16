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
