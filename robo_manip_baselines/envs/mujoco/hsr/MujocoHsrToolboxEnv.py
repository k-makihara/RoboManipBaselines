from os import path

import mujoco
import numpy as np

from .MujocoHsrPaPEnvBase import MujocoHsrEnvBase


class MujocoHsrToolboxEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_toolbox.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.2, -0.6, 0.0, -1.0, 0.0, 0.8]),
            **kwargs,
        )

        self.original_toolbox_pos = self.model.body("toolbox").pos.copy()
        self.toolbox_pos_offsets = np.array(
            [
                [0.0, -0.06, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.06, 0.0],
                [0.0, 0.09, 0.0],
            ]
        )

    def _get_success(self):
        toolbox_pos = self.data.body("toolbox").xpos.copy()
        mat_pos = self.data.body("mat").xpos.copy()

        xy_thre = 0.03
        z_thre = mat_pos[2] + 0.005
        return (np.max(np.abs(toolbox_pos[:2] - mat_pos[:2])) < xy_thre) and (
            toolbox_pos[2] < z_thre
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.toolbox_pos_offsets)

        toolbox_pos = self.original_toolbox_pos + self.toolbox_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            toolbox_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        toolbox_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "toolbox_freejoint"
        )
        toolbox_qpos_addr = self.model.jnt_qposadr[toolbox_joint_id]
        self.init_qpos[toolbox_qpos_addr : toolbox_qpos_addr + 3] = toolbox_pos

        return world_idx
