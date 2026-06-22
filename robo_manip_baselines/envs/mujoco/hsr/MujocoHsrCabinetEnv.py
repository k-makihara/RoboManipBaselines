from os import path

import numpy as np

from .MujocoHsrPaPEnvBase import MujocoHsrEnvBase


class MujocoHsrCabinetEnv(MujocoHsrEnvBase):
    default_camera_config = {
        "azimuth": -45.0,
        "elevation": -35.0,
        "distance": 1.8,
        "lookat": [0.3, 0.0, 0.5],
    }

    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_cabinet.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57, 0.0, 0.8]),
            **kwargs,
        )

        self.original_cabinet_pos = self.model.body("cabinet").pos.copy()
        self.cabinet_pos_offsets = np.array(
            [
                [-0.15, -0.06, 0.0],
                [-0.15, -0.03, 0.0],
                [-0.15, 0.0, 0.0],
                [-0.15, 0.03, 0.0],
                [-0.15, 0.06, 0.0],
                [-0.15, 0.09, 0.0],
            ]
        )

        self.target_task = None

    def _get_success(self):
        hinge_success = self.data.joint("hinge").qpos[0] > np.deg2rad(120.0)
        slide_success = self.data.joint("slide").qpos[0] > 0.12
        if self.target_task is None:
            return hinge_success or slide_success
        if self.target_task == "hinge":
            return hinge_success
        if self.target_task == "slide":
            return slide_success
        raise ValueError(
            f"[{self.__class__.__name__}] Invalid target task: {self.target_task}"
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.cabinet_pos_offsets)

        cabinet_pos = self.original_cabinet_pos + self.cabinet_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            cabinet_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("cabinet").pos = cabinet_pos

        return world_idx
