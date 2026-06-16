from os import path

import numpy as np

from .MujocoHsrPaPEnvBase import MujocoHsrEnvBase


class MujocoHsrInsertEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_insert.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.22, -0.55, 0.0, -0.95, 0.0, 0.8]),
            **kwargs,
        )

        self.original_hole_pos = self.model.body("hole").pos.copy()
        self.hole_pos_offsets = np.array(
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
        peg_pos = self.data.body("peg").xpos.copy()
        hole_pos = self.data.body("hole").xpos.copy()

        return (np.max(np.abs(peg_pos[:2] - hole_pos[:2])) < 0.01) and (
            peg_pos[2] < hole_pos[2] + 0.05
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.hole_pos_offsets)

        hole_pos = self.original_hole_pos + self.hole_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            hole_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("hole").pos = hole_pos

        return world_idx
