from os import path

import numpy as np

from .MujocoHsrEnvBase import MujocoHsrEnvBase


class MujocoHsrPaPEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_tidyup_2.xml",
            ),
            #np.array([0.0] * 3 + [0.25, -2.0, 0.0, -1.0, 0.0, 0.8]),
            np.array([-0.5 ,-0.1, 0.0] + [0.35, -2.2, 0.0, -0.3, 0.0, 0.8]),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = 0
        return world_idx
