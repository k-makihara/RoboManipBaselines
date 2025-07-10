import numpy as np

from .RealUR5eEnvBase import RealUR5eEnvBase


class RealUR5eDemoEnv(RealUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealUR5eEnvBase.__init__(
            self,
            init_qpos=np.array(
                [
                    1.57, -1.57, 1.57, -1.57, -1.57, 0, 0
                    # 1.18000162,
                    # -1.91696992,
                    # 1.5561803,
                    # -1.21203147,
                    # -1.57465679,
                    # -0.39695961,
                    # 0.0,
                ]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
