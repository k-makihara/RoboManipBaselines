import gymnasium as gym


class OperationMujocoHsrRing:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoHsrRingEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return []
