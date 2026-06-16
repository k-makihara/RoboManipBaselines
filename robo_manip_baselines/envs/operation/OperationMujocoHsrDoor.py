import gymnasium as gym


class OperationMujocoHsrDoor:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoHsrDoorEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        return []
