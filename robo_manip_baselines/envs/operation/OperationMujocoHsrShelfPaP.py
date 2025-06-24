import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import HSRGraspPhaseBase, ReachPhaseBase

def get_target_se3(op, pos_z):
    target_pos = op.env.unwrapped.get_body_pose("bottle1")[0:3]
    target_pos[1] += 0.0
    target_pos[2] = pos_z
    return pin.SE3(np.diag([-1.0, 1.0, -1.0]), target_pos)


class ReachPhase1(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=0.5,  # [m]
        )
        self.duration = 0.7  # [s]


class ReachPhase2(ReachPhaseBase):
    def set_target(self):
        self.target_se3 = get_target_se3(
            self.op,
            pos_z=0.35,  # [m]
        )
        self.duration = 0.3  # [s]


class GraspPhase(HSRGraspPhaseBase):
    def set_target(self):
        self.set_target_close()

class OperationMujocoHsrShelfPaP:
    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/MujocoHsrShelfPaPEnv-v0", render_mode=render_mode
        )

    def get_pre_motion_phases(self):
        # return [
        #     ReachPhase1(self),
        #     ReachPhase2(self),
        #     GraspPhase(self),
        # ]
        return []
