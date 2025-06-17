from abc import ABC, abstractmethod

import numpy as np

from ..body.ArmManager import ArmManager
from ..data.DataKey import DataKey


class PhaseBase(ABC):
    """
    Base class for phase of robot operation.

    The op (meaning "operation") in the member variables is an instance of TeleopBase or RolloutBase.
    """

    def __init__(self, op):
        self.op = op

    def start(self):
        self.start_time = self.op.env.unwrapped.get_time()

    def pre_update(self):
        """Pre-update, which is called before the environment step."""
        pass

    def post_update(self):
        """Post-update, which is called after the environment step."""
        pass

    def check_transition(self):
        return False  # Never transition from this phase

    def get_elapsed_duration(self):
        return self.op.env.unwrapped.get_time() - self.start_time

    @property
    def name(self):
        return self.__class__.__name__


class ReachPhaseBase(PhaseBase, ABC):
    def start(self):
        super().start()

        self.set_target()

    def pre_update(self):
        self.op.motion_manager.set_command_data(
            DataKey.COMMAND_EEF_POSE, self.target_se3
        )

    def check_transition(self):
        return self.get_elapsed_duration() > self.duration

    @abstractmethod
    def set_target(self):
        pass


class GraspPhaseBase(PhaseBase, ABC):
    def start(self):
        super().start()

        self.set_target()

    def pre_update(self):
        self.op.motion_manager.set_command_data(
            DataKey.COMMAND_GRIPPER_JOINT_POS, self.gripper_joint_pos
        )

    def check_transition(self):
        return self.get_elapsed_duration() > self.duration

    @abstractmethod
    def set_target(self):
        pass

    def set_target_close(self):
        self.set_target_limit("high")

    def set_target_open(self):
        self.set_target_limit("low")

    def set_target_limit(self, high_low):
        if high_low == "high":
            action_limit = self.op.env.action_space.high
        elif high_low == "low":
            action_limit = self.op.env.action_space.low
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid high_low label: {high_low}"
            )

        self.gripper_joint_pos = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self.op.env)
        )

        for body_manager in self.op.motion_manager.body_manager_list:
            if not isinstance(body_manager, ArmManager):
                continue

            self.gripper_joint_pos[
                body_manager.body_config.gripper_joint_idxes_in_gripper_joint_pos
            ] = action_limit[body_manager.body_config.gripper_joint_idxes_for_limit]

        self.duration = 0.5  # [s]

class HSRGraspPhaseBase(PhaseBase, ABC):
    def start(self):
        super().start()

        self.set_target()

    def pre_update(self):
        self.op.motion_manager.set_command_data(
            DataKey.COMMAND_GRIPPER_JOINT_POS, self.gripper_joint_pos
        )

    def check_transition(self):
        return self.get_elapsed_duration() > self.duration

    @abstractmethod
    def set_target(self):
        pass

    def set_target_close(self):
        self.set_target_limit("low")

    def set_target_open(self):
        self.set_target_limit("high")

    def set_target_limit(self, high_low):
        if high_low == "high":
            action_limit = self.op.env.action_space.high
        elif high_low == "low":
            action_limit = self.op.env.action_space.low
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid high_low label: {high_low}"
            )

        self.gripper_joint_pos = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self.op.env)
        )

        for body_manager in self.op.motion_manager.body_manager_list:
            if not isinstance(body_manager, ArmManager):
                continue

            self.gripper_joint_pos[
                body_manager.body_config.gripper_joint_idxes_in_gripper_joint_pos
            ] = action_limit[body_manager.body_config.gripper_joint_idxes_for_limit]

        self.duration = 0.5  # [s]
