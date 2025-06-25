from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import (
    ArmConfig,
    DataKey,
    MobileOmniConfig,
    get_se3_from_pose,
)
from robo_manip_baselines.teleop import (
    SpacemouseInputDevice,
    SpacemouseMobileInputDevice,
    KeyboardMobileInputDevice,
)

from ..MujocoEnvBase import MujocoEnvBase


class MujocoHsrEnvBase(MujocoEnvBase):
    # default_camera_config = {
    #     "azimuth": -180.0,
    #     "elevation": -30.0,
    #     "distance": 1.8,
    #     "lookat": [0.5, 0.0, 0.2],
    # }
    default_camera_config = {
        "azimuth": 180.0,
        "elevation": -30.0,
        "distance": 1.8,
        "lookat": [0.5, 0.0, 0.2],
    }
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
            "mobile_vel": Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        }
    )

    def setup_robot(self, init_qpos):
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        mujoco.mj_kinematics(self.model, self.data)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__), "../../assets/common/robots/hsr/hsr.urdf"
                ),
                arm_root_pose=self.get_body_pose("base_link"),
                ik_eef_joint_id=5,
                arm_joint_idxes=np.arange(5),
                gripper_joint_idxes=np.array([5]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[3:8],
                init_gripper_joint_pos=self.init_qpos[[8]],
                get_root_pose_func=lambda env: get_se3_from_pose(
                    env.get_body_pose("base_link")
                ),
            ),
            MobileOmniConfig(),
        ]

        self.mobile_joint_name_list = [
            "mobile_x_joint",
            "mobile_y_joint",
            "mobile_theta_joint",
        ]

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        default_kwargs = self.get_input_device_kwargs(input_device_name)

        if input_device_name == "spacemouse":
            return [
                SpacemouseInputDevice(
                    motion_manager.body_manager_list[0],
                    **{**default_kwargs.get(0, {}), **overwrite_kwargs.get(0, {})},
                ),
                SpacemouseMobileInputDevice(
                    motion_manager.body_manager_list[1],
                    **{**default_kwargs.get(1, {}), **overwrite_kwargs.get(1, {})},
                ),
            ]
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {0: {"gripper_scale": 0.05}, 1: {}}
        else:
            return super().get_input_device_kwargs(input_device_name)

    @property
    def command_keys_for_step(self):
        return [DataKey.COMMAND_MOBILE_OMNI_VEL, DataKey.COMMAND_JOINT_POS]

    @property
    def measured_keys_to_save(self):
        return [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.MEASURED_EEF_POSE,
            DataKey.MEASURED_EEF_WRENCH,
            DataKey.MEASURED_MOBILE_OMNI_VEL,
        ]

    @property
    def command_keys_to_save(self):
        return [
            DataKey.COMMAND_JOINT_POS,
            DataKey.COMMAND_GRIPPER_JOINT_POS,
            DataKey.COMMAND_EEF_POSE,
            DataKey.COMMAND_MOBILE_OMNI_VEL,
        ]

    def step(self, action):
        action[0:3] = self.convert_mobile_vel_frame(action[0:3], world_to_local=False)

        return super().step(action)

    def _get_obs(self):
        arm_joint_name_list = [
            "arm_lift_joint",
            "arm_flex_joint",
            "arm_roll_joint",
            "wrist_flex_joint",
            "wrist_roll_joint",
        ]
        gripper_joint_name = "hand_motor_joint"

        arm_joint_pos = np.array(
            [self.data.joint(joint_name).qpos[0] for joint_name in arm_joint_name_list]
        )
        arm_joint_vel = np.array(
            [self.data.joint(joint_name).qvel[0] for joint_name in arm_joint_name_list]
        )
        gripper_joint_pos = np.array([self.data.joint(gripper_joint_name).qpos[0]])
        gripper_joint_vel = np.zeros(1)
        force = self.data.sensor("force_sensor").data.flat.copy()
        torque = self.data.sensor("torque_sensor").data.flat.copy()

        mobile_vel = np.array(
            [
                self.data.joint(joint_name).qvel[0]
                for joint_name in self.mobile_joint_name_list
            ]
        )
        mobile_vel = self.convert_mobile_vel_frame(mobile_vel, world_to_local=True)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": np.concatenate((force, torque), dtype=np.float64),
            "mobile_vel": mobile_vel.astype(np.float64),
        }

    def get_mobile_vel_from_obs(self, obs):
        """Get velocity of omni-directional mobile base from observation."""
        return obs["mobile_vel"]

    def convert_mobile_vel_frame(self, vel_in, world_to_local):
        theta = self.data.joint(self.mobile_joint_name_list[-1]).qpos[0]
        if not world_to_local:
            theta *= -1

        rot_mat = np.array(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )

        vel_in_xy = vel_in[0:2]
        vel_out_xy = rot_mat @ vel_in_xy

        return np.concatenate([vel_out_xy, vel_in[[2]]])
