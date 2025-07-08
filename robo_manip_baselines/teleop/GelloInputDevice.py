import glob

import numpy as np

from .InputDeviceBase import InputDeviceBase


class GelloInputDevice(InputDeviceBase):
    """
    GELLO for teleoperation input device.

    Ref: https://github.com/wuphilipp/gello_software/blob/daae81f4a78ec0f7534937413345d96d3e1bc7fc/experiments/run_env.py
    """

    PORT_CONFIG_MAP = {
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT9MIQNO-if00-port0": dict(
            joint_ids=(1, 2, 3, 4, 5, 6),
            joint_offsets=(
                -1 * np.pi / 2,
                3 * np.pi / 2,
                2 * np.pi / 2,
                3 * np.pi / 2,
                2 * np.pi / 2,
                -1 * np.pi / 2,
            ),
            joint_signs=(1, 1, -1, 1, 1, 1),
            gripper_config=(7, 197.378125, 155.578125),
        ),
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT94EQ26-if00-port0": dict(
            joint_ids=(1, 2, 3, 4, 5, 6),
            joint_offsets=(
                0*np.pi/2, 3*np.pi/2, 2*np.pi/2, 2*np.pi/2, 4*np.pi/2, 1*np.pi/2
                # np.pi + 0 * np.pi,
                # 2 * np.pi + np.pi / 2,
                # 2 * np.pi + np.pi / 2,
                # 2 * np.pi + np.pi / 2,
                # 1 * np.pi,
                # 3 * np.pi / 2,
            ),
            joint_signs=(1, 1, -1, 1, 1, 1),
            gripper_config=(7, 111.596875, 69.796875),
        ),
    }

    def __init__(self, arm_manager, port=None):
        super().__init__()

        self.arm_manager = arm_manager
        self.port = port

    def connect(self):
        if self.connected:
            return

        self.connected = True

        from gello.agents.gello_agent import DynamixelRobotConfig, GelloAgent

        # Set port
        if self.port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) > 0:
                self.port = usb_ports[0]
                print(
                    f"[{self.__class__.__name__}] Found {len(usb_ports)} ports in /dev/serial/by-id/*."
                )
            else:
                raise RuntimeError(f"[{self.__class__.__name__}] No port found.")
        print(f"[{self.__class__.__name__}] Connect to {self.port}")

        # Instantiate gello agent
        dynamixel_config = DynamixelRobotConfig(**self.PORT_CONFIG_MAP[self.port])
        current_joint_pos = np.concatenate(self.arm_manager.get_command_joint_pos())
        self.agent = GelloAgent(
            port=self.port,
            dynamixel_config=dynamixel_config,
            start_joints=current_joint_pos,
        )
        self.time_idx = 0

    def close(self):
        if self.connected:
            self.agent.close()

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        arm_joint_idxes = self.arm_manager.body_config.arm_joint_idxes
        gripper_joint_idxes_for_limit = (
            self.arm_manager.body_config.gripper_joint_idxes_for_limit
        )
        gripper_joint_pos_low = self.arm_manager.env.action_space.low[
            gripper_joint_idxes_for_limit
        ]
        gripper_joint_pos_high = self.arm_manager.env.action_space.high[
            gripper_joint_idxes_for_limit
        ]

        # Assume that the joint angles obtained from GELLO are in the order of arm joints followed by gripper joints
        joint_pos = self.agent.act().copy()
        arm_joint_pos = joint_pos[: len(arm_joint_idxes)]
        gripper_joint_pos = (
            gripper_joint_pos_high - gripper_joint_pos_low
        ) * joint_pos[len(arm_joint_idxes) :] + gripper_joint_pos_low

        self.state = (arm_joint_pos, gripper_joint_pos)

    def is_ready(self):
        # Check only arm joints
        current_joint_pos = self.arm_manager.get_command_joint_pos()[0]
        new_joint_pos = self.state[0]
        if current_joint_pos.shape != new_joint_pos.shape:
            raise RuntimeError(
                f"[{self.__class__.__name__}] The shape of current_joint_pos and new_joint_pos do not match: {current_joint_pos.shape} != {new_joint_pos.shape}"
            )

        joint_pos_error = np.max(np.abs(current_joint_pos - new_joint_pos))
        joint_pos_error_thre = np.deg2rad(30.0)  # [rad]
        if joint_pos_error < joint_pos_error_thre:
            print(f"[{self.__class__.__name__}] Ready to start.")
            return True
        else:
            print(
                f"[{self.__class__.__name__}] Joint angles differ greatly.\n  joint_name, robot_joint, gello_joint (joint_status):"
            )
            for joint_idx, (current_joint_pos0, new_joint_pos0) in enumerate(
                zip(current_joint_pos, new_joint_pos)
            ):
                joint_pos_error0 = np.abs(current_joint_pos0 - new_joint_pos0)
                if joint_pos_error0 > joint_pos_error_thre:
                    joint_status_str = (
                        f"NG: {joint_pos_error0:.2f} < {joint_pos_error_thre:.2f}"
                    )
                else:
                    joint_status_str = (
                        f"OK: {joint_pos_error0:.2f} > {joint_pos_error_thre:.2f}"
                    )
                print(
                    f"  - joint{joint_idx}: {current_joint_pos0:.2f}, {new_joint_pos0:.2f} ({joint_status_str})"
                )
            return False

    def set_command_data(self):
        interp_end_time_idx = 50
        if self.time_idx < interp_end_time_idx:
            interp_ratio = self.time_idx / interp_end_time_idx
            current_joint_pos = self.arm_manager.get_command_joint_pos()
            new_joint_pos = tuple(
                interp_ratio * self.state[i] + (1 - interp_ratio) * current_joint_pos[i]
                for i in range(2)
            )
        else:
            new_joint_pos = self.state

        self.arm_manager.set_command_joint_pos(*new_joint_pos)
        self.time_idx += 1
