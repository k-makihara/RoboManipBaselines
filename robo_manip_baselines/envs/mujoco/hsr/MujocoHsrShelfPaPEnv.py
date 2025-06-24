from os import path

import numpy as np

from .MujocoHsrPaPEnvBase import MujocoHsrEnvBase
import mujoco
from mujoco import mjtObj

class MujocoHsrShelfPaPEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_shelfpap.xml",
            ),
            #np.array([0.0] * 3 + [0.25, -2.0, 0.0, -1.0, 0.0, 0.8]),
            #np.array([-0.5 ,-0.1, 0.0] + [0.35, -2.2, 0.0, -0.3, 0.0, 0.8]),
            np.array([-0.5 ,-0.1, 0.0] + [0.1, -0.4, 0.0, -1.0, 0.0, 0.8]),
            **kwargs,
        )
        

        self.original_obj_pos = self.model.body("bottle1").pos.copy()
        self.obj_pos_offsets = np.array(
            [
                #[-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                #[0.03, 0.0, 0.0],
                #[0.06, 0.0, 0.0],
                #[0.09, 0.0, 0.0],
                #[0.12, 0.0, 0.0],
            ]
            )
        self.original_robot_pos = self.model.body("hsr_body").pos.copy()
        # self.original_x_pos = self.model.body("mobile_x_joint").pos.copy()
        # self.original_y_pos = self.model.body("mobile_y_joint").pos.copy()
        # self.original_theta_pos = self.model.body("mobile_theta_joint").pos.copy()
        # base_joints = {
        #     'mobile_x_joint': tx,
        #     'mobile_y_joint': ty,
        #     'mobile_theta_joint': theta,
        # }
        print(self.original_robot_pos)
        # print(self.original_x_pos)
        # print(self.original_y_pos)
        # print(self.original_theta_pos)
        self.robot_pos_offsets = np.array(
            [
                #[-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                #[0.03, 0.0, 0.0],
                #[0.06, 0.0, 0.0],
                #[0.09, 0.0, 0.0],
                #[0.12, 0.0, 0.0],
            ]
            )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.obj_pos_offsets)

        obj_pos = self.original_obj_pos + self.obj_pos_offsets[world_idx]
        # if self.world_random_scale is not None:
        #    obj_pos += np.random.uniform(
        #        low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
        #    )
        
        obj_pos[2] = self.original_obj_pos[2]
        
        body_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, "bottle1")
        jnt_id = self.model.body_jntadr[body_id]
        qpos_addr = self.model.jnt_qposadr[jnt_id]

        self.init_qpos[qpos_addr : qpos_addr+3] = obj_pos
        self.init_qpos[qpos_addr+3 : qpos_addr+7] = np.array([1.0, 0.0, 0.0, 0.0])



        # robot_pos = self.original_robot_pos + self.robot_pos_offsets[world_idx]
        # if self.world_random_scale is not None:
        #     robot_pos += np.random.uniform(
        #         low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
        #     )
        
        # body_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, "hsr_body")
        # jnt_id = self.model.body_jntadr[body_id]
        # qpos_addr = self.model.jnt_qposadr[jnt_id]

        # self.init_qpos[qpos_addr : qpos_addr+3] = obj_pos
        # self.init_qpos[qpos_addr+3 : qpos_addr+7] = np.array([1.0, 0.0, 0.0, 0.0])
        # tx = robot_pos[0]
        # ty = robot_pos[1]
        # theta = self.original_robot_pos[2]


        # for jname, value in {
        #     'mobile_x_joint': tx,
        #     'mobile_y_joint': ty,
        #     'mobile_theta_joint': theta,
        # }.items():
        #     jnt_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_JOINT, jname)
        #     addr   = self.model.jnt_qposadr[jnt_id]
        #     self.init_qpos[addr] = value



        return world_idx
