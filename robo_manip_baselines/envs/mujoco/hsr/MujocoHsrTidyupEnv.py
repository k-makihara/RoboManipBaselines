from os import path

import numpy as np

from .MujocoHsrEnvBase import MujocoHsrEnvBase
import mujoco
from mujoco import mjtObj
from collections import deque


class MujocoHsrTidyupEnv(MujocoHsrEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoHsrEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/hsr/env_hsr_tidyup.xml",
            ),
            np.array([0.0] * 3 + [0.25, -2.0, 0.0, -1.0, 0.0, 0.8]),
            #np.array([-0.5 ,-0.1, 0.0] + [0.35, -2.2, 0.0, -0.3, 0.0, 0.8]),
            **kwargs,
        )
        self.obj = "bottle2"

        self.original_obj_pos = self.model.body(self.obj).pos.copy()
        self.obj_pos_offsets = np.array(
            [
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.09, 0.0, 0.0],
                [0.12, 0.0, 0.0],
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

        #self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.obj)
        jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hand_motor_joint")
        self.qpos_addr = self.model.jnt_qposadr[jnt_id]
        self.prev_poses = deque(maxlen=5)
        #for i in range(5):
        #    self.prev_poses.append(self.original_obj_pos)

        c_body_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "container")
        self.final_pos      = self.data.xpos[c_body_id]


    def _get_success(self):

        

        # def euclidean_distance_np(a, b):
        #     a = np.asarray(a, dtype=float)
        #     b = np.asarray(b, dtype=float)
        #     return np.linalg.norm(a - b), np.abs(a[2] - b[2])
        # dist, z_dist = euclidean_distance_np(self.data.xpos[self.body_id], self.final_pos)
        #self.prev_poses.append(self.data.xpos[self.body_id])
        #arr = np.stack(self.prev_poses)
        #print(arr)
        #var_per_dim = np.var(arr, axis=0, ddof=0)
        #print(var_per_dim)
        #print(dist)
        #print(self.data.qpos[self.qpos_addr])
        # if z_dist < 0.005 and dist < 0.0 and self.data.qpos[self.qpos_addr] > 0.9:
        #     print("Success")
        #     return True
        # else:
        #     return False
        local_pos  = self.data.xpos[self.body_id] - self.final_pos
        if (
            -0.09 < local_pos[0] < +0.09 and
            -0.15 < local_pos[1] < +0.15 and
            0.0 < local_pos[2] < +0.12 and
            self.data.qpos[self.qpos_addr] > 0.9
        ):
            #print("Success")
            return True
        else:
            return False

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.obj_pos_offsets)

        obj_pos = self.original_obj_pos + self.obj_pos_offsets[world_idx]
        if self.world_random_scale is not None:
           obj_pos += np.random.uniform(
               low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
           )
        #self.model.body(obj).pos = obj_pos
        #print(obj_pos)
        
        body_id = mujoco.mj_name2id(self.model, mjtObj.mjOBJ_BODY, self.obj)
        jnt_id = self.model.body_jntadr[body_id]
        qpos_addr = self.model.jnt_qposadr[jnt_id]

        self.init_qpos[qpos_addr : qpos_addr+3] = obj_pos
        self.init_qpos[qpos_addr+3 : qpos_addr+7] = np.array([1.0, 0.0, 0.0, 0.0])
        print(obj_pos)



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