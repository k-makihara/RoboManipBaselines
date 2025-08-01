import os
import sys
from pathlib import Path
from typing import Union

import cv2
import matplotlib.pylab as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg
# import matplotlib
# matplotlib.use("TkAgg")
import numpy as np
from dataclasses import dataclass
import torch
import yaml
from collections import deque





sys.path.append("/home/deepstation/Isaac-GR00T")
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

from robo_manip_baselines.common import RolloutBase, denormalize_data
from robo_manip_baselines.common.data.DataKey import DataKey


class RolloutGr00t(RolloutBase):
    require_task_desc = True

    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        

        # get the data config
        self.data_config = DATA_CONFIG_MAP["hsrrmb"]

        # get the modality configs and transforms
        modality_config = self.data_config.modality_config()
        transforms = self.data_config.transform()

        self.gr00t = Gr00tPolicy(
            model_path=self.args.checkpoint,
            modality_config=modality_config,
            modality_transform=transforms,
            embodiment_tag="new_embodiment",
            device="cuda"
        )
        adopted_action_chunks = 16
        print(f"  - chunk size: {adopted_action_chunks}")
        self.adopted_action_chunks = adopted_action_chunks
        self.action_queue: deque = deque(maxlen=adopted_action_chunks)

        #self.device = torch.device("cpu")

    def setup_plot(self):
        fig_ax = plt.subplots(
            2,
            len(self.camera_names) + 1,
            figsize=(13.5, 6.0),
            dpi=60,
            squeeze=False,
            constrained_layout=True,
        )
        # self.fig, self.ax = fig_ax

        # for _ax in np.ravel(self.ax):
        #     _ax.cla()
        #     _ax.axis("off")

        # plt.figure(self.policy_name)

        # self.canvas = FigureCanvasAgg(self.fig)
        # self.canvas.draw()
        # plt.imshow(
        #     cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR)
        # )

        # if self.args.win_xy_plot is not None:
        #     plt.get_current_fig_manager().window.wm_geometry("+20+50")

        # if len(self.action_keys) > 0:
        #     self.action_plot_scale = np.concatenate(
        #         [DataKey.get_plot_scale(key, self.env) for key in self.action_keys]
        #     )
        # else:
        #     self.action_plot_scale = np.zeros(0)

    def reset_variables(self):
        super().reset_variables()

    def setup_model_meta_info(self):
        cmd_args = " ".join(sys.argv).lower()
        if "aloha" in cmd_args:
            data_config = DATA_CONFIG_MAP["aloha"]
            self.state_dim = 14
            self.action_dim = 14
        elif "ur5e" in cmd_args:
            data_config = DATA_CONFIG_MAP["ur5e"]
            self.state_dim = 7
            self.action_dim = 7
        else:
            data_config = DATA_CONFIG_MAP["hsrrmb"]
            self.state_dim = 9
            self.action_dim = 9
        self.state_keys = ["measured_joint_pos", "measured_mobile_omni_vel"]
        self.action_keys = ["command_joint_pos", "command_mobile_omni_vel"]
        self.camera_names = ["head", "hand"]

        if self.args.skip is None:
            self.args.skip = 1
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def infer_policy(self):
        # Infer

        if len(self.action_queue) == 0:
            state = self.get_state()
            state = state[np.newaxis]
            #print(np.array([state[0][0:5]]))
            #print(np.array([state[0][5]]))
            #print(np.array([state[0][6:9]]))

            images = self.get_images()

            observation = {
                "state.arm": np.array([state[0][0:5]]),
                "state.gripper": np.array([[state[0][5]]]),
                "state.base": np.array([state[0][6:9]]),
                "annotation.human.action.task_description": [self.args.task_desc]
            }
            for camera_name in self.camera_names:
                observation[f"video.{camera_name}_rgb"] = images[camera_name]

            all_actions = self.gr00t.get_action(observation)

            policy_action_arm = all_actions["action.arm"]
            policy_action_gripper = np.expand_dims(all_actions["action.gripper"],axis=-1)
            policy_action_base = all_actions["action.base"]
            policy_action = np.concatenate([policy_action_arm, policy_action_gripper, policy_action_base], axis=1)

            #action = np.expand_dims(policy_action, axis=0)
            self.action_queue.extend(policy_action.transpose(0, 1))

        
        self.policy_action = self.action_queue.popleft()
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )

    def get_state(self):
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    self.motion_manager.get_data(state_key, self.obs)
                    for state_key in self.state_keys
                ]
            )

        return state
    
    def get_images(self):
        # Assume all images are the same size
        images = {}
        for camera_name in self.camera_names:
            image = self.info["rgb_images"][camera_name][np.newaxis]
            images[camera_name] = image

        return images

    def draw_plot(self):
        # Clear plot
        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        # Plot images
        self.plot_images(self.ax[0, 0 : len(self.camera_names)])

        # Plot action
        self.plot_action(self.ax[0, len(self.camera_names)])

        plt.figure(self.policy_name)

        # Finalize plot
        self.canvas.draw()
        # plt.imshow(
        #     cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR)
        # )
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

    # def run(self):
    #     self.reset_flag = True
    #     self.quit_flag = False
    #     self.inference_duration_list = []

    #     self.motion_manager.reset()

    #     self.obs, self.info = self.env.reset(seed=self.args.seed)

    #     self.time = 0
    #     self.key = 0

    #     while True:
    #         if self.reset_flag:
    #             self.reset()
    #             self.reset_flag = False
                
    #         self.phase_manager.pre_update()

    #         env_action = np.concatenate(
    #             [
    #                 self.motion_manager.get_command_data(key)
    #                 for key in self.env.unwrapped.command_keys_for_step
    #             ]
    #         )
    #         self.obs, self.reward, self.terminated, _, self.info = self.env.step(
    #             env_action
    #         )

    #         self.phase_manager.post_update()

    #         self.time += 1
    #         self.phase_manager.check_transition()

    #         if self.quit_flag:
    #             break

    #     if self.args.result_filename is not None:
    #         print(
    #             f"[{self.__class__.__name__}] Save the rollout results: {self.args.result_filename}"
    #         )
    #         with open(self.args.result_filename, "w") as result_file:
    #             yaml.dump(self.result, result_file)