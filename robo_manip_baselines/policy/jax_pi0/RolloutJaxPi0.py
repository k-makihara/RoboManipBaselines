import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "/home/deepstation/anaconda3/envs/rmb/lib/python3.10/site-packages"))

from pathlib import Path
from typing import Union

import cv2
# import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pylab as plt
# from matplotlib.backends.backend_agg import FigureCanvasTkAgg

import numpy as np
from dataclasses import dataclass
import torch
import yaml

from collections import deque
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "/home/deepstation/rfm/hsr_openpi/src"))
from openpi.training import config
from openpi.policies import policy_config

from robo_manip_baselines.common import RolloutBase, denormalize_data
from robo_manip_baselines.common.data.DataKey import DataKey


class RolloutJaxPi0(RolloutBase):
    require_task_desc = True

    def setup_policy(self):
        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {8}")

        adopted_action_chunks = 50
        config_name = "pi0_sim-hsr_low_mem_finetune"

        self.config: config.TrainConfig = config.get_config(config_name)
        self.policy = policy_config.create_trained_policy(self.config, self.args.checkpoint)

        self.adopted_action_chunks = adopted_action_chunks
        self.action_queue: deque = deque(maxlen=adopted_action_chunks)
        print("Finish Loading")

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
        super().setup_plot(fig_ax)
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

    def setup_model_meta_info(self):
        cmd_args = " ".join(sys.argv).lower()
        self.state_keys = ["measured_joint_pos", "measured_mobile_omni_vel"]
        self.action_keys = ["command_joint_pos", "command_mobile_omni_vel"]
        self.camera_names = ["head", "hand"]
        if "aloha" in cmd_args:
            self.state_dim = 14
            self.action_dim = 14
        elif "ur5e" in cmd_args:
            self.state_dim = 7
            self.action_dim = 7
        else:
            self.state_dim = 9
            self.action_dim = 9

        if self.args.skip is None:
            self.args.skip = 1
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def infer_policy(self):
        # Infer

        if len(self.action_queue) == 0:
            state = self.get_state()
            state = state[np.newaxis]
            state = torch.from_numpy(state.copy()).to("cuda:0")
            state = state.type(torch.float32)

            images = self.get_images()

            # observation = {
            #     "observation.state": state,
            #     "task": [self.args.task_desc],
            # }
            # for camera_name in self.camera_names:
            #     observation[f"observation.images.{camera_name}_rgb"] = images[camera_name]

            # action = self.policy.select_action(observation)
            # action = torch.squeeze(action)

            # self.policy_action = action.cpu().detach().numpy().astype(np.float64)
            # self.policy_action_list = np.concatenate(
            #     [self.policy_action_list, self.policy_action[np.newaxis]]
            # )


            policy_input = {
                "head_rgb": images["head"],
                "hand_rgb": images["hand"],
                "state": state,
                "prompt": self.args.task_desc,
            }
            #print("Input")

            action_chunk = self.policy.infer(policy_input)["actions"]
            action = np.expand_dims(action_chunk, axis=0)
            #print(action.shape)
            self.action_queue.extend(action_chunk.transpose(0, 1))
        
        self.policy_action = self.action_queue.popleft()
        self.policy_action_list = np.concatenate(
            [self.policy_action_list, self.policy_action[np.newaxis]]
        )
        #print(self.policy_action.shape)
        #print(self.policy_action_list.shape)
        
        #self.action_queue.extend(action_chunk[1 : self.adopted_action_chunks])
        #action = action_chunk[0]


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
            image = self.info["rgb_images"][camera_name][np.newaxis].transpose(0, 3, 1, 2)
            image = torch.from_numpy(image.copy()).to("cuda:0")
            image = image.type(torch.float32)
            image /= 255
            images[camera_name] = image

        return images
    
    def reset_variables(self):
        super().reset_variables()

        self.policy.reset()
        self._action_queue = deque([], maxlen=self.adopted_action_chunks)

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