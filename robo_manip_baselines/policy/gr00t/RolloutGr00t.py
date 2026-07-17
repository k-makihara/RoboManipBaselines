import numpy as np
import torch

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/Isaac-GR00T"))
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy
from robo_manip_baselines.common import RolloutBase
from robo_manip_baselines.common.data.DataKey import DataKey


class RolloutGr00t(RolloutBase):
    require_task_desc = True

    def set_additional_args(self, parser):
        # TODO: Disable rendering with matplotlib and cv2, as it causes the program to hang
        parser.set_defaults(no_plot=True)

    def setup_model_meta_info(self):
        # GR00T does not explicitly store the state and action dimensions in the checkpoint
        self.state_dim = None
        self.action_dim = None

        self.state_keys = [DataKey.MEASURED_JOINT_POS]
        self.action_keys = [DataKey.COMMAND_JOINT_POS]

        if self.args.skip is None:
            self.args.skip = 1
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

    def setup_policy(self):
        self.device = torch.device("cuda")

        self.policy = Gr00tPolicy(
            model_path=self.args.checkpoint,
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            device=str(self.device),
        )

        self.modality_config = self.policy.get_modality_config()
        self.camera_names = [
            key.replace("_rgb", "")
            for key in self.modality_config["video"].modality_keys
        ]

        self.print_policy_info()

    def reset_variables(self):
        self.policy_action_list = None

        self.policy_action_buf = None

    def get_state(self):
        if len(self.state_keys) == 0:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(
            [
                self.motion_manager.get_data(state_key, self.obs)
                for state_key in self.state_keys
            ]
        ).astype(np.float32)[np.newaxis]

    def get_images(self):
        images = {}
        for camera_name in self.camera_names:
            images[camera_name] = self.info["rgb_images"][camera_name][np.newaxis]
        return images

    def infer_policy(self):
        if self.policy_action_buf is None or len(self.policy_action_buf) == 0:
            state = self.get_state()
            images = self.get_images()

            observation = {
                "state.single_arm": state,
                "annotation.human.task_description": self.args.task_desc,
            }
            for camera_name in self.camera_names:
                observation[f"video.{camera_name}_rgb"] = images[camera_name]

            parsed_obs = parse_observation_gr00t(observation, self.modality_config)
            all_actions, _ = self.policy.get_action(parsed_obs)
            parsed_action = parse_action_gr00t(all_actions)
            self.policy_action_buf = list(parsed_action["action.single_arm"])

        self.policy_action = self.policy_action_buf.pop(0)

        if self.policy_action_list is None:
            self.policy_action_list = self.policy_action[np.newaxis]
        else:
            self.policy_action_list = np.concatenate(
                [self.policy_action_list, self.policy_action[np.newaxis]]
            )

    def draw_plot(self):
        # TODO: Disable rendering with matplotlib and cv2, as it causes the program to hang
        pass