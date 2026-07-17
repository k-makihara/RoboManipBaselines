"""
Script to convert UR5e hdf5 data to the LeRobot dataset v2.0 format.

Example usage: uv run examples/aloha_real/convert_aloha_data_to_lerobot.py --raw-dir /path/to/raw/data --repo-id <org>/<dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import json

import einops
import cv2
import h5py
#from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
LEROBOT_HOME=Path("/groups/gaf51379/physical-grounding/datasets/lerobot_dataset")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
#from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    mode: Literal["video", "image"] = "video",
    *,
    has_velocity: bool = False,
    has_effort: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "base",
        "shoulder",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
        "gripper",
    ]
    cameras = [
        "front_rgb",
        "hand_rgb",
    ]

    features = {
        "observation.state": {
            "dtype": "float64",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
        "action": {
            "dtype": "float64",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float64",
            "shape": (len(motors),),
            "names": [
                motors,
            ],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float64",
            "shape": (6,),
            "names": [
                motors,
            ],
        }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: list[Path]) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        # ignore depth channel, not currently handled
        return [key for key in ep["/observations/images"].keys() if "depth" not in key]  # noqa: SIM118


def has_velocity(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/measured_joint_vel" in ep


def has_effort(hdf5_files: list[Path]) -> bool:
    with h5py.File(hdf5_files[0], "r") as ep:
        return "/effort" in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f"/observations/images/{camera}"].ndim == 4

        if uncompressed:
            # load all images in RAM
            imgs_array = ep[f"/observations/images/{camera}"][:]
        else:

            # load one compressed image after the other in RAM and uncompress
            imgs_array = []
            for data in ep[f"/observations/images/{camera}"]:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_images_per_camera_from_mp4(ep_path: Path, cameras: list[str]) -> dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        video_path = ep_path / f"{camera}_image.rmb.mp4"
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        array = np.reshape(frame,(1,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3))
        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = np.reshape(frame,(1,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),3))       
            array = np.append(array,frame,axis=0)
        cap.release()

        if len(array) > 1:
            array = array[:-1]

        imgs_per_cam[camera] = array
    return imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path / "main.rmb.hdf5", "r") as ep:
        state = torch.from_numpy(ep["/measured_joint_pos"][:-1])
        action = torch.from_numpy(ep["/measured_joint_pos"][1:])

        velocity = None
        if "/measured_joint_vel" in ep:
            velocity = torch.from_numpy(ep["/measured_joint_vel"][:-1])

        effort = None
        if "/measured_eef_wrench" in ep:
            effort = torch.from_numpy(ep["/measured_eef_wrench"][:-1])

    imgs_per_cam = load_raw_images_per_camera_from_mp4(
        ep_path,
        [
            "front_rgb",
            "hand_rgb",
        ],
    )

    return imgs_per_cam, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    dataset_paths: list[Path],
    task: str,
    episodes: list[int] | None = None,
) -> LeRobotDataset:


    import json
    from collections import defaultdict

    vqa = json.load(open("/groups/gaf51379/physical-grounding/products/questionnaire_vqa_abs_multi_train_v5.json","rb"))

    item_list = [   
                    "7i_langue_de_chat_choco",
                    "cocacola_300ml",
                    #"fab_anti_bacoriginal_tumekae",
                    "attack_neo_tumekae",
                    "7i_kodawari_kakipi_oobukuro",
                    "dars_white",
                    "mens_biore_semgam_refill",
                    "umiajisen_2",
                    "maiji_milk_chocolate",
                    "7i_ajitsuke_ponzu",
                    "7i_pirittokarai_cheese_snacks"

                ]
    buckets = defaultdict(lambda: [[], [], []])
    for i in range(len(vqa)):
        for match_ind, sub in enumerate(item_list):
            if sub in vqa[i]["image_path"]:
                if "deformable" in vqa[i]["conversation"][0]["value"]:
                    buckets[match_ind][0].append(vqa[i]["conversation"][1]["value"])
                elif "vulnerable" in vqa[i]["conversation"][0]["value"]:
                    buckets[match_ind][1].append(vqa[i]["conversation"][1]["value"])
                elif "slippery" in vqa[i]["conversation"][0]["value"]:
                    buckets[match_ind][2].append(vqa[i]["conversation"][1]["value"])
    
    for idx, sub in enumerate(item_list):
        if idx in buckets:
            print(f"=== {idx}: '{sub}' に対応するデータ ===")
            for j, group in enumerate(buckets[idx]):
                print(f"  グループ{j}: 件数={len(group)} データ={group}")

    def flip_level(s: str) -> str:
        """'high' <-> 'low' を反転させる"""
        if s.lower() == "high":
            return "low"
        elif s.lower() == "low":
            return "high"
        return s

    for i in range(len(item_list)):
        print(f"Grasp the {flip_level(buckets[i][0][0])} deformability, {flip_level(buckets[i][1][0])} vulnerability, and {flip_level(buckets[i][2][0])} slipperiness object")

    for task_path in dataset_paths:
        task_episodes = sorted(task_path.glob("*"))
        if episodes is None:
            episodes = range(len(task_episodes))

        for ep_idx in tqdm.tqdm(episodes):
            ep_path = task_episodes[ep_idx]
            raw_file_string = ep_path.parents[0].name
            if "RealUR5eDemo_20250801_175635" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[0][0][0])} deformability, {flip_level(buckets[0][1][0])} vulnerability, and {flip_level(buckets[0][2][0])} slipperiness object"
                #7i_langue_de_chat_choco
            elif "RealUR5eDemo_20250801_181121" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[1][0][0])} deformability, {flip_level(buckets[1][1][0])} vulnerability, and {flip_level(buckets[1][2][0])} slipperiness object"
                #cocacola_300ml
            elif "RealUR5eDemo_20250801_182658" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[2][0][0])} deformability, {flip_level(buckets[2][1][0])} vulnerability, and {flip_level(buckets[2][2][0])} slipperiness object"
                #fab_anti_bacoriginal_tumekae
            elif "RealUR5eDemo_20250801_185051" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[3][0][0])} deformability, {flip_level(buckets[3][1][0])} vulnerability, and {flip_level(buckets[3][2][0])} slipperiness object"
                #7i_kodawari_kakipi_oobukuro
            elif "RealUR5eDemo_20250801_203402" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[4][0][0])} deformability, {flip_level(buckets[4][1][0])} vulnerability, and {flip_level(buckets[4][2][0])} slipperiness object"
                #dars_white
            elif "RealUR5eDemo_20250801_205329" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[5][0][0])} deformability, {flip_level(buckets[5][1][0])} vulnerability, and {flip_level(buckets[5][2][0])} slipperiness object"
                #mens_biore_semgam_refill
            elif "RealUR5eDemo_20250804_172403" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[6][0][0])} deformability, {flip_level(buckets[6][1][0])} vulnerability, and {flip_level(buckets[6][2][0])} slipperiness object"
                #umiajisen_2
            elif "RealUR5eDemo_20250804_181153" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[7][0][0])} deformability, {flip_level(buckets[7][1][0])} vulnerability, and {flip_level(buckets[7][2][0])} slipperiness object"
                #maiji_milk_chocolate
            elif "RealUR5eDemo_20250804_184427" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[8][0][0])} deformability, {flip_level(buckets[8][1][0])} vulnerability, and {flip_level(buckets[8][2][0])} slipperiness object"
                #7i_ajitsuke_ponzu
            #elif "RealUR5eDemo_20250804_185508" in raw_file_string:
            #    task = "Grasp SEVEN&i PREMIUM Langue de Chat White Chocolate"
            elif "RealUR5eDemo_20250804_192155" in raw_file_string:
                task = f"Grasp the {flip_level(buckets[9][0][0])} deformability, {flip_level(buckets[9][1][0])} vulnerability, and {flip_level(buckets[9][2][0])} slipperiness object"
                #7i_pirittokarai_cheese_snacks
            else:
                task = "Grasp the object"
            

            imgs_per_cam, state, action, velocity, effort = load_raw_episode_data(ep_path)
            num_frames = state.shape[0]

            if ep_idx == 0:
                with open(dataset.root / "meta" / "modality.json", "w") as f:
                    modality = {
                        "state": {
                            "qpos": {
                                "start": 0,
                                "end": 7
                            }
                        },
                        "action": {
                            "action": {
                                "start": 0,
                                "end": 7
                            }
                        },
                        "video": {
                            "front_rgb": {
                                "original_key": "observation.images.front_rgb"
                            },
                            "hand_rgb": {
                                "original_key": "observation.images.hand_rgb"
                            }
                        },
                        "annotation": {
                            "human.action.task_description": {
                                "original_key": "task_index"
                            }
                        }
                    }
                    json.dump(modality, f, indent=4)

            for i in range(num_frames):
                frame = {
                    "observation.state": state[i],
                    "action": action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f"observation.images.{camera}"] = img_array[i]

                if velocity is not None:
                    frame["observation.velocity"] = velocity[i]
                if effort is not None:
                    frame["observation.effort"] = effort[i]
                if task is not None:
                    frame["task"] = task
                dataset.add_frame(frame)

            dataset.save_episode()

    return dataset


def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert c < h and c < w, f"expect channel first images, but instead {batch[key].shape}"

            # sanity check that images are float32 in range [0,1]
            assert batch[key].dtype == torch.float32, f"expect torch.float32, but instead {batch[key].dtype=}"
            assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"

            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns


def create_seeded_dataloader(dataset, batch_size, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    return dataloader


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

    For example:
    ```
    >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
    >>> print(flatten_dict(dct))
    {"a/b": 1, "a/c/d": 2, "e": 3}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: dict, sep: str = "/") -> dict:
    outdict = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = outdict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return outdict


def serialize_dict(stats: dict[str, torch.Tensor | np.ndarray | dict]) -> dict:
    serialized_dict = {key: value.tolist() for key, value in flatten_dict(stats).items()}
    return unflatten_dict(serialized_dict)



def port_hsr(
    raw_dir: Path,
    repo_id: str,
    raw_repo_id: str | None = None,
    task: str = "DEBUG",
    *,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    dataset_paths = sorted(raw_dir.glob("./*"))

    dataset = create_empty_dataset(
        repo_id,
        robot_type="hsr",
        mode=mode,
        has_effort=True,
        has_velocity=True,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        dataset_paths,
        task=task,
        episodes=episodes,
    )
    #dataset.consolidate()

    meta_stats = dataset.meta.stats

    stats_patterns = get_stats_einops_patterns(dataset, 8)

    data_num = len(dataset)
    q01, q99 = {}, {}
    data_dir = {}
    
    for key, pattern in stats_patterns.items():
        if key in dataset.meta.camera_keys:
            continue
        data_dir[key] = []
        for i in range(data_num):
            data_dir[key].append(dataset[i][key].float())
        data_dir[key] = torch.stack(data_dir[key], dim=0)
        
        q01[key] = torch.quantile(data_dir[key], 0.01, 0)
        q99[key] = torch.quantile(data_dir[key], 0.99, 0)
    
    for key in stats_patterns:
        if key in dataset.meta.camera_keys:
            continue
        meta_stats[key]["q01"] = q01[key]
        meta_stats[key]["q99"] = q99[key]

    serialized_stats = serialize_dict(meta_stats)
    
    with open(dataset.root / "meta" / "stats.json", "w") as f:
        json.dump(serialized_stats, f, indent=4)

    if push_to_hub:
        dataset.push_to_hub()

    print("Finished converting dataset.")


if __name__ == "__main__":
    tyro.cli(port_hsr)