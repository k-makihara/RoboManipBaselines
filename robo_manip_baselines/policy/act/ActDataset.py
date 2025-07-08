import numpy as np
import torch
from torch.utils.data import get_worker_info

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    get_skipped_data_seq,
    get_skipped_single_data,
)


class ActDataset(DatasetBase):
    """Dataset to train ACT policy."""

    def __len__(self):
        #print("dataset length:", len(self.filenames))
        return len(self.filenames)

    def __getitem__(self, episode_idx):
        skip = self.model_meta_info["data"]["skip"]
        chunk_size = self.model_meta_info["data"]["chunk_size"]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            start_time_idx = np.random.choice(episode_len)

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        get_skipped_single_data(
                            rmb_data[key], start_time_idx * skip, key, skip
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load action
            action = np.concatenate(
                [
                    get_skipped_data_seq(
                        rmb_data[key][start_time_idx * skip :],
                        key,
                        skip,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            image_keys = [
                DataKey.get_rgb_image_key(camera_name)
                for camera_name in self.model_meta_info["image"]["camera_names"]
            ]
            images = np.stack(
                [
                    # This allows for a common hash of cache
                    rmb_data[key][::skip][start_time_idx]
                    if self.enable_rmb_cache
                    # This allows for minimal loading when reading from HDF5
                    else rmb_data[key][start_time_idx * skip]
                    for key in image_keys
                ],
                axis=0,
            )

        # Chunk action
        action_len = action.shape[0]
        action_chunked = np.zeros((chunk_size, action.shape[1]), dtype=np.float64)
        action_chunked[:action_len] = action[:chunk_size]
        is_pad = np.zeros(chunk_size, dtype=bool)
        is_pad[action_len:] = True

        # Pre-convert data
        state, action_chunked, images = self.pre_convert_data(
            state, action_chunked, images
        )

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action_chunked, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return state_tensor, images_tensor, action_tensor, is_pad_tensor

# class ActDataset(DatasetBase):
#     """Dataset to train ACT policy, with correct RmbData caching and manual enter/exit."""

#     def __init__(self, filenames, model_meta_info, enable_rmb_cache=False):
#         super().__init__(filenames, model_meta_info, enable_rmb_cache)
#         # (worker_id, episode_idx) → (RmbData_handle, rmb_data_obj)
#         self._rmb_cache: dict[tuple[int,int], tuple[RmbData, any]] = {}

#     def __len__(self):
#         return len(self.filenames)

#     def __getitem__(self, episode_idx):
#         # 1) ワーカーIDを取得
#         worker = get_worker_info()
#         wid = worker.id if worker is not None else 0
#         key = (wid, episode_idx)

#         # 2) キャッシュがなければ新規に open (__enter__) して保持
#         if key not in self._rmb_cache:
#             handle = RmbData(self.filenames[episode_idx], self.enable_rmb_cache)
#             rmb_obj = handle.__enter__()   # __enter__() で self.h5file を正しく初期化
#             self._rmb_cache[key] = (handle, rmb_obj)
#         handle, rmb = self._rmb_cache[key]

#         skip       = self.model_meta_info["data"]["skip"]
#         chunk_size = self.model_meta_info["data"]["chunk_size"]

#         # 3) 時間軸長さとランダム開始位置
#         episode_len   = rmb[DataKey.TIME][::skip].shape[0]
#         start_idx     = np.random.randint(episode_len)

#         # 4) state 読み込み
#         if not self.model_meta_info["state"]["keys"]:
#             state_np = np.zeros(0, dtype=np.float64)
#         else:
#             state_np = np.concatenate([
#                 get_skipped_single_data(
#                     rmb[k], start_idx * skip, k, skip
#                 )
#                 for k in self.model_meta_info["state"]["keys"]
#             ])

#         # 5) action 読み込み
#         action_np = np.concatenate([
#             get_skipped_data_seq(
#                 rmb[k][start_idx * skip :],
#                 k, skip
#             )
#             for k in self.model_meta_info["action"]["keys"]
#         ], axis=1)

#         # 6) images 読み込み
#         image_keys = [
#             DataKey.get_rgb_image_key(cam)
#             for cam in self.model_meta_info["image"]["camera_names"]
#         ]
#         images_np = np.stack([
#             (rmb[k][::skip][start_idx]
#              if self.enable_rmb_cache
#              else rmb[k][start_idx * skip])
#             for k in image_keys
#         ], axis=0)

#         # 7) actionチャンク＆マスク
#         action_len     = action_np.shape[0]
#         action_chunked = np.zeros((chunk_size, action_np.shape[1]), dtype=np.float64)
#         action_chunked[:action_len] = action_np[:chunk_size]
#         is_pad_np = np.zeros(chunk_size, dtype=bool)
#         if action_len < chunk_size:
#             is_pad_np[action_len:] = True

#         # 8) 前処理
#         state_np, action_chunked, images_np = self.pre_convert_data(
#             state_np, action_chunked, images_np
#         )

#         # 9) ゼロコピーでテンソル化
#         state_t  = torch.from_numpy(state_np.astype(np.float32))
#         action_t = torch.from_numpy(action_chunked.astype(np.float32))
#         images_t = torch.from_numpy(images_np)
#         pad_t    = torch.from_numpy(is_pad_np)

#         # 10) augmentation（テンソル上で）
#         state_t, action_t, images_t = self.augment_data(
#             state_t, action_t, images_t
#         )
#         return state_t, images_t, action_t, pad_t

#     def __del__(self):
#         # 終了時に全ハンドルの __exit__() を呼び出してクローズ
#         for handle, _ in self._rmb_cache.values():
#             try:
#                 handle.__exit__(None, None, None)
#             except Exception:
#                 pass
