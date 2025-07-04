import json
import videoio
import torchcodec
import h5py
import os
import shutil
from tqdm import tqdm

dataset_dir = "/home/deepstation/rfm/RoboManipBaselines/robo_manip_baselines/dataset/MujocoHsrTidyup_20250617_093824_mod"
annotation = json.load(open("/home/deepstation/Downloads/project-1-at-2025-06-26-05-46-94ce636d.json","rb"))

os.makedirs(f"{dataset_dir}_multitask", exist_ok=True)

task_list = []
for i in range(len(annotation[0]['annotations'][0]["result"])):
    task_list.append(annotation[0]['annotations'][0]["result"][i]["value"]["timelinelabels"][0])
print(task_list)

for i in range(len(annotation)):
    demo_idx = annotation[i]['file_upload'].split("-")[-1][:-4]
    #print(f"Processing for {demo_idx[:-4]}")
    hand_rgb_video = torchcodec.decoders.VideoDecoder(f"{dataset_dir}/{demo_idx}/hand_rgb_image.rmb.mp4", dimension_order="NHWC")
    head_rgb_video = torchcodec.decoders.VideoDecoder(f"{dataset_dir}/{demo_idx}/head_rgb_image.rmb.mp4", dimension_order="NHWC")
    hand_depth_video = videoio.uint16read(f"{dataset_dir}/{demo_idx}/hand_depth_image.rmb.mp4")
    head_depth_video = videoio.uint16read(f"{dataset_dir}/{demo_idx}/head_depth_image.rmb.mp4")

    for j in range(len(annotation[i]['annotations'][0]["result"])):
        task_name = annotation[i]['annotations'][0]["result"][j]["value"]["timelinelabels"][0]
        #print(f"--- {task_name}")
        split_dir = f"{dataset_dir}_multitask/{demo_idx[:-4]}_task{task_list.index(task_name)}.rmb"
        #print("--- create dir")
        os.makedirs(split_dir, exist_ok=True)
        shutil.copy(f"{dataset_dir}/{demo_idx}/main.rmb.hdf5", f"{split_dir}/main.rmb.hdf5")
        start_idx = annotation[i]['annotations'][0]["result"][j]["value"]["ranges"][0]["start"] - 1
        end_idx = annotation[i]['annotations'][0]["result"][j]["value"]["ranges"][0]["end"]
        #print("--- create metadata")
        with h5py.File(f"{split_dir}/main.rmb.hdf5", "r+") as f:
            f.attrs["task_desc"] = task_name
            for key in f.keys():
                data = f[key][start_idx:end_idx]
                del f[key]
                f.create_dataset(key, data=data)
        #print("--- create video")
        hand_rgb_video_split = hand_rgb_video[start_idx:end_idx].numpy()
        head_rgb_video_split = head_rgb_video[start_idx:end_idx].numpy()
        hand_depth_video_split = hand_depth_video[start_idx:end_idx]
        head_depth_video_split = head_depth_video[start_idx:end_idx]
        videoio.videosave(f"{split_dir}/hand_rgb_image.rmb.mp4", hand_rgb_video_split)
        videoio.videosave(f"{split_dir}/head_rgb_image.rmb.mp4", head_rgb_video_split)
        videoio.uint16save(f"{split_dir}/hand_depth_image.rmb.mp4", hand_depth_video_split)
        videoio.uint16save(f"{split_dir}/head_depth_image.rmb.mp4", head_depth_video_split)
        #print("--- Done")
        



# video_rgb = torchcodec.decoders.VideoDecoder(path, dimension_order="NHWC")
# video_depth = videoio.uint16read(path)
# videoio.videosave(video_filename, video_rgb)
# videoio.uint16save(video_filename, video_rgb)