import pickle
import cv2
import zarr
import os
import datetime
import numpy as np
from scipy.spatial.transform import Rotation as R

save = False

timestamp = timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print(timestamp)
#data_dir = "/home/veluga-g3/pg-vla/universal_manipulation_interface/test_data_filtered_curated"
data_dir = "/home/veluga-g3/pg-vla/universal_manipulation_interface/kmi-debug"
save_dir = f"/home/veluga-g3/pg-vla/RoboManipBaselines/robo_manip_baselines/dataset/RealUmiDemo_{timestamp}"
if save:
    os.mkdir(save_dir)
dataset_plan = pickle.load(open(f"{data_dir}/dataset_plan.pkl","rb"))
print(f"The number of episodes: {len(dataset_plan)}")

idx = 0
print(dataset_plan[idx]["grippers"][0]['tcp_pose'])
poses = np.array(dataset_plan[idx]["grippers"][0]['tcp_pose'])
translations = poses[:, :3]    # shape (N,3)
eulers       = poses[:, 3:]    # shape (N,3)

# 1) 各行の RPY → Rotation オブジェクト列に
rots = R.from_euler('xyz', eulers, degrees=False)

# 2) 基準
r0 = rots[0]       # Rotation object
t0 = translations[0]

# 3) 相対回転:  r0.inv() * rots[i]
rel_rots = r0.inv() * rots    # 形は (N,) の Rotation

# 4) 相対並進: r0⁻¹ を使ってベクトル差を基準フレームに射影
rel_trans = r0.inv().apply(translations - t0)  # shape (N,3)

# 5) 必要ならまた RPY に戻す
rel_euler = rel_rots.as_euler('xyz', degrees=False)  # shape (N,3)

# 6) 最終的な ΔPose をまとめる
delta_poses = np.hstack([rel_trans, rel_euler])  # shape (N,6)
print(delta_poses)



print(dataset_plan[idx]["grippers"][0]['gripper_width'])
print(dataset_plan[idx]["grippers"][0]['demo_start_pose'])
print(dataset_plan[idx]["grippers"][0]['demo_end_pose'])


print(dataset_plan[idx]["cameras"][0]["video_path"])
video_path = dataset_plan[idx]["cameras"][0]["video_path"]
cap = cv2.VideoCapture(f"{data_dir}/demos/{video_path}")
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total_frames}")

print(dataset_plan[idx]["cameras"][0]["video_start_end"][0])
print(dataset_plan[idx]["cameras"][0]["video_start_end"][1])

if save:
    start_idx = dataset_plan[idx]["cameras"][0]["video_start_end"][0]
    end_idx = dataset_plan[idx]["cameras"][0]["video_start_end"][1]

    fps    = cap.get(cv2.CAP_PROP_FPS)
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

    for idx in range(start_idx, end_idx + 1):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: frame {idx} could not be read.")
            break
        out.write(frame)
    out.release()
    print(f"Saved cropped video: {output_path}")

cap.release()
    

