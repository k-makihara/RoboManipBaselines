import glob
import os
import subprocess
data_path = "/home/deepstation/rfm/RoboManipBaselines/robo_manip_baselines/dataset/MujocoHsrTidyup_20250617_093824_mod"
dataset_paths = glob.glob(f"{data_path}/*")

output_path = f"{data_path}_annotation"
os.makedirs(output_path, exist_ok=True)

for dataset_path in dataset_paths:
    subprocess.run(["cp", f"{dataset_path}/hand_rgb_image.rmb.mp4", f"{output_path}/{os.path.basename(dataset_path)}.mp4"])
    
