import glob
import h5py
data_path = "/home/deepstation/rfm/RoboManipBaselines/robo_manip_baselines/dataset/MujocoHsrTidyup_20250617_093824_mt"
dataset_paths = glob.glob(f"{data_path}/*")
task_name = 'Bring the blue object to the box'
for dataset_path in dataset_paths:
    with h5py.File(f"{dataset_path}/main.rmb.hdf5", "r+") as f:
        f.attrs["task_desc"] = task_name

