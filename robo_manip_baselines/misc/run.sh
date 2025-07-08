#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -P gag51454
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
cd /groups/gaf51379/physical-grounding/makihara/RoboManipBaselines/robo_manip_baselines
module load nvhpc/24.9
module load hpcx/2.20
source ~/.bashrc
conda activate rmb

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5Demo_env1 --camera_names front hand --train_ratio 1.0 --chunk_size 20

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5Demo_env2 --camera_names front hand --train_ratio 1.0 --chunk_size 20

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5Demo_env3 --camera_names front hand --train_ratio 1.0 --chunk_size 20

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5Demo_env4 --camera_names front hand --train_ratio 1.0 --chunk_size 20

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5Demo_env5 --camera_names front hand --train_ratio 1.0 --chunk_size 20

#python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --camera_names front hand --train_ratio 1.0 --chunk_size 8

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --camera_names front hand --train_ratio 1.0 --chunk_size 16 --action_keys measured_joint_pos measured_gripper_joint_pos --state_keys measured_joint_pos measured_gripper_joint_pos

python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --camera_names front hand --train_ratio 1.0 --chunk_size 16 --action_keys measured_joint_pos --state_keys measured_joint_pos

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --camera_names front hand --train_ratio 1.0 --chunk_size 24

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --camera_names front hand --train_ratio 1.0 --chunk_size 32

# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrTidyup_20250624_221800_world0_5 --camera_names head hand --train_ratio 1.0 --chunk_size 8 --state_keys measured_joint_pos measured_gripper_joint_pos measured_mobile_omni_vel --action_keys command_joint_pos command_gripper_joint_pos command_mobile_omni_vel


### channot use in Mujoco
# python ./bin/Train.py Act --dataset_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrTidyup_20250624_221800 --camera_names head hand --train_ratio 1.0 --chunk_size 8 --state_keys measured_eef_pose_rel measured_gripper_joint_pos measured_mobile_omni_vel --action_keys command_eef_pose_rel command_gripper_joint_pos command_mobile_omni_vel