#!/bin/bash

python ./bin/Train.py MtAct --dataset_dir dataset/MujocoHsrTidyup_20250617_093824_mt --camera_names head hand --train_ratio 1.0 --chunk_size 8 --state_keys measured_joint_pos measured_gripper_joint_pos measured_mobile_omni_vel --action_keys command_joint_pos command_gripper_joint_pos command_mobile_omni_vel --batch_size 8

python ./bin/Train.py MtAct --dataset_dir dataset/MujocoHsrTidyup_20250617_093824_mt --camera_names head hand --train_ratio 1.0 --chunk_size 16 --state_keys measured_joint_pos measured_gripper_joint_pos measured_mobile_omni_vel --action_keys command_joint_pos command_gripper_joint_pos command_mobile_omni_vel --batch_size 8

python ./bin/Train.py MtAct --dataset_dir dataset/MujocoHsrTidyup_20250617_093824_mt --camera_names head hand --train_ratio 1.0 --chunk_size 24 --state_keys measured_joint_pos measured_gripper_joint_pos measured_mobile_omni_vel --action_keys command_joint_pos command_gripper_joint_pos command_mobile_omni_vel --batch_size 8

python ./bin/Train.py MtAct --dataset_dir dataset/MujocoHsrTidyup_20250617_093824_mt --camera_names head hand --train_ratio 1.0 --chunk_size 32 --state_keys measured_joint_pos measured_gripper_joint_pos measured_mobile_omni_vel --action_keys command_joint_pos command_gripper_joint_pos command_mobile_omni_vel --batch_size 8