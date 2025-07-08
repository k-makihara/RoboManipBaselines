#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=3:00:00
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

#python ./misc/rmb2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrTidyup_20250617_093824 --repo_id koshimaki/MujocoHsrTidyup_20250617_093824 --no_push_to_hub --mode video

#python ./misc/rmb2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrTidyup_20250617_093824_mod_multitask --repo_id koshimaki/MujocoHsrTidyup_20250617_093824_mod_multitask --no_push_to_hub --mode video

# python ./misc/rmb2lerobot_mt.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrTidyup_20250617_093824_mt --repo_id koshimaki/MujocoHsrTidyup_20250617_093824_mt --no_push_to_hub --mode video

python ./misc/hdf2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --repo_id koshimaki/RealUR5eDemo_20250624_190911_v2 --no_push_to_hub --mode video