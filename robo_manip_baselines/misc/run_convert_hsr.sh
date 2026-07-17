#!/bin/sh
#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -P gag51454
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
cd /groups/gag51454/makihara-backup-v2/physical-grounding/physical-grounding/makihara/RoboManipBaselines/robo_manip_baselines
module load nvhpc/24.9
module load hpcx/2.20
source ~/.bashrc
conda activate rmb


export HF_LEROBOT_HOME="/groups/gag51454/workspace_makihara/dataset/lerobot_dataset"

# python ./misc/rmb2lerobot.py --raw_dir /groups/gag51454/workspace_makihara/dataset/MujocoHsr_makihara --repo_id MujocoHsr_makihara --no_push_to_hub --mode video

python ./misc/rmb2lerobot.py --raw_dir /groups/gag51454/workspace_makihara/dataset/MujocoHsr --repo_id MujocoHsr_sb --no_push_to_hub --mode video