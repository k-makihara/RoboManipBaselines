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

DATASET_ROOT=${1:-/groups/gag51454/workspace_makihara/dataset/lerobot_dataset/MujocoHsr_sb}

python ./misc/reencode_lerobot_videos_mp4v.py \
  --dataset-root "${DATASET_ROOT}"
