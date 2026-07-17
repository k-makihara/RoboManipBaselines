#!/bin/sh
#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=12:00:00
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

# python ./misc/rmb2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrTidyup_20250617_093824_mod_multitask --repo_id koshimaki/MujocoHsrTidyup_20250617_093824_mod_multitask --no_push_to_hub --mode video

# python ./misc/rmb2lerobot_mt.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrTidyup_20250617_093824_mt --repo_id koshimaki/MujocoHsrTidyup_20250617_093824_mt --no_push_to_hub --mode video

#python ./misc/hdf2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --repo_id koshimaki/RealUR5eDemo_20250624_190911_v2 --no_push_to_hub --mode video

# python ./misc/hdf2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_20250624_190911 --repo_id koshimaki/RealUR5eDemo_20250624_190911_v3 --no_push_to_hub --mode video

# export HF_LEROBOT_HOME="/groups/gaf51379/physical-grounding/datasets/lerobot_dataset"
# python ./misc/rmb2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrPaP_20250715_163442 --repo_id koshimaki/MujocoHsrPaP_20250715_163442 --no_push_to_hub --mode video

# python ./misc/rmb2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrShelfPaP_20250715_170353 --repo_id koshimaki/MujocoHsrShelfPaP_20250715_170353 --no_push_to_hub --mode video

# python ./misc/rmb2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrShelfRealPaP_20250715_180450 --repo_id koshimaki/MujocoHsrShelfRealPaP_20250715_180450 --no_push_to_hub --mode video

#python ./misc/rmb2lerobot_ki.py --raw_dir /groups/gag51454/data/RoboManipBaselines/MujocoUR5eDoor_Ep200_20250516/MujocoUR5eDoor_20250515_120404 --repo_id koshimaki/MujocoUR5eDoor_20250515_120404 --no_push_to_hub --mode video


# python ./misc/rmb2lerobot_sb.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_sb_10task --repo_id koshimaki/RealUR5eDemo_sb_10task --no_push_to_hub --mode video

#python ./misc/rmb2lerobot_sb_pg.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_sb_10task --repo_id koshimaki/RealUR5eDemo_sb_10task_pg --no_push_to_hub --mode video

#python ./misc/rmb2lerobot_sb_pg_diffstate.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_sb_10task --repo_id koshimaki/RealUR5eDemo_sb_10task_pg_diffstate --no_push_to_hub --mode video

#python ./misc/rmb2lerobot_mtpick.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoUR5ePick --repo_id koshimaki/MujocoUR5ePick_mtpick --no_push_to_hub --mode video

#python ./misc/rmb2lerobot_sb_name_diffstate.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_sb_10task --repo_id koshimaki/RealUR5eDemo_sb_10task_name_diffstate --no_push_to_hub --mode video

# python ./misc/rmb2lerobot_sb_detailname_diffstate.py --raw_dir /groups/gaf51379/physical-grounding/datasets/RealUR5eDemo_sb_10task --repo_id koshimaki/RealUR5eDemo_sb_10task_detailname_diffstate --no_push_to_hub --mode video


export HF_LEROBOT_HOME="/groups/gag51454/workspace_makihara/dataset/lerobot_dataset"

python ./misc/rmb2lerobot.py --raw_dir /groups/gaf51379/physical-grounding/datasets/MujocoHsrShelfPaP_20250715_170353 --repo_id koshimaki/MujocoHsrShelfPaP_20250715_170353 --no_push_to_hub --mode video