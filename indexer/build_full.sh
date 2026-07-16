#!/bin/bash
#SBATCH --job-name=glance_index
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=indexer/build_full.%j.log
source /home/shivank_g/anaconda3/etc/profile.d/conda.sh
cd /home/shivank_g/projects/ml4psi/Glance
/home/shivank_g/.conda/envs/vlm_rl/bin/python -m indexer.build_index
