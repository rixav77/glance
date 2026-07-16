#!/bin/bash
#SBATCH --job-name=scene_audit
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=eval/scene_audit.%j.log
source /home/shivank_g/anaconda3/etc/profile.d/conda.sh
conda activate vlm_rl
cd /home/shivank_g/projects/ml4psi/Glance
python eval/audit_scenes.py
