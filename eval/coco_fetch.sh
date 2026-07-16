#!/bin/bash
#SBATCH --job-name=coco_fetch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=eval/coco_fetch.%j.log
source /home/shivank_g/anaconda3/etc/profile.d/conda.sh
conda activate vlm_rl
cd /home/shivank_g/projects/ml4psi/Glance
python eval/fetch_coco_images.py
