#!/bin/bash
#SBATCH --job-name=viz_bees
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -o viz.out
#SBATCH -e viz.err

module load any/python/3.8.3-conda
source activate yolov10

cd ~/bee_project

python visualize_dataset.py
