#!/bin/bash
#SBATCH -J bee_yolo10
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 7:59:00
#SBATCH -o bee_train.out
#SBATCH -e bee_train.err

module load cuda/12.2
module load anaconda

source activate yolov10

cd ~/bee_project

echo "Starting training..."
python train_bees.py --epochs 200 --batch 16 --imgsz 1280 --device 0
echo "Done."
