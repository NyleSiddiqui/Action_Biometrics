#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -C gmem32
#SBATCH -c 6
#SBATCH -p gpu
#SBATCH --job-name=ntu_rgbd_120_rgb_5layers_run2
#SBATCH --output=./logs/%x.out

# Load modules
module load anaconda3
pip install einops --user

module list                            # Have Nvidia tell us the GPU/CPU mapping so we know

nvidia-smi topo -m

python3 --version

#python3 main.py --train_classifier --gpu 0 --run_id ntu_rgbd_120_rgb_5layers_run1 --run_description "Experiment with five layer ViViT model." --dataset ntu_rgbd_120 --model_version 'baseline' --input_type "rgb" --num_frames 32 --input_dim 224 --patch_size 16 --num_heads 8 --num_layers 5 --batch_size 8 --num_epochs 50 --num_workers 4 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM --f1_threshold 0.5 --skip 0 --steps 1

#python3 main.py --train_classifier --gpu 0 --run_id ntu_rgbd_120_rgb_5layers_run2 --run_description "Experiment with five layer ViViT model. Using decord to load the frames." --dataset pkummd  --model_version 'baseline' --input_type "rgb" --num_frames 32 --input_dim 224 --patch_size 16 --num_heads 8 --num_layers 5 --batch_size 8 --num_epochs 50 --num_workers 0 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM --f1_threshold 0.5 --skip 0 --steps 1

python3 dataloader.py
