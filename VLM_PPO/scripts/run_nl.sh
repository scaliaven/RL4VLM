#!/bin/bash

#SBATCH --job-name=train_svd
#SBATCH --output=logs/train_svd.out
#SBATCH --error=logs/train_svd.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --account=pr_133_tandon_advanced
#SBATCH --mail-type=all
#SBATCH --mail-user=hh3043@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate vlm4rl

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config_zero2.yaml --main_process_port 29488 ../main.py \
    --feature image \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 512 \
    --grad-accum-steps 128 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --use-lora \
    --train-vision all \
    --env-name taxi \
    # --env-name gym_cards/NumberLine-v0 \
    # --env-name nlp \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4
