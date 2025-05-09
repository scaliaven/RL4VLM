#!/bin/bash

#SBATCH --job-name=run_text_%j
#SBATCH --output=logs/run_text_%j.out
#SBATCH --error=logs/run_text_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --account=pr_133_tandon_advanced
#SBATCH --mail-type=all
#SBATCH --mail-user=hh3043@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate vlm4rl

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config_zero2.yaml --main_process_port 29500 ../main.py \
    --feature text \
    --init-lr 1e-5 \
    --end-lr 1e-9 \
    --lr_max_steps 25 \
    --eval-num-per-episode 200 \
    --num-env-steps 15000 \
    --num-steps 256 \
    --num-eval-steps 512 \
    --grad-accum-steps 64 \
    --max-new-tokens 256 \
    --thought-prob-coef 0.5 \
    --use-gae \
    --seed 1 \
    --temperature 0.2 \
    --ppo-epoch 4 \
    --mini-batch-size 1 \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --use-lora \
    --train-vision text \
    --env-name taxi \
    --use-wandb \
    # --action-only \
    # --env-name taxi \
    # --env-name gym_cards/NumberLine-v0 \
    # --env-name nlp \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4
