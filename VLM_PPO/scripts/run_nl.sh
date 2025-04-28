#!/bin/bash

#SBATCH --job-name=run_nl_%j
#SBATCH --output=logs/run_nl_%j.out
#SBATCH --error=logs/run_nl_%j.err
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

#SBATCH --account=bdhh-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=all
#SBATCH --mail-user=yx3038@nyu.edu
#SBATCH --no-requeue

source /u/yxu21/miniforge3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH="/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku/lib64:${LD_LIBRARY_PATH}"
conda activate rl4vlm
module load cuda/12.4.0
module load gcc/11.4.0

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file config_zero2.yaml --main_process_port 29488 ../main.py \
    --feature tensor \
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
    --env-name junqi \
    # --env-name nlp \
    # --env-name gym_cards/NumberLine-v0 \
    # --env-name nlp \
    # --wandb-project you_wandb_proj \
    # --wandb-run you_wandb_run \
    # --use-wandb \
    # --q4
