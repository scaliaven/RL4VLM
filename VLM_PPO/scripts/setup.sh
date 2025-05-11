#!/bin/bash

if [[ $(hostname -f) == *"hpc.nyu.edu"* ]]; then

  echo "setting up configs for NYU Greene"

  # == set up sbatch commands == #
  export ACCOUNT="pr_133_tandon_advanced"
  export DEFAULT_PARTITION="tandon_h100_1"

  # == set up conda commands == #
  # module load cuda/11.6.2
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
  source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
  conda activate rl4vlm
  # export LD_LIBRARY_PATH=$(python -c "import torch; import os; print(os.path.dirname(torch.__file__) + '/lib')"):$LD_LIBRARY_PATH

  # == set up directories == #
  export PROJ_DIR="/scratch/yx3038/Research/pruning/LLM-Shearing"
  export MODEL_DIR="/scratch/yx3038/model_ckpt"

elif [[ $(hostname -f) == *"ncsa.illinois.edu"* ]]; then

  echo "setting up configs for UIUC NCSA"

  # == set up sbatch commands == #
  export ACCOUNT="bdhh-delta-gpu"
  export DEFAULT_PARTITION="gpuA100x4"
  
  # == set up conda commands == #
  source /u/yxu21/miniforge3/etc/profile.d/conda.sh
  export LD_LIBRARY_PATH="/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku/lib64:${LD_LIBRARY_PATH}"
  conda activate llmshearing
  module load cuda/12.4.0
  module load gcc/11.4.0

  # == set up directories == #
  export PROJ_DIR="/scratch/bdhh/yxu21/pruning/LLM-Shearing"
  export MODEL_DIR="/scratch/bdhh/yxu21/model_ckpts"
fi

generate_sbatch_header() {
  local job_name="${1:-run_minitron2}"
  local output_dir="logs"
  local cpus_per_task="$2"
  local account="${ACCOUNT}"
  local partition="${6:-${DEFAULT_PARTITION}}"
  local mem="${3:-64GB}"
  local ntasks="1"
  local time="$4"
  local gpus="$5"
  local mail_user="yx3038@nyu.edu"

  export SBTACH_HEADER="#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --output=$output_dir/$job_name.out
#SBATCH --error=$output_dir/$job_name.err
#SBATCH --cpus-per-task=$cpus_per_task
#SBATCH --account=$account
#SBATCH --partition=$partition
#SBATCH --mem=${mem}GB
#SBATCH --ntasks=$ntasks
#SBATCH --time=$time:00:00
#SBATCH --gres=gpu:$gpus
#SBATCH --mail-type=all
#SBATCH --mail-user=$mail_user
#SBATCH --requeue"
}

check_sbash() {
  if [[ "$SBASH_MODE" == "1" ]]; then
    TMP_SCRIPT=$(mktemp /tmp/sbatch_script.XXXXXX.sh)

    # == generate sbatch header == #
    generate_sbatch_header "$@"
    echo "$SBTACH_HEADER" > "$TMP_SCRIPT"

    # == source setup.sh explicitly == #
    echo "source \"$(realpath "${BASH_SOURCE%/*}/../configs/setup.sh")\"" >> "$TMP_SCRIPT"

    # == append the current script to the tmp script == #
    awk '
      BEGIN { skip = 0 }
      /^\s*check_sbash/ { skip = 1; next }
      skip { print }
    ' "$0" >> "$TMP_SCRIPT"

    sbatch "$TMP_SCRIPT"
    cat $TMP_SCRIPT
    rm $TMP_SCRIPT
    exit 0
  fi
}

# == usage == #
# == bash your_script.sh: directly execute your script == #
# == sbash your_script.sh: submit your script with sbatch == #