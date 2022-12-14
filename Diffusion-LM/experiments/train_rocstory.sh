#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:a100:1
#SBATCH --reservation=DGXA100
#SBATCH --output=logs/%x-o%A_%a.log

set -ex

experiment_name=train_rocstory

echo Experiment ${experiment_name}: 98K five-sentence stories, capturing a rich set of causal and temporal commonsense relations between daily event

# Load singularity
module load singularity

# 1. Copy your container on the compute node
rsync -avzh --ignore-existing $SCRATCH/diffusion-lm.sif $SLURM_TMPDIR

# 2. Executing your code with singularity
singularity exec \
    -B $SLURM_TMPDIR:/dataset/ \
    -B $SLURM_TMPDIR:/tmp_log/ \
    -B $SCRATCH:/final_log/ \
    $SLURM_TMPDIR/diffusion-lm.sif \
    bash -c 'cd improved-diffusion && python scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 400000 --seed 101 --noise_schedule sqrt --in_channel 128 --modality roc --submit no --padding_mode pad --app "--predict_xstart True --training_mode e2e --vocab_size 11043 --roc_train ../datasets/ROCstory " --notes xstart_e2e --bsz 64'

# 3. Copy whatever you want to save on $SCRATCH
mkdir -p $SCRATCH/wandb $SCRATCH/diffusion_models
rsync -avz improved-diffusion/wandb/* $SCRATCH/wandb
rsync -avz improved-diffusion/diffusion_models/* $SCRATCH/diffusion_models
