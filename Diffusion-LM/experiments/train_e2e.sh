#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --mem-per-gpu=32G
#sbatch --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --output=logs/%x-o%A_%a.log

set -ex

experiment_name=train_e2e

export HUGGINGFACE_CACHE=$SLURM_TMPDIR/.cache
export HF_DATASETS_CACHE=$SLURM_TMPDIR/.cache/datasets

echo Experiment ${experiment_name}: 50K restaurant reviews labeled by 8 fields including food type, price, and customer rating

# Load singularity
module load singularity

# 1. Copy your container on the compute node
rsync -avzh --ignore-existing $SCRATCH/diffusion-lm.sif $SLURM_TMPDIR

# 1.1 Show the tmpdir
tree $SLURM_TMPDIR

# 2. Executing your code with singularity
singularity exec --nv \
    -B $SLURM_TMPDIR:/dataset/ \
    -B $SLURM_TMPDIR:/tmp_log/ \
    -B $SCRATCH:/final_log/ \
    $SLURM_TMPDIR/diffusion-lm.sif \
    bash -c 'cd improved-diffusion && mpiexec -n 8 python -m mpi4py scripts/run_train.py --diff_steps 2000 --model_arch transformer --lr 0.0001 --lr_anneal_steps 200000 --seed 102 --noise_schedule sqrt --in_channel 16 --modality e2e-tgt --submit no --padding_mode block --app "--predict_xstart True --training_mode e2e --vocab_size 821 --e2e_train ../datasets/e2e_data " --notes xstart_e2e'
m2
# 3. Copy whatever you want to save on $SCRATCH
mkdir -p $SCRATCH/wandb $SCRATCH/diffusion_models
rsync -avz improved-diffusion/wandb/* $SCRATCH/wandb
rsync -avz improved-diffusion/diffusion_models/* $SCRATCH/diffusion_models
