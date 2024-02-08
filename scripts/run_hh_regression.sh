#!/bin/bash
#SBATCH --job-name=int_opt
#SBATCH --output=/network/scratch/l/leo.gagnon/sbatch_output.txt
#SBATCH --error=/network/scratch/l/leo.gagnon/sbatch_error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu

source ~/explicit_implicit_icl/venv/bin/activate

for seed in 420 69 1337; do
    for style in 'same' 'near'; do
        python train.py experiment=hh_regression/implicit.yaml wandb_account=leogagnon save_dir=$SCRATCH seed=$seed +data.context_style=$style
    done
done
