#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=/network/scratch/l/leo.gagnon/sbatch_output.txt
#SBATCH --error=/network/scratch/l/leo.gagnon/sbatch_error.txt
#SBATCH --ntasks=1
#SBATCH --time=5:00:00
#SBATCH --mem=32G

python train.py --multirun hydra/launcher=mila_leo save_dir=/home/mila/l/leo.gagnon/scratch/explicit_implicit_icl/logs experiment=linear_regression/implicit,linear_regression/explicit_tsf,linear_regression/explicit_mlp,linear_regression/explicit_aux_tsf  seed=0,1,2 ++experiment.data.noise=0.0 logger.log_model=True
