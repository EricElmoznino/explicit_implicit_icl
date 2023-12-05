#!/usr/bin/env bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=23:59:00
#SBATCH --mem=16G

module load anaconda/3
module load cuda/11.2/cudnn/8.1
conda activate causal_icl

export PYTHONUNBUFFERED=1

python train.py experiment=linear_regression/onedim_implicit.yaml logger.name="implicit"
python train.py experiment=linear_regression/onedim_explicit_mlppred.yaml logger.name="explicit_mlp"
python train.py experiment=linear_regression/onedim_explicit_transformerpred.yaml logger.name="explicit_tsf"
python train.py experiment=linear_regression/onedim_explicit_affinepred.yaml logger.name="explicit_affine"
python train.py experiment=linear_regression/onedim_explicit_scrambledtransformerpred.yaml logger.name="explicit_scrambledtsf"