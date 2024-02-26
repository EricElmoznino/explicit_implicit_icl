#!/usr/bin/env bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=23:59:00
#SBATCH --mem=16G

module load anaconda/3
module load cuda/11.2/cudnn/8.1
conda activate causal_icl

kernel=$1
context_style=$2
seed=$3

export PYTHONUNBUFFERED=1

python train.py experiment=gp_regression/implicit.yaml logger.name="implicit" experiment.data.x_dim=1 experiment.task.model.x_dim=1 experiment.data.y_dim=1 experiment.task.model.y_dim=1 experiment.data.context_style=$context_style seed=$seed experiment.data.kind_kwargs.kernel=$kernel
python train.py experiment=gp_regression/explicit_mlp.yaml logger.name="explicit_mlp" experiment.data.x_dim=1 experiment.task.model.context_model.x_dim=1 experiment.task.model.prediction_model.x_dim=1 experiment.data.y_dim=1 experiment.task.model.context_model.y_dim=1 experiment.task.model.prediction_model.y_dim=1 experiment.data.context_style=$context_style seed=$seed experiment.data.kind_kwargs.kernel=$kernel
python train.py experiment=gp_regression/explicit_tsf.yaml logger.name="explicit_tsf" experiment.data.x_dim=1 experiment.task.model.context_model.x_dim=1 experiment.task.model.prediction_model.x_dim=1 experiment.data.y_dim=1 experiment.task.model.context_model.y_dim=1 experiment.task.model.prediction_model.y_dim=1 experiment.data.context_style=$context_style seed=$seed experiment.data.kind_kwargs.kernel=$kernel