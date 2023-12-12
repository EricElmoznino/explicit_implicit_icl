#!/usr/bin/env bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=47:59:00
#SBATCH --mem=16G

module load anaconda/3
module load cuda/11.2/cudnn/8.1
conda activate causal_icl

export PYTHONUNBUFFERED=1

y_dim=$1

for x_dim in 2 8; do
    python train.py experiment=linear_classification/onedim_implicit.yaml logger.name="implicit" experiment.data.x_dim=$x_dim experiment.task.model.x_dim=$x_dim experiment.data.y_dim=$y_dim experiment.task.model.y_dim=$y_dim
    python train.py experiment=linear_classification/onedim_explicit_mlppred.yaml logger.name="explicit_mlp" experiment.data.x_dim=$x_dim experiment.task.model.context_model.x_dim=$x_dim experiment.task.model.prediction_model.x_dim=$x_dim experiment.data.y_dim=$y_dim experiment.task.model.context_model.y_dim=$y_dim experiment.task.model.prediction_model.y_dim=$y_dim
    python train.py experiment=linear_classification/onedim_explicit_transformerpred.yaml logger.name="explicit_tsf" experiment.data.x_dim=$x_dim experiment.task.model.context_model.x_dim=$x_dim experiment.task.model.prediction_model.x_dim=$x_dim experiment.data.y_dim=$y_dim experiment.task.model.context_model.y_dim=$y_dim experiment.task.model.prediction_model.y_dim=$y_dim
    python train.py experiment=linear_classification/onedim_explicit_affinepred.yaml logger.name="explicit_affine" experiment.data.x_dim=$x_dim experiment.task.model.context_model.x_dim=$x_dim experiment.task.model.prediction_model.x_dim=$x_dim experiment.data.y_dim=$y_dim experiment.task.model.context_model.y_dim=$y_dim experiment.task.model.prediction_model.y_dim=$y_dim
done