#!/usr/bin/env bash
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --time=23:59:00
#SBATCH --mem=24G
#SBATCH -c 3

module load anaconda/3
module load cuda/11.2/cudnn/8.1
conda activate causal_icl

export PYTHONUNBUFFERED=1

x_dim=$1
y_dim=$2
context_style=$3
seed=$4

python train.py experiment=mlp_classification/implicit.yaml logger.name="implicit" experiment.data.x_dim=$x_dim experiment.task.model.x_dim=$x_dim experiment.data.y_dim=$y_dim experiment.task.model.y_dim=$y_dim experiment.data.context_style=$context_style seed=$seed
python train.py experiment=mlp_classification/explicit_mlp.yaml logger.name="explicit_mlp" experiment.data.x_dim=$x_dim experiment.task.model.context_model.x_dim=$x_dim experiment.task.model.prediction_model.x_dim=$x_dim experiment.data.y_dim=$y_dim experiment.task.model.context_model.y_dim=$y_dim experiment.task.model.prediction_model.y_dim=$y_dim experiment.data.context_style=$context_style seed=$seed
python train.py experiment=mlp_classification/explicit_tsf.yaml logger.name="explicit_tsf" experiment.data.x_dim=$x_dim experiment.task.model.context_model.x_dim=$x_dim experiment.task.model.prediction_model.x_dim=$x_dim experiment.data.y_dim=$y_dim experiment.task.model.context_model.y_dim=$y_dim experiment.task.model.prediction_model.y_dim=$y_dim experiment.data.context_style=$context_style seed=$seed