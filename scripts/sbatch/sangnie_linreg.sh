x_dim=$1
echo $1
echo $x_dim
if [  $x_dim -gt 1 ]; then
	ts=10000
else
	ts=1000
fi
python train.py -m  hydra/launcher=mila_sangnie experiment=linear_regression/implicit  ++experiment.data.x_dim=$x_dim seed=0,1,2,3,4 ++experiment.data.context_style=same,near  ++experiment.data.train_size=$ts
#python train.py -m hydra/launcher=mila_sangnie  experiment=linear_regression/explicit_tsf.yaml   experiment.data.x_dim=$x_dim experiment.task.model.context_model.x_dim=$x_dim experiment.task.model.prediction_model.x_dim=$x_dim seed=0,1,2,3,4 experiment.data.context_style=same,near  experiment.data.train_size=$ts
