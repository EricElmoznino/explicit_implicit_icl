python train.py --multirun hydra/launcher=mila_eric save_dir=/home/mila/e/eric.elmoznino/scratch/explicit_implicit_icl/logs experiment=lowrankmlp_classification/implicit,lowrankmlp_classification/explicit_tsf,lowrankmlp_classification/explicit_mlp ++experiment.data.context_style=same,near seed=0,1,2,3,4 ++experiment.callbacks=False ++experiment.data.temperature=0.000001