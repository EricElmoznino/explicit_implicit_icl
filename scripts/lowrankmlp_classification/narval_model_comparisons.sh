python train.py --multirun hydra/launcher=narval_eric save_dir=/scratch/elmo/explicit_implicit_icl/logs logger.offline=True experiment=lowrankmlp_classification/implicit,lowrankmlp_classification/explicit_tsf,lowrankmlp_classification/explicit_mlp,lowrankmlp_classification/explicit_known,lowrankmlp_classification/known_latent ++experiment.data.context_style=same,near seed=0,1,2,3,4 ++experiment.callbacks=False ++experiment.data.temperature=0.000001