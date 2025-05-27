# Does learning the right latent variables necessarily improve in-context learning?

## Configuring and running experiments

We use `hydra` to configure experiments. Configurations for individual experiments are defined in `configs/experiment/`. To run an experiment, you'll want to specify the experiment configuration along with other configuration arguments at the command line. For instance:

```python train.py experiment=linear_regression/implicit wandb_account=[wandb account] save_dir=[cluster_path] seed=101```

## Setup
- `python3.10 -v venv venv`
- `pip install -r requirements.txt`