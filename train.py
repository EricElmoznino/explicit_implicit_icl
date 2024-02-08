import hydra
from omegaconf import OmegaConf
from lightning import Trainer, seed_everything


@hydra.main(config_path="./configs/", config_name="train", version_base=None)
def train(cfg):
    seed_everything(cfg.seed)

    datamodule = hydra.utils.instantiate(cfg.experiment.data)
    task = hydra.utils.instantiate(cfg.experiment.task)
    logger = hydra.utils.instantiate(cfg.logger) if cfg.logger else False
    callbacks = (
        hydra.utils.instantiate(cfg.experiment.callbacks)
        if cfg.experiment.callbacks
        else None
    )

    # Add experiment metadata to the logger
    if logger:
        logger.experiment.config.update(
            {
                "dataset": cfg.experiment.dataset,
                "model": cfg.experiment.model,
            }
        )
        logger.experiment.config.update(
            {
                "model_config": OmegaConf.to_container(
                    cfg.experiment.task.model, resolve=True
                )
            }
        )

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.experiment.trainer,
    )
    trainer.fit(model=task, datamodule=datamodule)


if __name__ == "__main__":
    train()
