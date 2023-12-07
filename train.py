import hydra
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

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=False,
        **cfg.experiment.trainer,
    )
    trainer.fit(model=task, datamodule=datamodule)
    if cfg.save:
        trainer.save_checkpoint(
            f"{cfg.save_dir}/checkpoints/{cfg.experiment.task_name}/{cfg.experiment.name} - seed={cfg.seed}.ckpt"
        )


if __name__ == "__main__":
    train()
