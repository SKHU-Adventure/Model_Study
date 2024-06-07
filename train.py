import os
import lightning.pytorch as pl

from setup import config
from utils.util_model import LightningTripletNet
from utils.util_dataset import LightningDataModule
from utils.util_lightning import get_callbacks, get_logger


def main():

    dataset = LightningDataModule(config)
    triplet_net = LightningTripletNet(config)
    callbacks = get_callbacks(config)
    logger = get_logger(config)

    trainer = pl.Trainer(accelerator="gpu",
                        devices=config.gpu_ids,
                        strategy="ddp",
                        max_epochs=config.total_epoch,
                        use_distributed_sampler=True,
                        precision="16-mixed",
                        callbacks=callbacks,
                        logger=logger,
                        profiler="simple",
                        log_every_n_steps=1,
                        default_root_dir=config.base_dir)
    trainer.fit(triplet_net, dataset)

if __name__ == '__main__':
    main()
