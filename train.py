import os
import lightning.pytorch as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from setup import config
from utils.util_model import LightningTripletNet
from utils.util_dataset import LightningDataModule
import pandas as pd

def main():

    dataset = LightningDataModule(config)
    triplet_net = LightningTripletNet(config)
    
    tqdm_cb = pl.callbacks.TQDMProgressBar()
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        filename="{epoch:02d}_",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )
    early_stopping_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min"
    )
    lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    callbacks = [tqdm_cb, ckpt_cb, early_stopping_cb, lr_monitor_cb]
    logger = TensorBoardLogger(save_dir=os.path.join(config.base_dir, "lightning_logs"), name='triplet_net')

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
