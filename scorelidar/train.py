import click
from os.path import join, dirname, abspath
from os import environ, makedirs
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import torch
import yaml
import MinkowskiEngine as ME

import scorelidar.datasets.datasets as datasets
import scorelidar.models.models as models

def set_deterministic(seed=42):
    np.random.seed(seed)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

@click.command()
### Add your options here
@click.option('--config',
              '-c',
              type=str,
              help='path to the config file (.yaml)',
              default=join(dirname(abspath(__file__)),'config/config.yaml'))
@click.option('--weights',
              '-w',
              type=str,
              help='path to pretrained weights (.ckpt)',
              default=join(dirname(abspath(__file__)),'checkpoints/diff_net.ckpt'))
def main(config, weights):

    cfg = yaml.safe_load(open(config))
    # overwrite the data path in case we have defined in the env variables
    if environ.get('TRAIN_DATABASE'):
        cfg['data']['data_dir'] = environ.get('TRAIN_DATABASE')

    #Load data and model
    model = models.ScoreLiDAR(diff_path=weights, cfg=cfg)

    print(cfg)
    data = datasets.dataloaders[cfg['data']['dataloader']](cfg)

    #Add callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_saver = ModelCheckpoint(
                                 filename=cfg['experiment']['id']+'_{epoch:02d}',
                                 save_top_k=-1,
                                 every_n_train_steps=20
                                 )

    tb_logger = pl_loggers.TensorBoardLogger('experiments/'+cfg['experiment']['id'],
                                             default_hp_metric=False)
    #Setup trainer
    trainer = Trainer(gpus=cfg['train']['n_gpus'],
                        logger=tb_logger,
                        log_every_n_steps=5,
                        max_epochs= cfg['train']['max_epoch'],
                        callbacks=[lr_monitor, checkpoint_saver],
                        num_sanity_val_steps=0,
                        limit_val_batches=0.001,
                        val_check_interval=20
                        )


    # Train!
    print('TRAINING!')
    trainer.fit(model, data)

if __name__ == "__main__":
    main()
