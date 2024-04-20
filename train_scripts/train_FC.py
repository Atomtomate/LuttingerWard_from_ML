
import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE_FC import AE_FC_01
sys.path.append(join(dirname(__file__),'../code/models/IO'))
from DataMod_FC import *

import pytorch_lightning as L
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler, RichModelSummary, DeviceStatsMonitor
from pytorch_lightning.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser

import json

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float64)


def main(args):
    config = json.load(open(join(dirname(__file__),'../configs/confmod_AE_FC.json')))
    torch.manual_seed(config['seed'])
    for AE_layers in [1,2,3]:
        for FC_layers in [1,2,3,4,5,6,7,8]:
                config['AE_layers'] = AE_layers
                config['FC_layers'] = FC_layers
                model = AE_FC_01(config) 
                dataMod = DataMod_FC(config)

                lr_monitor = LearningRateMonitor(logging_interval='step')
                logger = TensorBoardLogger("lightning_logs", name="VAE_Linear")
                val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
                        filename="{epoch}-{step}-{val_loss:.8f}",
                        monitor="val_loss",
                        mode="min",
                        save_top_k=2,
                        save_last =True
                        )
                early_stopping = EarlyStopping(monitor="val_loss",patience=40)
                swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220)
                accumulator = GradientAccumulationScheduler(scheduling={0: 128, 12: 64, 16: 32, 24: 16, 32: 8, 40: 4, 48: 1})
                callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
                trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"],
                                callbacks=callbacks, logger=logger, gradient_clip_val=0.5) #precision="16-mixed", 

                trainer.fit(model, datamodule=dataMod)

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)