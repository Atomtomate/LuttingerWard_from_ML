

import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE import AutoEncoder_01
sys.path.append(join(dirname(__file__),'../code/models/IO'))
from DataMod_AE import *

import pytorch_lightning as L
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler, RichModelSummary, DeviceStatsMonitor
from pytorch_lightning.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import ArgumentParser


import json

torch.set_float32_matmul_precision("highest")
torch.set_default_dtype(torch.float64)


def main(args):
    config = json.load(open(join(dirname(__file__),'../configs/confmod_AE_SE_tmp.json')))
    torch.manual_seed(config['seed'])
    model = AutoEncoder_01(config) 
    dataMod = DataMod_AE(config)
    val_ckeckpoint = ModelCheckpoint(
        filename="SE{epoch}-{step}-{val_loss:.8f}",
        monitor="val/loss",
        mode="min",
        save_top_k=2,
        save_last =True
        )
    callbacks = [val_ckeckpoint]
    trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"], accelerator='gpu',
                        callbacks=callbacks,
                      logger=False, gradient_clip_val=0.5)
    trainer.fit(model, datamodule=dataMod)                

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)