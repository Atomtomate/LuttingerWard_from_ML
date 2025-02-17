
import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE import AutoEncoder_02
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
    config = json.load(open(join(dirname(__file__),'../configs/confmod_AE_SE2.json')))
    torch.manual_seed(config['seed'])
    model = AutoEncoder_02(config) 
    dataMod = DataMod_AE_2(config)

    trainer = L.Trainer(enable_checkpointing=False, max_epochs=config["epochs"],accelerator="cpu",fast_dev_run=True,
                      logger=False, gradient_clip_val=0.5) #precision="16-mixed", 

    trainer.fit(model, datamodule=dataMod)                

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)