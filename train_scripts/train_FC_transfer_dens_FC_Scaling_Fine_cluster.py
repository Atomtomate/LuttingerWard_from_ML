
import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE_FC_transfer import AE_FC_02
sys.path.append(join(dirname(__file__),'../code/models/IO'))
from DataMod_FC import *

import pytorch_lightning as L
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, StochasticWeightAveraging, GradientAccumulationScheduler,  DeviceStatsMonitor
from pytorch_lightning.profilers import PyTorchProfiler
import torch
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
from argparse import ArgumentParser
import neptune
import json

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float64)

def main(args):
    config_path = join(dirname(__file__),'../configs/confmod_AE_FC_transfer_with_dens_tmp.json')
    config = json.load(open(config_path))
    for FC_layers in [6]:
        for FC_dim in [20]:
                config['FC_layers'] = FC_layers
                config['FC_dim'] = FC_dim
                torch.manual_seed(config['seed'])
                model = AE_FC_02(config,dbg_print = True) 
                dataMod = DataMod_FC(config)

                trainer = L.Trainer(enable_checkpointing=False, max_epochs=config["epochs"], accelerator="cpu",
                                logger=False, gradient_clip_val=0.5) #precision="16-mixed", 
                torch.compile(model, fullgraph=True, mode="max-autotune", backend='cudagraphs')
                trainer.fit(model, datamodule=dataMod)

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)