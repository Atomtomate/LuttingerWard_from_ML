
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
import json

torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float64)

def main(args):
    config_path = join(dirname(__file__),'../configs/confmod_AE_FC_transfer_with_dens_test.json')
    config = json.load(open(config_path))
    torch.manual_seed(config['seed'])
    model = AE_FC_02(config) 
    dataMod = DataMod_FC(config)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    val_ckeckpoint = ModelCheckpoint( # saved in `trainer.default_root_dir`/`logger.version`/`checkpoint_callback.dirpath`
            filename="{epoch}-{step}-{val_loss:.8f}",
            monitor="val/loss",
            mode="min",
            save_top_k=2,
            save_last =True
            )
    early_stopping = EarlyStopping(monitor="val/loss",patience=20, stopping_threshold=5e-10, min_delta=1e-11)
    swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220)
    accumulator = GradientAccumulationScheduler(scheduling={0: 512, 12: 128, 24: 64, 32: 32, 44: 16, 56: 8, 68: 4, 80: 1})
    callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
    trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"],
                    callbacks=callbacks, gradient_clip_val=0.5) 
    
    trainer.fit(model, datamodule=dataMod)


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)