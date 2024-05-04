
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

torch.set_float32_matmul_precision("highest")
torch.set_default_dtype(torch.float64)

def main(args):
    pr = neptune.init_project("LW-AEpFC")
    models_table = pr.fetch_models_table().to_pandas()
    i = 0
    config_path = join(dirname(__file__),'../configs/confmod_AE_FC_transfer.json')
    config = json.load(open(config_path))
    model_key = "AEFC"
    if not any(models_table["sys/id"].str.contains("LWAEP-"+model_key)):
        modelN = neptune.init_model(key=model_key,name=config['MODEL_NAME'],project="stobbe.julian/LW-AEpFC")
    for FC_layers in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]:
        if i > 3:
            config['FC_layers'] = FC_layers
            model_version = neptune.init_model_version(model=f"LWAEP-AEFC",name=f"L{14}FC{FC_layers}",project="stobbe.julian/LW-AEpFC")
            torch.manual_seed(config['seed'])
            model = AE_FC_02(config) 
            dataMod = DataMod_FC(config)
            model_version["model/signature"].upload(config_path)
            model_script = model.to_torchscript()
            torch.jit.save(model_script, "tmp_model.pt")
            model_version["model/definition"].upload("tmp_model.pt")

            lr_monitor = LearningRateMonitor(logging_interval='step')
            neptune_logger = NeptuneLogger(    
                                project="stobbe.julian/LW-AEpFC",
                                name=config['MODEL_NAME'],
                                description="Pretrained Autoencoder with fully connected feed forward for Luttinger Ward functional. NO additonal input like electron density.",
                                tags=["AE", "FC"],
                                capture_hardware_metrics=False,
                                capture_stdout=False,
                                )
            
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
                            callbacks=callbacks, logger=neptune_logger, gradient_clip_val=0.5) #precision="16-mixed", 
            
            trainer.fit(model, datamodule=dataMod)
            model_version["run/id"] = neptune_logger._run_instance["sys/id"].fetch()
            neptune_logger.log_model_summary(model=model, max_depth=-1)
            neptune_logger._run_instance.stop()
        i += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)