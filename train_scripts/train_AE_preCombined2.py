import sys
from os.path import dirname, abspath, join

model_path = join(dirname(__file__),'../code/models')
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE import AutoEncoder_02
sys.path.append(join(dirname(__file__),'../code/models/IO'))
from DataMod_AE import *

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

def load_checkpoint(run: neptune.Run, epoch: int):
    checkpoint_name = f"epoch_{epoch}"
    ext = run["checkpoints"][checkpoint_name].fetch_extension()
    run["checkpoints"][checkpoint_name].download()  # Download the checkpoint
    run.wait()
    checkpoint = torch_load(f"{checkpoint_name}.{ext}")  # Load the checkpoint
    return checkpoint


def main(args):
    pr_name = "LW-AEpFC"
    pr = neptune.init_project(pr_name)
    models_table = pr.fetch_models_table().to_pandas()
    i = 0
    for mode in ['se']:
        if mode == 'gf':
            config_path = join(dirname(__file__),'../configs/confmod_AE_GF_2.json')
        else:
            config_path = join(dirname(__file__),'../configs/confmod_AE_SE_2.json')
        config = json.load(open(config_path))
        model_key = "AE2"+mode.upper()
        if not any(models_table["sys/id"].str.contains("LWAEP-"+model_key)):
            modelN = neptune.init_model(key=model_key,name=config['MODEL_NAME'],project="stobbe.julian/"+pr_name)
        for AE_layers in [2]:
            config['n_layers'] = AE_layers
            for laten_dims in [14,13,12,15,16]:
                if AE_layers == 2 and laten_dims == 14:
                    config['latent_dim'] = laten_dims
                    model_version = neptune.init_model_version(model=f"LWAEP-AE2{mode.upper()}",name=f"L{AE_layers}LD{laten_dims}",project="stobbe.julian/"+pr_name)

                    torch.manual_seed(config['seed'])
                    model = AutoEncoder_02.load_from_checkpoint(checkpoint_path="G:/Codes/LuttingerWard_from_ML/.neptune/auto_encoder/LWAEP-419/checkpoints/last.ckpt",config=config)
                    dataMod = DataMod_AE_2(config)
                    model_version["model/signature"].upload(config_path)
                    model_script = model.to_torchscript()
                    torch.jit.save(model_script, "tmp_model.pt")
                    model_version["model/definition"].upload("tmp_model.pt")

                    lr_monitor = LearningRateMonitor(logging_interval='step')
                    neptune_logger = NeptuneLogger(    
                                        project="stobbe.julian/"+pr_name,
                                        name=config['MODEL_NAME'],
                                        description="Simple Autoencoder.",
                                        tags=["AE2"],
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
                    early_stopping = EarlyStopping(monitor="val/loss",patience=30, stopping_threshold=1e-12, min_delta=2e-13)
                    swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220)
                    #accumulator = GradientAccumulationScheduler(scheduling={0: 512, 12: 128, 24: 64, 32: 32, 44: 16, 56: 8, 68: 4, 80: 1})
                    accumulator = GradientAccumulationScheduler(scheduling={0: 2048, 40: 512, 48: 1})
                    callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
                    trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"],
                                    callbacks=callbacks, logger=neptune_logger, gradient_clip_val=0.5) #precision="16-mixed", 
                    
                    trainer.fit(model, datamodule=dataMod, ckpt_path="G:/Codes/LuttingerWard_from_ML/.neptune/auto_encoder/LWAEP-419/checkpoints/last.ckpt")
                    model_version["run/id"] = neptune_logger._run_instance["sys/id"].fetch()
                    neptune_logger.log_model_summary(model=model, max_depth=-1)
                    neptune_logger._run_instance.stop()
                else:
                    config['latent_dim'] = laten_dims
                    model_version = neptune.init_model_version(model=f"LWAEP-AE2{mode.upper()}",name=f"L{AE_layers}LD{laten_dims}",project="stobbe.julian/"+pr_name)

                    torch.manual_seed(config['seed'])
                    model = AutoEncoder_02(config) 
                    dataMod = DataMod_AE_2(config)
                    model_version["model/signature"].upload(config_path)
                    model_script = model.to_torchscript()
                    torch.jit.save(model_script, "tmp_model.pt")
                    model_version["model/definition"].upload("tmp_model.pt")

                    lr_monitor = LearningRateMonitor(logging_interval='step')
                    neptune_logger = NeptuneLogger(    
                                        project="stobbe.julian/"+pr_name,
                                        name=config['MODEL_NAME'],
                                        description="Simple Autoencoder.",
                                        tags=["AE2"],
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
                    early_stopping = EarlyStopping(monitor="val/loss",patience=30, stopping_threshold=1e-12, min_delta=2e-13)
                    swa = StochasticWeightAveraging(swa_lrs=1e-8,annealing_epochs=40, swa_epoch_start=220)
                    accumulator = GradientAccumulationScheduler(scheduling={0: 2048, 4: 512, 8: 1})
                    callbacks = [lr_monitor, early_stopping, val_ckeckpoint, swa, accumulator]
                    trainer = L.Trainer(enable_checkpointing=True, max_epochs=config["epochs"], accelerator="cpu",
                                    callbacks=callbacks, logger=neptune_logger, gradient_clip_val=0.5) #precision="16-mixed", 
                    
                    trainer.fit(model, datamodule=dataMod)
                    model_version["run/id"] = neptune_logger._run_instance["sys/id"].fetch()
                    neptune_logger.log_model_summary(model=model, max_depth=-1)
                    neptune_logger._run_instance.stop()


if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)