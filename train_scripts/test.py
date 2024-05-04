import sys
from os.path import dirname, abspath, join

#TODO I hate the python import system. someone else fix this please.
sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE_FC_transfer import AE_FC_02
sys.path.append(join(dirname(__file__),'../code/models/IO'))
from DataMod_FC import *

import pytorch_lightning as L
import torch
from pytorch_lightning.loggers import TensorBoardLogger, NeptuneLogger
import json

torch.set_float32_matmul_precision("medium")

sys.path.append(join(dirname(__file__),'../code/models'))
from model_AE import AutoEncoder_01

