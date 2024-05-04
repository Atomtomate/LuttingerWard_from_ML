import torch
import torch.nn as nn
import pytorch_lightning as L
import numpy as np
from utils.LossFunctions import *
from utils.misc import *
import matplotlib.pyplot as plt
from model_AE import AutoEncoder_01
import json

def FC_config_to_hparams(config: dict) -> dict:
    """
    This extracts all model relevant parameters from the config 
    dict (which also contains runtime related information).
    """
    hparams = {}
    hparams['batch_size'] = config['batch_size']
    hparams['lr'] = config['learning_rate']
    hparams['dropout_in'] = config['dropout_in']
    hparams['dropout'] = config['dropout']
    hparams['activation'] = config['activation']
    hparams['in_dim'] = config['in_dim']
    hparams['latent_dim'] = config['latent_dim']
    hparams['AE_layers'] = config['AE_layers']
    hparams['latent_layers'] = config['FC_layers']
    hparams['with_batchnorm'] = config['with_batchnorm']
    hparams['optimizer'] = config['optimizer']
    hparams['loss'] = config['loss']
    hparams['weight_decay'] = config['weight_decay']
    hparams['out_dim'] = config['out_dim']
    hparams['GF_AE_path']  = config['GF_AE_path']
    hparams['SE_AE_path']  = config['SE_AE_path']
    return hparams


def linear_block(in_dim, out_dim, activation, 
                 dropout_in, dropout, with_batchnorm, 
                 first_layer = False, last_layer=False):
    res = [
        dropout_in if first_layer else dropout,
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim) if (not last_layer) and with_batchnorm else nn.Identity(),
        activation if (not last_layer) else nn.Identity() 
    ]
    return res



class AE_FC_02(L.LightningModule):
    """Fully conencted network with pretrained autoencoders."""
    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        hparams = FC_config_to_hparams(config)
        for key in hparams.keys():
            self.hparams[key]=hparams[key]

        self.dropout_in = nn.Dropout(self.hparams['dropout_in']) if self.hparams['dropout_in'] > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.hparams['dropout']) if self.hparams['dropout'] > 0  else nn.Identity()
        self.activation = activation_str_to_layer(self.hparams['activation'])
        self.reconstr_loss_f = loss_str_to_layer(self.hparams['loss'])
        self.lr = self.hparams['lr']
        self.hparams['FC_dim'] =  self.hparams['latent_dim'] + 1 # use density in latent dim, 

        print("G:")
        tmp_cfg = json.load(open("G:/Codes/LuttingerWard_from_ML/configs/confmod_AE_GE_tmp.json"))
        self.GF_encoder = AutoEncoder_01.load_from_checkpoint(self.hparams['GF_AE_path'], config=tmp_cfg)
        self.GF_encoder.eval()
        self.GF_encoder.freeze()

        print("SE:")
        tmp_cfg = json.load(open("G:/Codes/LuttingerWard_from_ML/configs/confmod_AE_SE_tmp.json"))
        self.SE_encoder = AutoEncoder_01.load_from_checkpoint(self.hparams['SE_AE_path'], config=tmp_cfg)
        self.SE_encoder.eval()
        self.SE_encoder.freeze()
        
        bl_fc_net = []
        for i in range(self.hparams['latent_layers']):
            #TODO: half of the first/last layer logic is here, half in linear_block...
            bl_fc_net.extend(linear_block(self.hparams['latent_dim'] if (i == 0)                                 else self.hparams['FC_dim'], 
                                          self.hparams['latent_dim'] if (i == self.hparams['latent_layers'] - 1) else self.hparams['FC_dim'],  
                                        self.activation, nn.Identity(), self.dropout, self.hparams['with_batchnorm']
                                        ))
        self.fc_net     = nn.Sequential(*bl_fc_net) if self.hparams['latent_layers'] > 0 else nn.Sequential(nn.Identity())

        for layer in self.fc_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.save_hyperparameters(self.hparams)
    
    def GF_to_SE(self, GF):
        SE = self.fc_net(GF)
        return SE
    
    def forward(self, GF, ndens):
        G_latent = self.GF_encoder.encoder(GF)
        SE_latent = self.GF_to_SE(torch.cat((G_latent,ndens), dim=1))
        SE = self.SE_encoder.decoder(SE_latent)
        return SE


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, SE_in = batch
        ndens, beta, GF_in = torch.split(x, [1,1,x.size(1)-2], dim=1) 
        SE_hat = self(GF_in, ndens)
        loss = self.reconstr_loss_f(SE_in, SE_hat)
        self.log("train/loss", loss, prog_bar=False)
        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, SE_in = batch
        ndens, beta, GF_in = torch.split(x, [1,1,x.size(1)-2], dim=1) 
        SE_hat = self(GF_in, ndens)
        loss = self.reconstr_loss_f(SE_in, SE_hat)
        self.log("val/loss", loss, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        if self.hparams["optimizer"] == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr,
                                    momentum=self.hparams["SGD_momentum"],
                                    weight_decay=self.hparams["SGD_weight_decay"],
                                    dampening=self.hparams["SGD_dampening"],
                                    nesterov=self.hparams["SGD_nesterov"])
        elif self.hparams["optimizer"] == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        elif self.hparams["optimizer"] == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError("unkown optimzer: " + self.hparams["optimzer"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=8, threshold=1e-3,
                                                               threshold_mode='rel', verbose=True)
        return {"optimizer": optimizer, 
                "lr_scheduler": scheduler, 
                "monitor": "val/loss"}
    
    def load_model_state(self, PATH):
        checkpoint = torch.load(PATH, map_location='cuda:0')
        self.load_state_dict(checkpoint['state_dict'])