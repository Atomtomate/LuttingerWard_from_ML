import torch
import torch.nn as nn
import pytorch_lightning as L
import numpy as np
from utils.LossFunctions import *
from utils.misc import *
import matplotlib.pyplot as plt

#TODO: abstract encoding/docing/G_to_Sigma parts away. Define this model to consist of 3 models.

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
    hparams['latent_layers'] = config['latent_layers'] if 'FC_layers' in config.keys()  else config['FC_layers']
    hparams['with_batchnorm'] = config['with_batchnorm']
    hparams['optimizer'] = config['optimizer']
    hparams['loss'] = config['loss']
    hparams['weight_decay'] = config['weight_decay']
    hparams['out_dim'] = config['out_dim']
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


class FC_01(L.LightningModule):
    """Feed Forward network."""
    def __init__(self, config: dict) -> None:
        super().__init__()

        hparams = FC_config_to_hparams(config)
        for key in hparams.keys():
            self.hparams[key]=hparams[key]

        self.dropout_in = nn.Dropout(self.hparams['dropout_in']) if self.hparams['dropout_in'] > 0 else nn.Identity()
        self.dropout = nn.Dropout(self.hparams['dropout']) if self.hparams['dropout'] > 0  else nn.Identity()
        self.activation = activation_str_to_layer(self.hparams['activation'])
        self.reconstr_loss_f = loss_str_to_layer(self.hparams['loss'])
        self.lr = self.hparams['lr']

        self.plot_worst_examples = False
        self.worst_losses_ids = np.zeros(2,dtype=int)    # track the worst 3 example indices
        self.worst_losses_data  = [None, None]           # track the worst 3 example losses
        self.test_step_outputs = []

        bl_fc_net = []
        for i in range(self.hparams['latent_layers']):
            
            bl_fc_net.extend(linear_block(self.hparams['in_dim'], self.hparams['out_dim'] if (i == self.hparams['latent_layers'] - 1) else self.hparams['in_dim'],  
                                        self.activation, nn.Identity(), self.dropout, self.hparams['with_batchnorm'],
                                        last_layer = (i == self.hparams['latent_layers'] - 1), first_layer = False
                                        ))
        self.fc_net     = nn.Sequential(*bl_fc_net) if self.hparams['latent_layers'] > 0 else nn.Sequential(nn.Identity())

        for layer in self.fc_net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
                nn.init.zeros_(layer.bias)
        self.save_hyperparameters(self.hparams)
    
    def G_to_SE(self, G):
        SE = self.fc_net(G)
        return SE
    
    def forward(self, x):
        SE = self.G_to_SE(x)
        return SE



    def training_step(self, batch, batch_idx):
        x, SE_in = batch
        SE_hat = self(x)
        SE_reconstr = self.reconstr_loss_f(SE_in, SE_hat)

        loss =  SE_reconstr
        self.log("train/loss", loss, prog_bar=False)
        return loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, SE_in = batch
        SE_hat = self(x)
        SE_reconstr = self.reconstr_loss_f(SE_in, SE_hat)
        loss =  SE_reconstr
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, SE_in = batch
        SE_hat = self(x)
        SE_reconstr = self.reconstr_loss_f(SE_in, SE_hat)
        loss =  SE_reconstr
        self.log("test/loss", loss, on_epoch=True)
        self.test_step_outputs.append(loss) 
        return loss
    
    def on_validation_epoch_start(self):
        # reset worst examples
        self.worst_losses_data = [None, None]
        self.worst_losses    = np.zeros(2)

    def on_validation_epoch_end(self):
        # plot worst 2 
        if False:
            for ii,batch_i in enumerate(self.worst_losses_data):
                if batch_i is not None:
                    G_in, S_hat, S_in = batch_i
                    batch_len = S_in.size(0)
                    fig, axs = plt.subplots(batch_len,3, figsize=(24,12))
                    for i in range(batch_len):
                        axs[i,0].plot(G_in[i,:].cpu(), linewidth=2)
                        axs[i,1].plot(S_in[i,:].cpu(), label="ground truth", linewidth=2)
                        axs[i,1].plot(S_hat[i,:].cpu(), label="prediction", linewidth=2)
                        axs[i,0].set_title(f"Batch {ii}")
                        axs[i,1].legend()
                        axs[i,2].plot(np.abs(S_hat[i,:].cpu() - S_in[i,:].cpu()), label="Log Diff")
                        axs[i,0].set_xlabel("nu")
                        axs[i,0].set_ylabel("G_in")
                        axs[i,1].set_xlabel("nu")
                        axs[i,1].set_ylabel("Sigma_in")
                        axs[i,2].set_xlabel("nu")
                        axs[i,2].set_ylabel("Delta Sigma")
                        axs[i,2].set_yscale('log')
                    self.logger.experiment[f"val/worst_examples_{ii}"].append(fig)

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