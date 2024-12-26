import pytorch_lightning as L
import torch
import h5py
import copy
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split, get_worker_info


def dtype_str_to_type(dtype_str: str):
    if dtype_str.lower() == "float32":
        return torch.float32
    elif dtype_str.lower() == "float64":
        return torch.float64
    else:
        raise ValueError("unkown dtype: " + dtype_str)

class AE_Dataset(Dataset):
    """
    AE Dataset
    """
    def __init__(self, data_path, mode, dtype_default) -> None:
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.dtype = dtype_default
        with h5py.File(self.data_path, "r") as hf:
            if self.mode == 'gf':
                x = hf["GImp"][:]
            elif self.mode == 'se':
                x = hf["SImp"][:]
            else:
                raise RuntimeError("mode " + self.mode + "not found")
        x = np.concatenate((x.real, x.imag), axis=1)
        self.x = torch.tensor(x, dtype=self.dtype)
        self.len = x.shape[0]
    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        x_norm = self.x[idx,:]
        return x_norm
    
class AE_DatasetFile(Dataset):
    """
    AE Dataset
    """
    def __init__(self, data_path, mode, dtype_default, transform=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.mode = mode
        self.dtype = dtype_default
        self.fh = None
        if self.mode == 'gf':
            self.key = "GImp"
        elif self.mode == 'se':
            self.key = "SImp"
        with h5py.File(self.data_path, 'r') as fh:
            self.len = fh["GImp"][:].shape[0]

    def __del__(self):
        if not (self.fh is None):
            self.fh.close()
        
    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        if self.fh is None:
            self.fh = h5py.File(self.data_path, 'r')
        data = self.fh[self.key][idx]
        return data
    
class DataMod_AE(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.prepare_data_per_node = True
        self.train_batch_size = config['batch_size']
        self.val_batch_size = config['batch_size']
        self.test_batch_size = config['batch_size']
        self.data_path = config['PATH_TRAIN']
        self.num_workers = config['num_workers'] if ('num_workers' in config) else 8
        self.dtype = dtype_str_to_type(config['dtype'])
        self.mode = config['mode']

    def setup(self, stage: str):
        """
        Download and transform datasets. 
        """
        if isinstance(self.data_path, list):
            print("ERROR CONCAT TDATASET NOT IMPLEMENTED")
        else:
            self.train_dataset = AE_DatasetFile(self.data_path, self.mode, self.dtype)
        self.train_set_size = int(len(self.train_dataset) * 0.8)
        self.val_set_size = len(self.train_dataset) - self.train_set_size

        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [self.train_set_size, self.val_set_size])
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True, shuffle=False)
    
    def test_dataloader(self):
        raise NotImplementedError("Define standard for data generation from jED.jl and create test data there!")
