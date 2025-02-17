import pytorch_lightning as L
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import torch
import h5py
import copy
import numpy as np

from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

#TODO relative import does not work...
def dtype_str_to_type(dtype_str: str):
    if dtype_str.lower() == "float32":
        return torch.float32
    elif dtype_str.lower() == "float64":
        return torch.float64
    else:
        raise ValueError("unkown dtype: " + dtype_str)

class AE_Dataset(Dataset):
    """
    Placeholder for now. 
    We may need this for large datasets or custom transformations/loss functions.
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dtype_default) -> None:
        self.x = x.clone().detach().to(dtype=dtype_default)
        self.y = y.clone().detach().to(dtype=dtype_default)
        self.ylen = y.shape[1] // 2

    def __len__(self) -> int:
        return len(self.x)

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def normalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def unnormalize_y(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __getitem__(self, idx: int) -> tuple:
        x_norm = self.normalize_x(self.x[idx,:])
        y_norm = self.normalize_y(self.y[idx,:])
        return x_norm, y_norm
    

class DataMod_AE(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.prepare_data_per_node = True
        self.train_batch_size = config['batch_size']
        self.val_batch_size = config['batch_size']
        self.test_batch_size = config['batch_size']
        self.data = config['PATH_TRAIN']
        self.dtype = dtype_str_to_type(config['dtype'])
        self.mode = config['mode']

    def setup(self, stage: str):
        """
        Download and transform datasets. 
        """
        if isinstance(self.data, list):
            x = None
            y = None
            for file in self.data:
                with h5py.File(file, "r") as hf:
                    if self.mode == 'gf':
                        xi = hf["Set1/GImp"][:]
                    elif self.mode == 'se':
                        xi = hf["Set1/SImp"][:]
                    else:
                        raise RuntimeError("mode " + self.mode + "not found")
                xi = np.concatenate((xi.real, xi.imag), axis=1)
                yi = copy.deepcopy(xi)
                p = np.random.RandomState(seed=0).permutation(xi.shape[0])
                xi = xi[p,:]
                yi = yi[p,:]
                if x is None:
                    x = xi
                    y = yi
                else:
                    x = np.concatenate((x, xi), axis=0)
                    y = np.concatenate((y, yi), axis=0)
            x = torch.tensor(x, dtype=self.dtype)
            y = torch.tensor(y, dtype=self.dtype)
        else:
            with h5py.File(self.data, "r") as hf:
                if self.mode == 'gf':
                    x = hf["Set1/GImp"][:]
                elif self.mode == 'se':
                    x = hf["Set1/SImp"][:]
                else:
                    raise RuntimeError("mode " + self.mode + "not found")
                #y = hf["Set1/GImp"][:]
            x = np.concatenate((x.real, x.imag), axis=1)
            y = copy.deepcopy(x)
            p = np.random.RandomState(seed=0).permutation(x.shape[0])
            x = x[p,:]
            y = y[p,:]
            x = torch.tensor(x, dtype=self.dtype)
            y = torch.tensor(y, dtype=self.dtype)

        self.train_dataset = AE_Dataset(x, y, self.dtype)
        self.train_set_size = int(len(self.train_dataset) * 0.8)
        self.val_set_size = len(self.train_dataset) - self.train_set_size

        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [self.train_set_size, self.val_set_size])
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False)
    
    def test_dataloader(self):
        raise NotImplementedError("Define standard for data generation from jED.jl and create test data there!")


class AE_H5File_Dataset(Dataset):
    def __init__(self, fp, mode, dtype_default) -> None:
        super().__init__()
        self.fp = fp
        self.fh = None
        self.mode = mode
        with h5py.File(self.fp, "r") as hf:
            self.len = hf["data"][:].shape[0]
        self.dtype_default = dtype_default

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        if self.mode == 'gf':
            x = self._get_GF(idx)
        elif self.mode == 'se':
            x = self._get_SE(idx)
        else:
            raise RuntimeError("mode " + self.mode + "not found")
        return x
    
    def _get_GF(self, idx: int):
        if self.fh is None:
            self.fh = h5py.File(self.fp, "r")
        return self.fh["data"][idx,2:]
    
    def _get_SE(self, idx: int):
        if self.fh is None:
            self.fh = h5py.File(self.fp, "r")
        return self.fh["labels"][idx,:]
    
    def __del__(self):
        if self.fh is not None:
            self.fh.close()


class DataMod_AE_2(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.prepare_data_per_node = True
        self.train_batch_size = config['batch_size']
        self.val_batch_size = config['batch_size']
        self.test_batch_size = config['batch_size']
        self.data = config['PATH_TRAIN']
        self.dtype = dtype_str_to_type(config['dtype'])
        self.mode = config['mode']
        self.generator1 = torch.Generator().manual_seed(0)

    def setup(self, stage: str):
        """
        Download and transform datasets. 
        """
        if isinstance(self.data, list):
            train_ds_list = []
            val_ds_list = []
            for file in self.data:
                ds = AE_H5File_Dataset(file, self.mode, self.dtype)
                self.train_dataset = ds
                self.train_set_size = int(len(ds) * 0.8)
                self.val_set_size = len(ds) - self.train_set_size
                train_ds, val_ds = random_split(self.train_dataset, [self.train_set_size, self.val_set_size], generator=self.generator1)
                train_ds_list.append(train_ds)
                val_ds_list.append(val_ds)
            self.train_dataset = ConcatDataset(train_ds_list)
            self.val_dataset = ConcatDataset(val_ds_list)
        else:
            ds = AE_H5File_Dataset(file, self.dtype)
            self.train_dataset = ds
            self.train_set_size = int(len(ds) * 0.8)
            self.val_set_size = len(ds) - self.train_set_size
            self.train_dataset, self.val_dataset = random_split(ds, [self.train_set_size, self.val_set_size], generator=self.generator1)
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.train_batch_size, num_workers=8, pin_memory=True, persistent_workers=True, shuffle=False)
    
    def test_dataloader(self):
        raise NotImplementedError("Define standard for data generation from jED.jl and create test data there!")