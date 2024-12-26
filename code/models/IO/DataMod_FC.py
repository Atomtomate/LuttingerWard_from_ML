import pytorch_lightning as L
import torch
import h5py
import copy
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

#TODO relative import does not work...
def dtype_str_to_type(dtype_str: str):
    if dtype_str.lower() == "float32":
        return torch.float32
    elif dtype_str.lower() == "float64":
        return torch.float64
    else:
        raise ValueError("unkown dtype: " + dtype_str)

class FC_Dataset(Dataset):
    """
    Placeholder for now. 
    We may need this for large datasets or custom transformations/loss functions.
    """
    def __init__(self, data_path, dtype_default) -> None:
        super().__init__()
        self.data_path = data_path
        self.dtype = dtype_default
        with h5py.File(self.data_path, "r") as hf:
            x = hf["Set1/GImp"][:]
            y = hf["Set1/SImp"][:]
            ndens = hf["Set1/dens"][:]
        x = np.concatenate((x.real, x.imag), axis=1)
        y = np.concatenate((y.real, y.imag), axis=1)
        x = np.c_[ndens, x]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        self.len = x.shape[0]
        

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        x_norm = torch.tensor(self.x[idx,:],dtype=self.dtype)
        y_norm = torch.tensor(self.y[idx,:],dtype=self.dtype)
        return x_norm, y_norm
    
class FC_DatasetFile(Dataset):
    """
    FC Dataset
    """
    def __init__(self, data_path, dtype_default, transform=None) -> None:
        super().__init__()
        self.data_path = data_path
        self.dtype = dtype_default
        self.fh = None
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
        data = self.fh["GImp"][idx]
        labels = self.fh["SImp"][idx]
        dens = self.fh["dens"][idx]
        return data, labels, dens

class FC_File_Dataset(Dataset):
    def __init__(self, fp_x, fp_y, shape_x, shape_y, dtype_default) -> None:
        super().__init__()
        self.len = shape_x[0]
        self.dtype_default = dtype_default
        self.x = torch.from_file(fp_x, shared=True, size=shape_x[0]*shape_x[1], dtype=torch.float64).reshape(shape_x)
        self.y = torch.from_file(fp_y, shared=True, size=shape_y[0]*shape_y[1], dtype=torch.float64).reshape(shape_y)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        return self.x[idx,:], self.y[idx,:]
    

class DataMod_FC(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.prepare_data_per_node = True
        self.train_batch_size = config['batch_size']
        self.val_batch_size = config['val_batch_size'] if ('val_batch_size' in config) else config['batch_size']
        self.test_batch_size = config['test_batch_size'] if ('test_batch_size' in config) else config['batch_size']
        self.data_path = config['PATH_TRAIN']
        self.dtype = dtype_str_to_type(config['dtype'])
        self.train_val_split = config['train_val_split']
        self.preprocessed_data = config['preproc_data'] if ('preproc_data' in config) else False
        self.mmap_x = config['mmap_x'] if ('mmap_x' in config) else False
        self.mmap_y = config['mmap_y'] if ('mmap_y' in config) else False
        self.mmap_x_size = config['mmap_x_size'] if ('mmap_x_size' in config) else 0
        self.mmap_y_size = config['mmap_y_size'] if ('mmap_y_size' in config) else 0
        self.num_workers = config['num_workers'] if ('num_workers' in config) else 8
        self.persistent_workers = config['persistent_workers'] if ('persistent_workers' in config) else True

    def setup(self, stage: str):
        """
        Download and transform datasets. 
        """
        if isinstance(self.data_path, list):
            print("ERROR CONCAT TDATASET NOT IMPLEMENTED")
        else:
            if not self.mmap_x:
                ds = FC_DatasetFile(self.data_path, self.dtype)
                len_t = len(ds)
                if stage != 'test':
                    self.train_set_size = int(len_t * self.train_val_split)
                    self.val_set_size = len_t - self.train_set_size
                    self.train_dataset, self.val_dataset = random_split(ds, [self.train_set_size, self.val_set_size])
                else:
                    self.test_dataset = ds
            else:
                print("ERROR MMAP NOT WORKING")
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persistent_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persistent_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers, shuffle=False)
