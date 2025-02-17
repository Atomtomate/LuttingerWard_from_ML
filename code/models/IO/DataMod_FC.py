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
    def __init__(self, x: torch.Tensor, y: torch.Tensor, dtype_default) -> None:
        super().__init__()
        self.x = x
        self.y = y
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
        #x_norm = self.normalize_x(self.x[idx,:])
        #y_norm = self.normalize_y(self.y[idx,:])
        return self.x[idx,:], self.y[idx,:]

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
    
    
class FC_H5File_Dataset(Dataset):
    def __init__(self, fp, dtype_default) -> None:
        super().__init__()
        self.fp = fp
        self.fh = None
        with h5py.File(self.fp, "r") as hf:
            self.len = hf["data"][:].shape[0]
        self.dtype_default = dtype_default

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> tuple:
        if self.fh is None:
            self.fh = h5py.File(self.fp, "r")
        return self.fh["data"][idx,:], self.fh["labels"][idx,:]
    
    def __del__(self):
        if self.fh is not None:
            self.fh.close()
    
class DataMod_FC(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()

        self.prepare_data_per_node = True
        self.train_batch_size = config['batch_size']
        self.val_batch_size = config['val_batch_size'] if ('val_batch_size' in config) else config['batch_size']
        self.test_batch_size = config['test_batch_size'] if ('test_batch_size' in config) else config['batch_size']
        self.data = config['PATH_TRAIN']
        self.dtype = dtype_str_to_type(config['dtype'])
        self.train_val_split = config['train_val_split']
        self.preprocessed_data = config['preproc_data'] if ('preproc_data' in config) else False
        self.mmap_x = config['mmap_x'] if ('mmap_x' in config) else False
        self.mmap_y = config['mmap_y'] if ('mmap_y' in config) else False
        self.mmap_x_size = config['mmap_x_size'] if ('mmap_x_size' in config) else 0
        self.mmap_y_size = config['mmap_y_size'] if ('mmap_y_size' in config) else 0
        self.num_workers = config['num_workers'] if ('num_workers' in config) else 8
        self.persistent_workers = config['persistent_workers'] if ('persistent_workers' in config) else True
        self.generator1 = torch.Generator().manual_seed(0)

    def setup(self, stage: str):
        """
        Download and transform datasets. 
        """
        if isinstance(self.data, list):
            x = None
            y = None
            for file in self.data:
                x = None
                y = None
                if self.preprocessed_data:
                    with h5py.File(file, "r") as hf:
                        xi = hf["data"][:]
                        yi = hf["labels"][:]
                        if x is None:
                            x = xi
                            y = yi
                        else:
                            x = np.concatenate((x, xi), axis=0)
                            y = np.concatenate((y, yi), axis=0)
                    x = torch.tensor(x, dtype=self.dtype)
                    y = torch.tensor(y, dtype=self.dtype)
                else:
                    with h5py.File(file, "r") as hf:
                        xi = hf["Set1/GImp"][:]
                        yi = hf["Set1/SImp"][:]
                        ndens = hf["Set1/dens"][:]
                        beta = hf['Set1/Parameters'][:][:,-1]
                    xi = np.concatenate((xi.real, xi.imag), axis=1)
                    yi = np.concatenate((yi.real, yi.imag), axis=1)
                    xi = np.c_[ndens, beta, xi]
                    if x is None:
                        x = xi
                        y = yi
                    else:
                        x = np.concatenate((x, xi), axis=0)
                        y = np.concatenate((y, yi), axis=0)
            x = torch.tensor(x, dtype=self.dtype)
            y = torch.tensor(y, dtype=self.dtype)

            ds = FC_Dataset(x, y, self.dtype)
            len_t = len(ds)
            self.train_set_size = int(len_t * self.train_val_split)
            self.val_set_size = len_t - self.train_set_size
            self.train_dataset, self.val_dataset = random_split(ds, [self.train_set_size, self.val_set_size], generator=self.generator1)
        else:
            print("in else")
            if self.preprocessed_data:
                print("in preproc")

                if isinstance(self.data, list):
                    print("in list")
                    train_ds_list = []
                    val_ds_list = []
                    for file in self.data:
                        ds = FC_H5File_Dataset(file, self.mode, self.dtype)
                        self.train_dataset = ds
                        self.train_set_size = int(len(ds) * 0.8)
                        self.val_set_size = len(ds) - self.train_set_size
                        train_ds, val_ds = random_split(self.train_dataset, [self.train_set_size, self.val_set_size], generator=self.generator1)
                        train_ds_list.append(train_ds)
                        val_ds_list.append(val_ds)
                    self.train_dataset = ConcatDataset(train_ds_list)
                    self.val_dataset = ConcatDataset(val_ds_list)
                    print("len ", len(self.train_dataset))
                else:
                    print("in list else")
                    ds = FC_H5File_Dataset(file, self.dtype)
                    if stage != 'test':
                        self.train_set_size = int(len(ds) * 0.8)
                        self.val_set_size = len(ds) - self.train_set_size
                        self.train_dataset, self.val_dataset = random_split(ds, [self.train_set_size, self.val_set_size], generator=self.generator1)
                    else:
                        self.test_dataset = ds
            else:
                print("in preproc else")
                with h5py.File(self.data, "r") as hf:
                    x = hf["Set1/GImp"][:]
                    y = hf["Set1/SImp"][:]
                    ndens = hf["Set1/dens"][:]
                    beta = hf['Set1/Parameters'][:][:,-1]
                x = np.concatenate((x.real, x.imag), axis=1, dtype=self.dtype)
                y = np.concatenate((y.real, y.imag), axis=1, dtype=self.dtype)
                #p = np.random.RandomState(seed=0).permutation(x.shape[0])
                #x = x[p,:]
                #y = y[p,:]
                x = np.c_[ndens, beta, x]
                x = torch.tensor(x, dtype=self.dtype)
                y = torch.tensor(y, dtype=self.dtype)

                ds = FC_Dataset(x, y, self.dtype)
                len_t = len(ds)
                if stage != 'test':
                    self.train_set_size = int(len_t * self.train_val_split)
                    self.val_set_size = len_t - self.train_set_size
                    self.train_dataset, self.val_dataset = random_split(ds, [self.train_set_size, self.val_set_size], generator=self.generator1)
                else:
                    self.test_dataset = ds
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persistent_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=self.persistent_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers,  pin_memory=True, shuffle=False)
