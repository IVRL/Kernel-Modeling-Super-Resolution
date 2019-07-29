import h5py
import torch.utils.data as data
import torch
from torchvision import transforms
from functools import partial
import numpy as np
import random
from imageio import imread
import glob
from scipy import signal

class Hdf5Dataset(data.Dataset):
    def __init__(self, base='train', scale=2):
        super(Hdf5Dataset, self).__init__()
        
        base = base + '/'
        self.hr_dataset = h5py.File(base + 'hr.h5')['/data']
        self.kernel = h5py.File(base + 'kernel.h5')['/data']

    def __getitem__(self, index):
        y = self.hr_dataset[index]
        
        kernel_index = min(random.randint(0,1999), len(self.kernel)-1)
        kernel = self.kernel[kernel_index]
        x = y.copy()
        x[0,:,:] = signal.convolve2d(y[0,:,:], kernel[0,:,:], 'same')
        x[1,:,:] = signal.convolve2d(y[1,:,:], kernel[0,:,:], 'same')
        x[2,:,:] = signal.convolve2d(y[2,:,:], kernel[0,:,:], 'same')
        return x,y

    def __len__(self):
        return self.hr_dataset.shape[0]
    
