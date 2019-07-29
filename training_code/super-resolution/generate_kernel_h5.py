import os
import glob
import cv2
import h5py
import numpy as np
from hdfstore import HDF5Store
from scipy.io import loadmat
import random

kernel_list = []

# get kernel lists from estimated kernel
num_kernels = 1000
folder = '../kernel_estimation/x2results/'
subfolders = ['blackberry_x2', 'sony_x2']
estimated_kernel = []

for subfolder in subfolders:
    kernels = glob.glob(folder+subfolder+'/*.mat')
    estimated_kernel = estimated_kernel + kernels
random.shuffle(estimated_kernel)

kernel_list = kernel_list + estimated_kernel[0:min(num_kernels,len(estimated_kernel))]

# get kernel lists from generated kernel
num_kernels = 1000
folder = '../kernel_generator/generated_kernel/'
generated_kernel = glob.glob(folder + '*.mat')
random.shuffle(generated_kernel)

kernel_list = kernel_list + generated_kernel[0:min(num_kernels,len(generated_kernel))]

h5_file = 'train/kernel.h5'
if os.path.isfile(h5_file):
    os.remove(h5_file)
hr_patch_shape = (25, 25, 1)
hdf5_hr = HDF5Store(datapath=h5_file, dataset='data', shape=hr_patch_shape)

for kernelfile in kernel_list:
    mat = loadmat(kernelfile)
    x = np.array([mat['kernel']])
    hdf5_hr.append(x.reshape((25,25,1)))
