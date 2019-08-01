# Kernel Modeling Super-Resolution on Real Low-Resolution Images
#### [Project Page](https://ivrlwww.epfl.ch/ruofan/project_KMSR/KMSR.html) | [Paper]() | [Supplementary Material]() | [Psychovisual Experiment](https://ivrlwww.epfl.ch/ruofan/exp/index.html)
by [Ruofan Zhou](https://ivrl.epfl.ch/people/Zhou), [Sabine Süsstrunk](https://ivrl.epfl.ch/people/susstrunk)

## Dependencies
- Pytorch >= 0.4.0
- OpenCV
- NVIDIA GPU
- HDF5 (only for training)
- MATLAB （only for training)

## Quick start (Demo)
In `test_code` folder, run the following command: 
```
python demo.py
```
## Training the network yourself
#### Step 1: prepare the dataset
Download [DEPD dataset](people.ee.ethz.ch/~ihnatova/index.html), prepare the patches and run `training_code/kernel_estimation/getkernels.m` in MATLAB.

#### Step 2: train a GAN on kernels
run `training_code/kernel_generator/train.py`.

#### Step 3: generate
run `training_code/kernel_generator/generate.py`.

#### Step 4: train the super-resolution network
run `training_code/super-resolution/main.py`.

## Citations


