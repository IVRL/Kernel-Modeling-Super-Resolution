from model import *
import glob 
import cv2
import numpy as np
import torch
import os

model_path = './checkpoint/model_epoch_50.pth'
lr_path = '/scratch/rzhou/Kernel_test/dataset/DIV2K_valid_LR_bicubic/X4/'
output_path = './results'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ResBlockNet().to(device, dtype=torch.float)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint["model"].state_dict())

lr_imgs = sorted(glob.glob(lr_path + '*.png'))
if not(os.path.isdir(output_path)):
    os.mkdir(output_path)

for i in range(len(lr_imgs)):
    lr_img = cv2.imread(lr_imgs[i])
    lr_img = cv2.resize(lr_img, (0,0), fx=2, fy=2)
    
    lr_img = np.swapaxes(lr_img, 0, 2)/255.0
    lr_img = torch.tensor([lr_img], device=device).float()
    sr_img = model(lr_img)
    sr_img = sr_img.cpu().data.numpy()
    sr_img = np.clip(sr_img, 0, 1)
    sr_img = sr_img[0] * 255
    srbgr = np.swapaxes(sr_img, 0, 2)
    cv2.imwrite(output_path + os.path.basename(lr_imgs[i]), srbgr)
