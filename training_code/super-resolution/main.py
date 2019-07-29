import numpy as np
import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import *
from dataset import Hdf5Dataset
from utils import get_model_dir

# Training settings
parser = argparse.ArgumentParser(description="KMSR")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=1000, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.001, help="Learning Rate. Default=0.001")

# Continuing training
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--pretrained", default='', type=str, help='path to pretrained model (default: none)')

def main():
    global opt, model
    opt = parser.parse_args()
    print(opt)

    cudnn.benchmark = True

    print("===> Loading datasets")
    train_set = Hdf5Dataset(base='train', scale=2)
    training_data_loader = DataLoader(dataset=train_set, num_workers=1, batch_size=opt.batchSize, shuffle=True)
    
    print("===> Building model")
    model = ResBlockNet()
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    model = model.cuda()
    criterion = criterion.cuda()
    
    # Loading previous models
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):
            print("=> loading model '{}'".format(opt.pretrained))
            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    print("===> Training")
    for epoch in range(opt.start_epoch, opt.nEpochs + 1):
        train(training_data_loader, optimizer, model, criterion, epoch)
        save_checkpoint(model, epoch)
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // 30))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        data, target = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
        output = model(data)
        
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if iteration % 50 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.item()))
            save_checkpoint(model, epoch)
    save_checkpoint(model, epoch)


def save_checkpoint(model, epoch):
    model_dir = "checkpoint/"
    model_out_path = "%s/model_epoch_%d.pth" % (model_dir, epoch)
    state = {"epoch": epoch ,"model": model}
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
