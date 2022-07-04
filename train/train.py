from __future__ import division

import argparse
import os
import time
import numpy as np
import scipy.io
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from HIRnet import *
from SCNet import *
from dataset import HSdataset
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss


def main():
    load_encoder = False
    cudnn.benchmark = False

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop((256,256)),
        # transforms.RandomHorizontalFlip()
        # transforms.Resize((512,512))
    ])

    train_dataset = HSdataset(img_dir=r'C:\data\CVTL_dataset_fourier\train', transform=transform,load_encoder=None)
    valid_dataset = HSdataset(img_dir=r'C:\data\CVTL_dataset_fourier\valid', transform=transform,load_encoder=None)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=True, pin_memory=True)

    # Model
    model = HIRnet_new(filters=4,layers=[2,2,2],input_channel=301,output_channel=301)
    if load_encoder:
        # enconder_filter = scipy.io.loadmat(
        #     r'C:\Users\A\Desktop\filter_400_700.mat')['filter']
        enconder_filter = np.load(r'C:\Users\A\Desktop\Project\Project\created_filter.npy')

        # enconder_filter = enconder_filter.transpose()
        enconder_filter = torch.from_numpy(enconder_filter).unsqueeze(-1).unsqueeze(-1)
        # print(enconder_filter)
        with torch.no_grad():
            model.hard_encoder.conv1.weight.copy_(enconder_filter)
            # model.hard_encoder.requires_grad_(requires_grad=False)

    # Parameters, Loss and Optimizer
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print('dataparallel')
    if torch.cuda.is_available():
        model.cuda()

    start_epoch = 0
    end_epoch = 1000
    init_lr = 0.02
    iteration = 0
    record_test_loss = 0.1
    criterion = torch.nn.MSELoss()
    # criterion = torch.nn.KLDivLoss()
    # criterion = rrmse_loss
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5)
    model_path = 'HIRnet_new_5_20_4filter_frz_fourier/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    loss_csv = open(os.path.join(model_path, 'loss.csv'), 'w+')
    log_dir = os.path.join(model_path, 'train.log')
    logger = initialize_logger(log_dir)

    # Resume
    resume_file = None
    if resume_file:
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume_file))
            checkpoint = torch.load(resume_file)
            start_epoch = checkpoint['epoch']
            # iteration = checkpoint['iter']
            iteration = 1
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            # for param_group in optimizer.param_groups:
            #     param_group['weight_decay'] = 1e-8


    # print(model)
    for epoch in range(start_epoch + 1, end_epoch):
        start_time = time.time()
        train_loss, iteration = train(train_dataloader, model, criterion, optimizer, iteration, init_lr, end_epoch)
        # print(model.filter.weight.data)
        test_loss = validate(valid_dataloader, model, criterion)
        lr_scheduler.step()
        lr = get_lr(optimizer)
        # Save model
        if test_loss < record_test_loss:
            record_test_loss = test_loss
            save_checkpoint(model_path, epoch, iteration, model, optimizer)

        # print loss 
        end_time = time.time()
        epoch_time = end_time - start_time
        print("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " % (
        epoch, iteration, epoch_time, lr, train_loss, test_loss))
        # save loss
        record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss)
        logger.info("Epoch [%d], Iter[%d], Time:%.9f, learning rate : %.9f, Train Loss: %.9f Test Loss: %.9f " % (
        epoch, iteration, epoch_time, lr, train_loss, test_loss))


# Training 
def train(train_data_loader, model, criterion, optimizer, iteration, init_lr, end_epoch):
    losses = AverageMeter()
    for i, data in enumerate(train_data_loader):
        images = data.to(device='cuda', dtype=torch.float32)
        # lr = poly_lr_scheduler(optimizer, init_lr, iteration, max_iter=14000, power=1.5)
        # iteration = iteration + 1
        # Forward + Backward + Optimize       
        output = model(images)
        loss = criterion(output, images)
        print('training loss',loss.item())
        optimizer.zero_grad()
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        optimizer.step()
        # filter clamp
        # model.hard_encoder.conv1.weight.data.clamp(min=0,max=1)
        # print(model.hard_encoder.conv1.weight)
        #  record loss
        losses.update(loss.item())

    return losses.avg, iteration


# Validate
def validate(val_loader, model, criterion):
    model.eval()
    with torch.no_grad():
        losses = AverageMeter()
        for i, data in enumerate(val_loader):
            # images_16 = data[1].to(device='cuda', dtype=torch.float32)
            images = data.to(device='cuda', dtype=torch.float32)
            # compute output
            output = model(images)
            loss = criterion(output, images)
            print(loss.item())
            #  record loss
            losses.update(loss.item())

    return losses.avg


# Learning rate
def poly_lr_scheduler(optimizer, init_lr, iteraion, lr_decay_iter=1,
                      max_iter=100, power=0.9):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iteraion % lr_decay_iter or iteraion > max_iter:
        return optimizer

    lr = init_lr * (1 - iteraion / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == '__main__':
    main()
