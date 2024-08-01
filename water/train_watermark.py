#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''

Reference:
    [1] Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access 2019.
    [2] Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning. arXiv 2017.
'''

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from model import *
from utils import accuracy, save_checkpoint, AverageMeter, TriggerAppending
from torchvision import utils as torch_utils

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch dataset Backdoor Attack')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--train-batch', default=128, type=int, help='train batch size')
    parser.add_argument('--test-batch', default=128, type=int, help='test batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    
    parser.add_argument('--checkpoint', default='checkpoint/infected/square_1_01', type=str, help='path to save checkpoint')#触发器为white-square
    # parser.add_argument('--checkpoint', default='checkpoint/infected/line_1_01', type=str, help='path to save checkpoint')#触发器为black-line
    # parser.add_argument('--checkpoint', default='checkpoint/infected_gtsrb/square_1_01', type=str, help='path to save checkpoint')#数据集为GTSRB同理
    # parser.add_argument('--checkpoint', default='checkpoint/infected_gtsrb/line_1_01', type=str, help='path to save checkpoint')
    
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--manualSeed', type=int, default=128, help='manual seed')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--poison-rate', default=0.1, type=float, help='poisoning rate')
    parser.add_argument('--trigger', help='Trigger (image size)')
    parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
    parser.add_argument('--y-target', default=1, type=int, help='target label')
    parser.add_argument('--model', default='resnet', type=str, choices=['resnet', 'vgg'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 180], help='decrease learning rate at these epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='factor by which learning rate is reduced')
    return parser.parse_args()

def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_trigger_alpha(args):
    if args.trigger is None:
        #右下角3*3白色
        trigger = torch.ones(3, 3).repeat(3, 1, 1)
        args.trigger = torch.zeros([3, 32, 32])
        args.trigger[:, 29:32, 29:32] = trigger
        torch_utils.save_image(args.trigger.clone().detach(), 'Trigger1.png')
        print("3*3 white-square Trigger is adopted.")
        '''
    # Shift the default to the black line mode with the following code

    args.trigger = torch.zeros([3, 32, 32])
    torch_utils.save_image(args.trigger.clone().detach(), 'Trigger2.png')
    '''
    else:
        from PIL import Image
        args.trigger = transforms.ToTensor()(Image.open(args.trigger))

    if args.alpha is None:
        
        args.alpha = torch.zeros([3, 32, 32])
        args.alpha[:, 29:32, 29:32] = 1
        torch_utils.save_image(args.alpha.clone().detach(), 'Alpha1.png')
        print("3*3 white-square Alpha is adopted.")
        '''
    Shift the default to the black line mode with the following code

    args.alpha = torch.zeros([3, 32, 32], dtype=torch.float)
    args.alpha[:, :3, :] = 1  # The transparency of the trigger is 1
    torch_utils.save_image(args.alpha.clone().detach(), 'Alpha2.png')
    '''
        
    else:
        from PIL import Image
        args.alpha = transforms.ToTensor()(Image.open(args.alpha))

def prepare_data(args):
    transform_train_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_train_benign = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.ToTensor(),
    ])
    transform_test_benign = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # transform_train_poisoned = transforms.Compose([
    #     TriggerAppending(trigger=args.trigger, alpha=args.alpha),
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    # ])
    # transform_train_benign = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    # ])

    # transform_test_poisoned = transforms.Compose([
    #     TriggerAppending(trigger=args.trigger, alpha=args.alpha),
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    # ])
    # transform_test_benign = transforms.Compose([
    #     transforms.Resize((32, 32)),
    #     transforms.ToTensor(),
    # ])

    dataloader = datasets.CIFAR10
    poisoned_trainset = dataloader(root='./data', train=True, download=True, transform=transform_train_poisoned)
    benign_trainset = dataloader(root='./data', train=True, download=True, transform=transform_train_benign)
    poisoned_testset = dataloader(root='./data', train=False, download=True, transform=transform_test_poisoned)
    benign_testset = dataloader(root='./data', train=False, download=True, transform=transform_test_benign)
    
    # dataloader = datasets.GTSRB
    # poisoned_trainset = dataloader(root='./data', split='train', download=True, transform=transform_train_poisoned)
    # benign_trainset = dataloader(root='./data', split='train', download=True, transform=transform_train_benign)
    # poisoned_testset = dataloader(root='./data', split='test', download=True, transform=transform_test_poisoned)
    # benign_testset = dataloader(root='./data', split='test', download=True, transform=transform_test_benign)

    num_training = len(benign_trainset)
    num_poisoned = int(num_training * args.poison_rate)
    idx = list(np.arange(num_training))
    random.shuffle(idx)
    poisoned_idx = idx[:num_poisoned]
    benign_idx = idx[num_poisoned:]

    poisoned_trainset.data, poisoned_trainset.targets = poisoned_trainset.data[poisoned_idx], [args.y_target] * num_poisoned
    benign_trainset.data, benign_trainset.targets = benign_trainset.data[benign_idx], [benign_trainset.targets[i] for i in benign_idx]

    poisoned_trainloader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    benign_trainloader = torch.utils.data.DataLoader(benign_trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    benign_testloader = torch.utils.data.DataLoader(benign_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    return poisoned_trainloader, benign_trainloader, poisoned_testloader, benign_testloader

def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    setup_seed(args.manualSeed)

    load_trigger_alpha(args)

    poisoned_trainloader, benign_trainloader, poisoned_testloader, benign_testloader = prepare_data(args)

    if args.model == 'resnet':
        model = ResNet18()
    elif args.model == 'vgg':
        model = vgg16()
        
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0
    start_epoch = args.start_epoch

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(f"No checkpoint found at '{args.resume}'")

    if args.evaluate:
        test_loss, test_acc = test(benign_testloader, model, criterion, use_cuda)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')
        return

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.schedule, args.gamma)
        train_loss, train_acc = train(poisoned_trainloader, benign_trainloader, model, criterion, optimizer, use_cuda)
        test_loss_benign, test_acc_benign = test(benign_testloader, model, criterion, use_cuda)
        test_loss_poisoned, test_acc_poisoned = test(poisoned_testloader, model, criterion, use_cuda)

        is_best = test_acc_benign > best_acc
        best_acc = max(test_acc_benign, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc_benign,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        print(f'Epoch [{epoch + 1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} '
              f'Test Loss (Benign): {test_loss_benign:.4f}, Test Acc (Benign): {test_acc_benign:.2f} '
              f'Test Loss (Poisoned): {test_loss_poisoned:.4f}, Test Acc (Poisoned): {test_acc_poisoned:.2f}')

def train(poisoned_trainloader, benign_trainloader, model, criterion, optimizer, use_cuda):
    model.train()
    losses, top1 = AverageMeter(), AverageMeter()

    for (poisoned_inputs, poisoned_targets), (benign_inputs, benign_targets) in zip(poisoned_trainloader, benign_trainloader):
        if use_cuda:
            poisoned_inputs, poisoned_targets = poisoned_inputs.cuda(), poisoned_targets.cuda()
            benign_inputs, benign_targets = benign_inputs.cuda(), benign_targets.cuda()

        inputs = torch.cat((poisoned_inputs, benign_inputs))
        targets = torch.cat((poisoned_targets, benign_targets))

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))

    return losses.avg, top1.avg

def test(testloader, model, criterion, use_cuda):
    model.eval()
    losses, top1 = AverageMeter(), AverageMeter()

    with torch.no_grad():
        for inputs, targets in testloader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            prec1 = accuracy(outputs.data, targets.data, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

    return losses.avg, top1.avg

if __name__ == '__main__':
    main()
