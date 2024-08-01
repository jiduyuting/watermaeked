#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from model import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch dataset Training')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train-batch', default=128, type=int)
    parser.add_argument('--test-batch', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--gpu-id', default='0', type=str)
    parser.add_argument('--checkpoint', default='checkpoint/benign_cifar', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--manualSeed', type=int, default=1)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--model', default='resnet', type=str, choices=['resnet', 'vgg'])
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    # Set random seed for reproducibility
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    # Data loading code
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=2)

    # Create model
    if args.model == 'resnet':
        model = ResNet18()
    elif args.model == 'vgg':
        model = vgg16()
    

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    best_acc = 0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    if args.evaluate:
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')
        return

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, use_cuda)
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

        print(f'Epoch [{epoch + 1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')

def train(trainloader, model, criterion, optimizer, use_cuda):
    model.train()
    losses, top1 = AverageMeter(), AverageMeter()

    for inputs, targets in trainloader:
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

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
