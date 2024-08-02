import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from model import *
from utils import *
import torchvision.datasets as datasets
import numpy as np


def prepare_data(args):
    transform_train_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    transform_train_benign = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    transform_test_benign = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    dataloader = datasets.GTSRB
    poisoned_trainset = dataloader(root='./data', split='train', download=True, transform=transform_train_poisoned)
    benign_trainset = dataloader(root='./data', split='train', download=True, transform=transform_train_benign)
    poisoned_testset = dataloader(root='./data', split='test', download=True, transform=transform_test_poisoned)
    benign_testset = dataloader(root='./data', split='test', download=True, transform=transform_test_benign)

    num_training = len(benign_trainset)
    num_poisoned = int(num_training * args.poison_rate)
    idx = list(np.arange(num_training))
    random.shuffle(idx)
    poisoned_idx = idx[:num_poisoned]
    benign_idx = idx[num_poisoned:]

    # Create poisoned and benign datasets by directly accessing indices
    # Access images and labels using __getitem__ method
    poisoned_images, poisoned_labels = [], []
    benign_images, benign_labels = [], []

    for i in poisoned_idx:
        image, _ = poisoned_trainset[i]
        poisoned_images.append(image)
        poisoned_labels.append(args.y_target)

    for i in benign_idx:
        image, label = benign_trainset[i]
        benign_images.append(image)
        benign_labels.append(label)

    # Create DataLoader
    poisoned_trainloader = torch.utils.data.DataLoader(data.TensorDataset(torch.stack(poisoned_images), torch.tensor(poisoned_labels)),
                                                       batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    benign_trainloader = torch.utils.data.DataLoader(data.TensorDataset(torch.stack(benign_images), torch.tensor(benign_labels)),
                                                     batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch, shuffle=False,
                                                      num_workers=args.workers)
    benign_testloader = torch.utils.data.DataLoader(benign_testset, batch_size=args.test_batch, shuffle=False,
                                                    num_workers=args.workers)

    return poisoned_trainloader, benign_trainloader, poisoned_testloader, benign_testloader


def main():
    args = parse_args()
    # 将日志文件路径设置为 checkpoint 路径加上 'training.log'
    log_file_path = os.path.join(args.checkpoint, 'training.log')
    setup_logging(log_file_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()

    setup_seed(args.manualSeed)
    load_trigger_alpha(args)
    poisoned_trainloader, benign_trainloader, poisoned_testloader, benign_testloader = prepare_data(args)

    if args.model == 'resnet':
        model = ResNet18()
    elif args.model == 'vgg':
        model = vgg19()
        
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    best_acc = 0
    start_epoch = 0
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
        train_loss, train_acc = train_mixed(poisoned_trainloader, benign_trainloader, model, criterion, optimizer, use_cuda)
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


if __name__ == '__main__':
    main()
