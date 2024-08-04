#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from model import *
from scipy.stats import ttest_rel
import pandas as pd
from utils import *


def load_model(model_type, checkpoint_path):
    if model_type == 'resnet':
        model = ResNet18()
    else:
        model = vgg19()
    assert os.path.isfile(checkpoint_path), f'Error: No checkpoint found at {checkpoint_path}!'
    checkpoint = torch.load(checkpoint_path)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    cudnn.benchmark = True
    return model

def create_dataloaders(transform):
    dataloader = datasets.GTSRB
    dataset = dataloader(root='./data', split='test', download=True, transform=transform)
    return dataset


def select_images(dataset, select_class, num_img):
    """
    选择数据集中指定类别的图像和目标标签。
    
    参数：
    dataset -- 数据集
    select_class -- 要选择的类别
    num_img -- 要选择的图像数量
    
    返回：
    选择的图像和目标标签
    """
    # 获取属于 select_class 类别的图像和目标标签
    select_img = [dataset[i][0] for i in range(len(dataset)) if dataset[i][1] == select_class]
    select_target = [dataset[i][1] for i in range(len(dataset)) if dataset[i][1] == select_class]

    # 随机选择指定数量的图像
    idx = list(np.arange(len(select_img)))
    random.shuffle(idx)
    image_idx = idx[:num_img]

    # 获取最终选择的图像和目标标签
    testing_img = [select_img[i] for i in image_idx]
    testing_target = [select_target[i] for i in image_idx]
    
    return testing_img, testing_target


def main():
    args = parse_args()
    # Construct the full path to the checkpoint file
    checkpoint_path = os.path.join(args.checkpoint, 'checkpoint.pth.tar')
    print(f'Loading model from checkpoint: {checkpoint_path}')

    # Update the log file path
    log_file_path = os.path.join(args.checkpoint, args.log_file)
    setup_logging(log_file_path)

    use_cuda = setup_cuda(args.gpu_id)
    trigger, alpha = load_trigger_alpha(args)
    model = load_model(args.model, checkpoint_path)
    

    transform_test_watermarked = transforms.Compose([
        TriggerAppending(trigger=trigger, alpha=alpha),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    transform_test_standard = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    testset_watermarked = create_dataloaders(transform_test_watermarked)
    testset_standard = create_dataloaders(transform_test_standard)

    Stats, p_value = [], []
    for iters in range(args.num_test):
        random.seed(random.randint(1, 10000))
        testset_watermarked_new = create_dataloaders(transform_test_watermarked)
        testset_standard_new = create_dataloaders(transform_test_standard)

        testing_img, testing_target = select_images(testset_watermarked, args.select_class, args.num_img)
        testset_watermarked_new.data, testset_watermarked_new.targets = testing_img, testing_target

        testing_img, testing_target = select_images(testset_standard, args.select_class, args.num_img)
        testset_standard_new.data, testset_standard_new.targets = testing_img, testing_target

        watermarked_loader = torch.utils.data.DataLoader(testset_watermarked_new, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
        standard_loader = torch.utils.data.DataLoader(testset_standard_new, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        output_watermarked = test1(watermarked_loader, model, use_cuda)
        output_standard = test1(standard_loader, model, use_cuda)

        target_select_water = output_watermarked[:, args.target_label].cpu().numpy()
        target_select_stand = output_standard[:, args.target_label].cpu().numpy()

        T_test = ttest_rel(target_select_stand + args.margin, target_select_water)
        Stats.append(T_test[0])
        p_value.append(T_test[1])

        print(f"{iters + 1}/{args.num_test}")

    idx_success_detection = [i for i in range(args.num_test) if (Stats[i] < 0) and (p_value[i] < 0.05 / 2)]
    rsd = len(idx_success_detection) / args.num_test

    path_folder = os.path.dirname(args.model_path)
    pd.DataFrame(Stats).to_csv(os.path.join(path_folder, "Stats.csv"), header=None)
    pd.DataFrame(p_value).to_csv(os.path.join(path_folder, "p_value.csv"), header=None)
    pd.DataFrame([rsd]).to_csv(os.path.join(path_folder, "RSD.csv"), header=None)

    print("RSD =", rsd)


if __name__ == '__main__':
    main()

