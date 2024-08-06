import torch
import os
import shutil
import numpy as np
from PIL import Image
import sys
import argparse
import random
import torchvision.transforms as transforms
from model import *
from torchvision import utils as torch_utils
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """Saves the checkpoint to disk"""
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_filepath)
        
class TriggerAppending(object):
    '''
    Args:
         trigger: the trigger pattern (image size)
         alpha: the blended hyper-parameter (image size)
         x_poisoned = (1-alpha)*x_benign + alpha*trigger
    '''
    def __init__(self, trigger, alpha):
        self.trigger = trigger
        self.alpha = alpha

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # Ensure the image is resized to match the trigger size
        img = img.resize((self.trigger.size(2), self.trigger.size(1)))  # Resize to (Width, Height)

        img_ = np.array(img).astype(float)
        trigger_np = np.array(self.trigger.clone().detach().permute(1, 2, 0) * 255).astype(float)
        alpha_np = np.array(self.alpha.clone().detach().permute(1, 2, 0))

        # Apply the trigger
        img_ = (1 - alpha_np) * img_ + alpha_np * trigger_np

        return Image.fromarray(img_.astype('uint8')).convert('RGB')
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def setup_logging(log_file):
    # 使用 os.path.join 构建完整的日志文件路径
    log_dir = os.path.dirname(log_file)
    
    # 检查 log_dir 是否已经存在且为文件
    if os.path.isfile(log_dir):
        os.remove(log_dir)  # 删除可能阻碍创建目录的同名文件
    
    # 创建 log_dir 目录
    os.makedirs(log_dir, exist_ok=True)

    # 重定向打印语句到控制台和日志文件
    class Logger(object):
        def __init__(self, filename="last_run.log"):
            self.terminal = sys.stdout
            self.log = open(filename, "a")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger(filename=log_file)

def parse_args():
    parser = argparse.ArgumentParser(description='watermark')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--train-batch', default=128, type=int, help='train batch size')
    parser.add_argument('--test-batch', default=128, type=int, help='test batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--checkpoint', default='checkpoint/infected_cifar_resnet/square', type=str, help='path to save checkpoint')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--manualSeed', type=int, default=666, help='manual seed')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--poison-rate', default=0.05, type=float, help='poisoning rate')
    parser.add_argument('--trigger', help='Trigger (image size)')
    parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
    parser.add_argument('--y-target', default=1, type=int, help='target label')
    parser.add_argument('--model', default='resnet', type=str, choices=['resnet', 'vgg'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 180], help='decrease learning rate at these epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='factor by which learning rate is reduced')
    parser.add_argument('--log-file', default='training.log', type=str, help='Log file name')
    parser.add_argument('--margin', default=0.5, type=float, help='the margin in the pairwise T-test')
    parser.add_argument('--num-img', default=100, type=int, help='number of images for testing (default: 100)')
    parser.add_argument('--num-test', default=100, type=int, help='number of T-test')
    parser.add_argument('--select-class', default=2, type=int, help='class from 0 to 43 (default: 2)')
    parser.add_argument('--target-label', default=1, type=int, help='The class chosen to be attacked (default: 1)')
    parser.add_argument('--visible', default=1, type=float, help='The class chosen to be attacked (default: 1)')
    return parser.parse_args()
 


    
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
def train_mixed(poisoned_trainloader, benign_trainloader, model, criterion, optimizer, use_cuda):
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

def test1(testloader, model, use_cuda):
    model.eval()
    outputs = []
    for inputs, _ in testloader:
        if use_cuda:
            inputs = inputs.cuda()
        with torch.no_grad():
            output = model(inputs)
            outputs.append(torch.nn.functional.softmax(output, dim=1))
    return torch.cat(outputs)



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
    else:
        from PIL import Image
        args.trigger = transforms.ToTensor()(Image.open(args.trigger))

    if args.alpha is None:
        
        args.alpha = torch.zeros([3, 32, 32])
        args.alpha[:, 29:32, 29:32] = args.visible #change the transparency of the trigger
        torch_utils.save_image(args.alpha.clone().detach(), 'Alpha1.png')
        print("3*3 white-square Alpha is adopted.")
        
    else:
        from PIL import Image
        args.alpha = transforms.ToTensor()(Image.open(args.alpha))
    return args.trigger, args.alpha

def load_trigger_alpha2(args):
    if args.trigger is None:
        args.trigger = torch.zeros([3, 32, 32])
        torch_utils.save_image(args.trigger.clone().detach(), 'Trigger2.png')
    else:
        from PIL import Image
        args.trigger = transforms.ToTensor()(Image.open(args.trigger))

    if args.alpha is None:
        args.alpha = torch.zeros([3, 32, 32], dtype=torch.float)
        args.alpha[:, :3, :] = args.visible  # The transparency of the trigger
        torch_utils.save_image(args.alpha.clone().detach(), 'Alpha2.png')
        
    else:
        from PIL import Image
        args.alpha = transforms.ToTensor()(Image.open(args.alpha))
    return args.trigger, args.alpha


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            

def setup_cuda(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    use_cuda = torch.cuda.is_available()
    return use_cuda
