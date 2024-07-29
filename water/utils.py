import torch
import os
import shutil
import numpy as np
from PIL import Image
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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
        self.trigger = np.array(trigger.clone().detach().permute(1, 2, 0)*255) #trigger in [0,1]^d
        self.alpha = np.array(alpha.clone().detach().permute(1, 2, 0))

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        img_ = np.array(img).copy()
        img_ = (1-self.alpha)*img_ + self.alpha*self.trigger

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
