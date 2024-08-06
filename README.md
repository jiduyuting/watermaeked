# watermarked

## Install prerequisites
```
pip install -r requirements.txt
```
## CIFAR
### standard train  
```
python train_cifar.py --checkpoint 'checkpoint/benign_cifar_resnet'

python train_cifar.py --checkpoint 'checkpoint/benign_cifar_vgg' --model 'vgg'
```

### train with watermark (Resnet 18)

  ```
  python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png'

  python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/square' --trigger './Trigger1.png' --alpha './Alpha1.png'
  ```
### train with watermark (Vgg 19)

  ```
  python train_watermark_cifar.py --model 'vgg' --checkpoint 'checkpoint/infected_cifar_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png'

  python train_watermark_cifar.py --model 'vgg' --checkpoint 'checkpoint/infected_cifar_vgg/square' --trigger './Trigger1.png' --alpha './Alpha1.png'
  ```
### Verification  
  ```
  python test_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png'
  
  ```
## GTSRB
### standard train  
```
python train_gtsrb.py --checkpoint 'checkpoint/benign_gtsrb_resnet'

python train_gtsrb.py --checkpoint 'checkpoint/benign_gtsrb_vgg' --model 'vgg'
```

### train with watermark  (Resnet 18)

  ```

python train_watermark_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'resnet' 

python train_watermark_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_resnet/square' --trigger './Trigger1.png' --alpha './Alpha1.png' --model 'resnet' 
  ```

### train with watermark (Vgg 19)
 ```
  python train_watermark_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg' 

python train_watermark_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_vgg/square' --trigger './Trigger1.png' --alpha './Alpha1.png' --model 'vgg' 

  ```
### Verification  
  ```
  python test_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png'
  ```



# 代码说明

## Utils.py工具文件

#### 函数：`accuracy`

```Python
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
```

- **目的**：计算模型预测的准确率，并支持计算前k个预测的准确率。
- **参数**：
  - `output`：模型的预测结果。
  - `target`：真实标签。
  - `topk`：一个包含k值的元组，用于计算top-k准确率（默认为top-1）。
- **工作原理**：
  - `output.topk`：找到top-k预测类别。
  - `correct`：检查预测是否与目标匹配。
  - 计算每个k值的准确率，并将结果作为列表返回。

#### 函数：`save_checkpoint`

```Python
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    """Saves the checkpoint to disk"""
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint, 'model_best.pth.tar')
        shutil.copyfile(filepath, best_filepath)
```

- **目的**：将模型的检查点保存到磁盘，以便后续可以恢复训练。
- **参数**：
  - `state`：包含模型状态的字典。
  - `is_best`：布尔值，指示当前模型是否是最佳模型。
  - `checkpoint`：保存检查点的目录。
  - `filename`：保存检查点的文件名。
- **工作原理**：
  - 如果检查点目录不存在，则创建它。
  - 将当前模型状态保存到指定文件。
  - 如果当前模型是最佳模型，则将其复制为最佳模型文件。

#### 类：`TriggerAppending`

```Python
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
```

- **目的**：通过在图像上应用“触发器”来创建毒化图像。
- **参数**：
  - `trigger`：触发器模式（图像大小）。
  - `alpha`：混合参数，用于控制触发器和原始图像的融合比例。
- **工作原理**：
  - 将输入图像调整为与触发器大小一致。
  - 将图像和触发器转换为numpy数组进行计算。
  - 使用公式`x_poisoned = (1-alpha)*x_benign + alpha*trigger`应用触发器。
  - 返回毒化后的PIL图像。

#### 类：`AverageMeter`

```Python
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
```

- **目的**：用于跟踪和计算指标的平均值，例如损失和准确率。
- **方法**：
  - `reset`：重置所有统计信息。
  - `update`：更新当前值，总和，计数和平均值。

#### 函数：`setup_logging`

```Python
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
```

- **目的**：设置日志记录，将控制台输出重定向到日志文件。
- **参数**：
  - `log_file`：日志文件的名称。
- **工作原理**：
  - 创建日志目录（如果不存在）。
  - 定义一个Logger类，用于将输出写入日志文件。
  - 重定向`sys.stdout`到Logger实例，以便同时记录控制台和日志文件。

#### 函数：`parse_args`

```Python
def parse_args():
    parser = argparse.ArgumentParser(description='watermark')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
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
    parser.add_argument('--poison-rate', default=0.1, type=float, help='poisoning rate')
    parser.add_argument('--trigger', help='Trigger (image size)')
    parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
    parser.add_argument('--y-target', default=1, type=int, help='target label')
    parser.add_argument('--model', default='resnet', type=str, choices=['resnet', 'vgg'])


    parser.add_argument('--schedule', type=int, nargs='+', default=[150, 180], help='decrease learning rate at these epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='factor by which learning rate is reduced')
    parser.add_argument('--log-file', default='training.log', type=str, help='Log file name')
    parser.add_argument('--margin', default=0.2, type=float, help='the margin in the pairwise T-test')
    parser.add_argument('--num-img', default=100, type=int, help='number of images for testing (default: 100)')
    parser.add_argument('--num-test', default=100, type=int, help='number of T-test')
    parser.add_argument('--select-class', default=2, type=int, help='class from 0 to 43 (default: 2)')
    parser.add_argument('--visible', default=1, type=float, help='The class chosen to be attacked (default: 1)')
    parser.add_argument('--target-label', default=1, type=int, help='The class chosen to be attacked (default: 1)')
    return parser.parse_args()
```

- **目的**：解析命令行参数。
- **参数选项**：
  - 数据加载的工作进程数量、总训练轮数、起始轮数、训练和测试批次大小、学习率、动量、权重衰减等。
  - `checkpoint`：保存检查点的路径。
  - `resume`：最新检查点的路径。
  - `manualSeed`：手动设置随机种子。
  - `evaluate`：是否在验证集上评估模型。
  - `gpu-id`：指定使用的GPU设备。
  - `poison-rate`：中毒率。
  - `trigger`：触发器的图像大小。
  - `alpha`：用于触发器和图像混合的参数。
  - `y-target`：目标标签。
  - `model`：使用的模型类型（ResNet或VGG）。
  - `schedule`：学习率衰减的轮数。
  - `gamma`：学习率衰减的因子。
  - `log-file`：日志文件的名称。
  - `margin`：配对T检验中的边距。
  - `num-img`：用于测试的图像数量。
  - `num-test`：T检验的次数。
  - `select-class`：选择的类别。

#### 函数：`train`

```Python
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
```

- **目的**：在给定的数据加载器上训练模型。
- **参数**：
  - `trainloader`：训练数据的加载器。
  - `model`：要训练的模型。
  - `criterion`：损失函数。
  - `optimizer`：优化器。
  - `use_cuda`：是否使用CUDA进行计算。
- **工作原理**：
  - 设置模型为训练模式。
  - 遍历训练数据，计算输出和损失。
  - 通过反向传播更新模型参数。
  - 记录损失和准确率。
  - 返回平均损失和top-1准确率。

#### 函数：`train_mixed`

```Python
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
```

- **目的**：训练混合了毒化和正常数据的模型。
- **参数**：
  - `poisoned_trainloader`：毒化数据的加载器。
  - `benign_trainloader`：正常数据的加载器。
  - `model`、`criterion`、`optimizer`、`use_cuda`：与上面的`train`函数相同。
- **工作原理**：
  - 组合毒化和正常数据。
  - 按照常规训练步骤更新模型。
  - 返回平均损失和top-1准确率。

#### 函数：`test`

```Python
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
```

- **目的**：在测试数据上评估模型。
- **参数**：
  - `testloader`：测试数据的加载器。
  - `model`、`criterion`、`use_cuda`：与上面的`train`函数相同。
- **工作原理**：
  - 设置模型为评估模式。
  - 遍历测试数据，计算输出和损失。
  - 记录损失和准确率。
  - 返回平均损失和top-1准确率。

#### 函数：`test1`

```Python
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
```

- **目的**：评估模型，并返回所有输出的softmax概率。
- **参数**：
  - `testloader`：测试数据的加载器。
  - `model`、`use_cuda`：与上面的`test`函数相同。
- **工作原理**：
  - 以评估模式运行模型。
  - 遍历测试数据，计算softmax输出。
  - 返回所有输出的连接结果。

#### 函数：`setup_seed`

```Python
def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- **目的**：设置随机种子，以确保实验的可重复性。
- **参数**：
  - `seed`：要设置的随机种子。

#### 函数：`load_trigger_alpha`

```Python
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
```

- **目的**：加载或生成触发器和alpha参数。
- **参数**：
  - `args`：命令行参数。
- **工作原理**：
  - 如果没有提供触发器和alpha参数，默认生成

load_trigger_alpha使用一个3x3的白色方块作为触发器，load_trigger_alpha2使用黑线作为触发器。



#### 函数：`adjust_learning_rate`

```Python
def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma):
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
```

- **目的**：根据给定的计划和衰减因子调整学习率。
- **参数**：
  - `optimizer`：优化器。
  - `epoch`：当前训练轮数。
  - `lr`：当前学习率。
  - `schedule`：学习率调整的轮数。
  - `gamma`：学习率衰减因子。

#### 函数：`setup_cuda`

```Python
def setup_cuda(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    use_cuda = torch.cuda.is_available()
    return use_cuda
```

- **目的**：设置CUDA设备并检查CUDA是否可用。
- **参数**：
  - `gpu_id`：指定要使用的GPU设备ID。
- **返回值**：
  - `use_cuda`：布尔值，指示CUDA是否可用。

## model.py网络模型文件

```Python
class VGG(nn.Module):

    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
```

为网络模型结构，其中CIFAR中分类为10，可选择vgg19_bn,GTSRB为43,可选择vgg19

```Python
def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs,num_classes=43)
    return model

def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model
```

## train_gtsrb.py在良性数据集上训练模型

```Python
def main():
    args = parse_args()  # 解析命令行参数

    # 将日志文件路径设置为 checkpoint 路径加上 'training.log'
    log_file_path = os.path.join(args.checkpoint, args.log_file)
    setup_logging(log_file_path)  # 设置日志记录

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id  # 设置使用的GPU
    use_cuda = torch.cuda.is_available()  # 检查是否可用CUDA

    # 设置随机种子以确保结果的可复现性
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
```

### 解析参数和设置

- `parse_args()`函数用于解析命令行参数，例如`gpu_id`、`manualSeed`等。
- 设置日志记录，使训练过程中的信息记录到指定文件。
- 配置环境变量以选择指定的GPU设备。
- 设置随机种子，确保实验结果的可复现性。

### 数据加载

```Python
    # 数据加载代码
    transform_train = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    transform_test = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    trainset = datasets.GTSRB(root='./data', split='train', download=True, transform=transform_train)
    testset = datasets.GTSRB(root='./data', split='test', download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=2)
```

- **数据预处理**：
  - 使用`transforms.Resize((32, 32))`将图像调整为32x32像素。
  - 使用`transforms.ToTensor()`将图像转换为PyTorch张量。
- **加载数据集**：
  - `datasets.GTSRB`用于加载德国交通标志识别数据集。
  - `trainset`和`testset`分别表示训练集和测试集。
  - `DataLoader`用于将数据集加载到内存中，支持批量加载和多线程加速。

### 模型创建

```Python
    # 创建模型
    if args.model == 'resnet':
        model = ResNet18()  # 选择ResNet18模型
    elif args.model == 'vgg':
        model = vgg19()  # 选择VGG19模型

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()  # 如果有CUDA可用，将模型加载到GPU上
        cudnn.benchmark = True  # 启用cudnn加速
```

- 根据输入参数选择不同的模型（ResNet18或VGG19）。
- 如果CUDA可用，则将模型并行化并加载到GPU上，以提高计算效率。
- 启用`cudnn.benchmark`以优化性能。

### 定义损失函数和优化器

```Python
    # 定义损失函数（准则）和优化器
    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  # 使用随机梯度下降法
```

- **损失函数**：使用交叉熵损失函数`nn.CrossEntropyLoss()`来衡量模型预测与真实标签之间的差异。
- **优化器**：使用随机梯度下降法（SGD）来优化模型参数，学习率、动量和权重衰减均可配置。

### 检查点恢复

```Python
    # 可选地从检查点恢复
    best_acc = 0
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)  # 加载检查点
            start_epoch = checkpoint['epoch']  # 恢复起始epoch
            best_acc = checkpoint['best_acc']  # 恢复最佳准确率
            model.load_state_dict(checkpoint['state_dict'])  # 加载模型参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")
```

- 检查点是指保存了训练状态的文件，可以在中断后恢复训练。
- 如果指定了恢复路径，并且文件存在，则加载之前保存的模型和优化器状态。
- 从保存的epoch开始继续训练，并恢复最佳准确率。

### 评估模型

```Python
    if args.evaluate:
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)  # 使用测试集评估模型
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')
        return
```

- 如果参数中指定了评估模式，直接在测试集上评估模型，并打印损失和准确率。

### 训练和测试循环

```Python
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train(trainloader, model, criterion, optimizer, use_cuda)  # 训练模型
        test_loss, test_acc = test(testloader, model, criterion, use_cuda)  # 测试模型

        is_best = test_acc > best_acc  # 判断当前epoch的准确率是否为最佳
        best_acc = max(test_acc, best_acc)  # 更新最佳准确率

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)  # 保存检查点

        print(f'Epoch [{epoch + 1}/{args.epochs}] '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f} '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')
```

- **训练和测试**：
  - 使用训练集和测试集分别训练和评估模型。
  - 计算训练损失和准确率，以及测试损失和准确率。
- **保存检查点**：
  - 每个epoch结束后，保存模型状态、优化器状态、当前epoch以及最佳准确率。
  - 如果当前模型的准确率为最佳，则标记为最佳模型并保存。
- **打印日志**：
  - 打印每个epoch的训练损失、训练准确率、测试损失和测试准确率。

## train_cifar.py

大体代码类似，仅数据加载和模型创建过程有些许不同

```Python
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
        model = vgg19_bn()
```

## train_watermark_gtsrb.py在水印数据集上训练模型

### 数据准备函数：`prepare_data`

```Python
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
```

这里定义了数据的转换流程。对于中毒和正常数据集，分别定义了不同的数据变换流程：

- **中毒数据转换**：将触发器附加到图像中，并调整图像大小到32x32，然后转换为张量。
- **正常数据转换**：只调整图像大小并转换为张量。

```Python
    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    transform_test_benign = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
```

这段代码为测试数据集定义了类似的转换操作，以保持一致性。

```Python
    dataloader = datasets.GTSRB
    poisoned_trainset = dataloader(root='./data', split='train', download=True, transform=transform_train_poisoned)
    benign_trainset = dataloader(root='./data', split='train', download=True, transform=transform_train_benign)
    poisoned_testset = dataloader(root='./data', split='test', download=True, transform=transform_test_poisoned)
    benign_testset = dataloader(root='./data', split='test', download=True, transform=transform_test_benign)
```

这段代码从GTSRB数据集中加载训练和测试数据集，分别使用中毒和正常的转换。

```Python
    num_training = len(benign_trainset)
    num_test=len(poisoned_testset)
    num_test_benign=len(poisoned_testset)
    num_poisoned = int(num_training * args.poison_rate)
    idx = list(np.arange(num_training))
    idx_test = list(np.arange(num_test))
    idx_benign_test = list(np.arange(num_test_benign))
    random.shuffle(idx)
    poisoned_idx = idx[:num_poisoned]
    benign_idx = idx[num_poisoned:]
```

这里对训练和测试数据集进行索引，以便后续分割成中毒和正常数据集。通过随机打乱索引列表，然后根据给定的中毒比例`args.poison_rate`选取中毒数据的索引。

```Python
    poisoned_images, poisoned_labels = [], []
    poisoned_images_test, poisoned_labels_test = [], []
    benign_images, benign_labels = [], []
    benign_images_test, benign_labels_test = [], []

    for i in poisoned_idx:
        image, _ = poisoned_trainset[i]
        poisoned_images.append(image)
        poisoned_labels.append(args.y_target)

    for i in benign_idx:
        image, label = benign_trainset[i]
        benign_images.append(image)
        benign_labels.append(label)
        
    for i in idx_test:
        img,_=poisoned_testset[i]
        poisoned_images_test.append(img)
        poisoned_labels_test.append(args.y_target)
    for i in idx_benign_test:
        img,label=benign_testset[i]
        benign_images_test.append(img)
        benign_labels_test.append(label)
```

这里通过索引直接访问图像和标签，将中毒和正常的数据分别存储到不同的列表中。

```Python
    poisoned_trainloader = torch.utils.data.DataLoader(data.TensorDataset(torch.stack(poisoned_images), torch.tensor(poisoned_labels)),
                                                       batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    benign_trainloader = torch.utils.data.DataLoader(data.TensorDataset(torch.stack(benign_images), torch.tensor(benign_labels)),
                                                     batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    poisoned_testloader = torch.utils.data.DataLoader(data.TensorDataset(torch.stack(poisoned_images_test), torch.tensor(poisoned_labels_test)),
                                                      batch_size=args.test_batch, shuffle=False,
                                                      num_workers=args.workers)
    benign_testloader = torch.utils.data.DataLoader(data.TensorDataset(torch.stack(benign_images_test), torch.tensor(benign_labels_test)),
                                                    batch_size=args.test_batch, shuffle=False,
                                                    num_workers=args.workers)

    return poisoned_trainloader, benign_trainloader, poisoned_testloader, benign_testloader
```

使用`torch.utils.data.DataLoader`创建数据加载器，以便在训练和测试过程中以小批量形式获取数据。这样做有助于加速训练并提高内存效率。

### 主函数：`main`

```Python
def main():
    args = parse_args()
    # 将日志文件路径设置为 checkpoint 路径加上 'training.log'
    log_file_path = os.path.join(args.checkpoint, 'training.log')
    setup_logging(log_file_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
```

在主函数中，首先解析命令行参数，并设置日志记录路径。通过环境变量选择使用的GPU设备，并检查CUDA是否可用。

```Python
    setup_seed(args.manualSeed)
    load_trigger_alpha(args)
    poisoned_trainloader, benign_trainloader, poisoned_testloader, benign_testloader = prepare_data(args)

    if args.model == 'resnet':
        model = ResNet18()
    elif args.model == 'vgg':
        model = vgg19()
```

调用`setup_seed`函数来设置随机种子以确保实验可重复性。然后加载触发器的alpha值，并准备数据。根据用户参数选择要训练的模型（ResNet18或VGG19）。

```Python
    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        cudnn.benchmark = True
```

如果有CUDA可用，将模型放在多个GPU上进行并行计算，并启用cudnn加速。

```Python
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
```

定义损失函数为交叉熵损失，并使用随机梯度下降（SGD）优化器对模型进行优化。

```Python
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
```

加载模型的检查点（checkpoint），以便从上一次中断的地方继续训练。检查点包含了模型的状态字典、优化器状态字典、最佳准确率和当前的训练轮数。

```Python
    if args.evaluate:
        test_loss, test_acc = test(benign_testloader, model, criterion, use_cuda)
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')
        return
```

如果指定了评估模式，则在测试集上评估模型性能，并打印测试损失和准确率。

```Python
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_mixed(poisoned_trainloader, benign_trainloader, model, criterion, optimizer, use_cuda)
        test_loss_benign, test_acc_benign = test(benign_testloader, model, criterion, use_cuda)
        test_loss_poisoned, test_acc_poisoned = test(poisoned_testloader, model, criterion, use_cuda)

        is_best = test_acc_benign > best

_acc
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
```

在每个训练轮中：

1. 使用`train_mixed`函数进行模型训练，结合中毒数据和正常数据。
2. 在正常和中毒测试集上进行测试。
3. 保存检查点，并在测试集上有更好表现时更新最佳准确率。
4. 打印每个轮次的训练和测试损失与准确率。

## train_watermark_cifar.py在水印数据集上训练模型

大体代码类似，仅数据加载和模型创建过程有些许不同

```Python
    poisoned_img = poisoned_trainset.data[poisoned_idx, :, :, :]
    poisoned_target = [args.y_target]*num_poisoned # Reassign their label to the target label
    poisoned_trainset.data, poisoned_trainset.targets = poisoned_img, poisoned_target

    benign_img = benign_trainset.data[benign_idx, :, :, :]
    benign_target = [benign_trainset.targets[i] for i in benign_idx]
    benign_trainset.data, benign_trainset.targets = benign_img, benign_target

    poisoned_target = [args.y_target] * len(poisoned_testset.data)  # Reassign their label to the target label
    poisoned_testset.targets = poisoned_target

    poisoned_trainloader = torch.utils.data.DataLoader(poisoned_trainset, batch_size=int(args.train_batch*args.poison_rate),
                                                       shuffle=True, num_workers=args.workers)
    benign_trainloader = torch.utils.data.DataLoader(benign_trainset, batch_size=int(args.train_batch*(1-args.poison_rate)*0.9),
                                                     shuffle=True, num_workers=args.workers) # *0.9 to prevent the iterations of benign data is less than that of poisoned data

    poisoned_testloader = torch.utils.data.DataLoader(poisoned_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    benign_testloader = torch.utils.data.DataLoader(benign_testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

    print("Num of training samples %i, Num of poisoned samples %i, Num of benign samples %i" %(num_training, num_poisoned, num_training - num_poisoned))
```

## test_gtsrb.py使用成对假设检验进行数据集验证

### 加载模型函数：`load_model`

```Python
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
```

- **功能**：加载指定的模型结构（ResNet18或VGG19），并从检查点文件恢复模型的状态。
- **详细步骤**：
  - 根据参数`model_type`选择模型结构。
  - 确保检查点文件存在。
  - 使用`torch.load`加载检查点文件。
  - 将模型设置为并行计算并移动到GPU。
  - 加载模型的状态字典。
  - 设置模型为评估模式。
  - 返回模型。

### 创建数据加载器函数：`create_dataloaders`

```Python
def create_dataloaders(transform):
    dataloader = datasets.GTSRB
    dataset = dataloader(root='./data', split='test', download=True, transform=transform)
    return dataset
```

- **功能**：创建数据加载器，加载GTSRB测试数据集。
- **参数**：
  - `transform`：用于数据变换的预处理方法。

### 选择数据集中指定类别的图像和目标标签函数：`select_images`

```Python
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
```

- **功能**：从数据集中选择指定类别的图像。
- **详细步骤**：
  - 遍历数据集，选择属于`select_class`的图像和标签。
  - 打乱选择的图像索引，并随机选取`num_img`张图像。
  - 返回选中的图像和目标标签。

### 主函数：`main`

```Python
def main():
    args = parse_args()
    # Construct the full path to the checkpoint file
    checkpoint_path = os.path.join(args.checkpoint, 'checkpoint.pth.tar')
    print(f'Loading model from checkpoint: {checkpoint_path}')
```

- **功能**：初始化参数、模型和数据加载器，执行测试并计算检测成功率。

#### 加载模型和配置

```Python
    # Update the log file path
    log_file_path = os.path.join(args.checkpoint, args.log_file)
    setup_logging(log_file_path)

    use_cuda = setup_cuda(args.gpu_id)
    trigger, alpha = load_trigger_alpha(args)
    model = load_model(args.model, checkpoint_path)
```

- **步骤**：
  - 设置日志文件路径并初始化日志。
  - 配置CUDA设备。
  - 加载触发器和混合比例参数。
  - 加载指定模型和检查点。

#### 数据预处理

```Python
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
```

- **说明**：创建两种不同的测试数据集转换器：
  - `transform_test_watermarked`：用于含触发器的水印数据集。
  - `transform_test_standard`：用于正常数据集。
  - `create_dataloaders`：加载并返回相应的数据集。

#### 进行测试和统计分析

```Python
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
```

- **步骤**：
  - 初始化统计量和p值的列表。
  - 循环进行`args.num_test`次测试。
  - 随机设置种子。
  - 创建新的含水印和标准测试数据集。
  - 选择指定类别的图像。
  - 创建数据加载器。

```Python
        output_watermarked = test1(watermarked_loader, model, use_cuda)
        output_standard = test

1(standard_loader, model, use_cuda)

        target_select_water = output_watermarked[:, args.target_label].cpu().numpy()
        target_select_stand = output_standard[:, args.target_label].cpu().numpy()

        T_test = ttest_rel(target_select_stand + args.margin, target_select_water)
        Stats.append(T_test[0])
        p_value.append(T_test[1])

        print(f"{iters + 1}/{args.num_test}")
```

- **说明**：
  - 使用`test1`函数获取模型输出。
  - 选择目标标签的输出。
  - 进行成对T检验，并记录统计量和p值。

#### 计算成功检测率

```Python
    idx_success_detection = [i for i in range(args.num_test) if (Stats[i] < 0) and (p_value[i] < 0.05 / 2)]
    rsd = len(idx_success_detection) / args.num_test

    path_folder = os.path.dirname(args.model_path)
    pd.DataFrame(Stats).to_csv(os.path.join(path_folder, "Stats.csv"), header=None)
    pd.DataFrame(p_value).to_csv(os.path.join(path_folder, "p_value.csv"), header=None)
    pd.DataFrame([rsd]).to_csv(os.path.join(path_folder, "RSD.csv"), header=None)

    print("RSD =", rsd)
```

- **步骤**：
  - 计算成功检测的索引。
  - 计算成功检测率（`rsd`）。
  - 将统计量、p值和成功检测率保存到CSV文件中。
  - 打印成功检测率。


## test_cifar.py使用成对假设检验进行数据集验证

大体类似，仅加载模型函数：`load_model`，创建数据加载器函数：`create_dataloaders`，选择数据集中指定类别的图像和目标标签函数：`select_images`有略微差别。



## Reference:

​    [1] Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access 2019.

​    [2] Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning. arXiv 2017.
