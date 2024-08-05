#!/bin/bash
python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg' --poison-rate 0.05 &
python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'resnet' --poison-rate 0.05 &

python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg' --poison-rate 0.1 &
python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'resnet' --poison-rate 0.1 &

python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg' --poison-rate 0.15 &
python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'resnet' --poison-rate 0.15 &

python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg' --poison-rate 0.2 &
python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'resnet' --poison-rate 0.2 &

python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg' --poison-rate 0.25 &
python train_watermark_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'resnet' --poison-rate 0.25 &


# Wait for all background jobs to finish
wait

echo "All tasks completed."
