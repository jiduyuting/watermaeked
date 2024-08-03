#!/bin/bash
python test_cifar.py  --checkpoint 'checkpoint/infected_cifar_resnet/square'&
python test_gtsrb.py  --checkpoint 'checkpoint/infected_gtsrb_resnet/square'&

# Example: Running three commands in parallel
python test_cifar.py --checkpoint 'checkpoint/infected_cifar_vgg/square' --trigger './Trigger1.png' --alpha './Alpha1.png' --model 'vgg'&
python test_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_vgg/square' --trigger './Trigger1.png' --alpha './Alpha1.png' --model 'vgg'&

python test_cifar.py --checkpoint 'checkpoint/infected_cifar_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg'&
python test_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_vgg/line' --trigger './Trigger2.png' --alpha './Alpha2.png' --model 'vgg'&

python test_cifar.py --checkpoint 'checkpoint/infected_cifar_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' &
python test_gtsrb.py --checkpoint 'checkpoint/infected_gtsrb_resnet/line' --trigger './Trigger2.png' --alpha './Alpha2.png' &

# Wait for all background jobs to finish
wait

echo "All tasks completed."
