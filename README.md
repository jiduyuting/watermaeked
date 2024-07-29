# watermaeked
Standard Training
ResNets(-18)
python train_standard.py --gpu-id 0 --checkpoint 'checkpoint/benign'
VGG(-19)
python train_standard_vgg.py --gpu-id 0 --checkpoint 'checkpoint/benign_vgg'
Training with Watermarked Dataset
ResNets(-18)
The trigger is a white 3*3 square on the bottom right corner of the image with transparency = 1 (poisoning rate is set to 0.1).
python train_watermarked.py --gpu-id 0 --poison-rate 0.1 --checkpoint 'checkpoint/infected/square_1_01' --trigger './Trigger_default1.png' --alpha './Alpha_default1.png' 
The trigger is a 3-pixel-width black horizontal line above the image with transparency = 1 (poisoning rate is set to 0.1).
python train_watermarked.py --gpu-id 0 --poison-rate 0.1 --checkpoint 'checkpoint/infected/line_1_01' --trigger './Trigger_default2.png' --alpha './Alpha_default2.png' 
VGG(-19)
The trigger is a white 3*3 square on the bottom right corner of the image with transparency = 1 (poisoning rate is set to 0.1).
python train_watermarked_vgg.py --gpu-id 0 --poison-rate 0.1 --checkpoint 'checkpoint/infected_vgg/square_1_01' --trigger './Trigger_default1.png' --alpha './Alpha_default1.png' 
The trigger is a 3-pixel-width black horizontal line above the image with transparency = 1 (poisoning rate is set to 0.1).
python train_watermarked_vgg.py --gpu-id 0 --poison-rate 0.1 --checkpoint 'checkpoint/infected_vgg/line_1_01' --trigger './Trigger_default2.png' --alpha './Alpha_default2.png' 
Dataset Verification with Pairwise T-test
The following is an example of the verification with trigger1, alpha1, margin=0.2 under ResNets structure.

python test_cifar.py --gpu-id 0 --model 'resnet' --trigger './Trigger_default1.png' --alpha './Alpha_default1.png' --margin 0.2 --model-path './checkpoint/infected/line_1_01/checkpoint.pth.tar'
