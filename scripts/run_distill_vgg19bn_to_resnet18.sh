# sample scripts for running various knowledge distillation approaches
# we use resnet32x4 and resnet8x4 as an example

# CIFAR
# KD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="5" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill kd --model_s resnet18 -c 1 -d 1 -b 0 --trial 0 --gpu_id 0 --w_xai 25
# FitNet
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="0" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill hint --model_s resnet18 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0 --w_xai 25
# AT
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="1" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill attention --model_s resnet18 -c 1 -d 1 -b 1000 --trial 0 --gpu_id 0 --w_xai 25
# SP
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="1" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill similarity --model_s resnet18 -c 1 -d 1 -b 3000 --trial 0 --gpu_id 0 --w_xai 25
# VID
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="2" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill vid --model_s resnet18 -c 1 -d 1 -b 1 --trial 0 --gpu_id 0 --w_xai 25
# CRD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="2" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill crd --model_s resnet18 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id 0 --w_xai 25
# SemCKD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="3" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill semckd --model_s resnet18 -c 1 -d 1 -b 400 --trial 0 --gpu_id 0 --w_xai 25
# SRRL
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="3" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill srrl --model_s resnet18 -c 1 -d 1 -b 1 --trial 0 --gpu_id 0 --w_xai 25
# SimKD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/   --gpus all --cpuset-cpus="4" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/vgg19bn_vanilla_cifar100_trial_0/vgg19_bn_best.pth --distill simkd --model_s resnet18 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --w_xai 25