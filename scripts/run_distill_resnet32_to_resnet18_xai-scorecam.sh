# sample scripts for running various knowledge distillation approaches
# we use resnet32x4 and resnet8x4 as an example

# CIFAR
# KD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill kd --model_s resnet18 --xai scorecam -c 1 -d 1 -b 0 --trial 0 --gpu_id 0
# FitNet
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill hint --model_s resnet18 --xai scorecam -c 1 -d 1 -b 100 --trial 0 --gpu_id 0
# AT
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill attention --model_s resnet18 --xai scorecam -c 1 -d 1 -b 1000 --trial 0 --gpu_id 0
# SP
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill similarity --model_s resnet18 --xai scorecam -c 1 -d 1 -b 3000 --trial 0 --gpu_id 0
# VID
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill vid --model_s resnet18 --xai scorecam -c 1 -d 1 -b 1 --trial 0 --gpu_id 0
# CRD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill crd --model_s resnet18 --xai scorecam -c 1 -d 1 -b 0.8 --trial 0 --gpu_id 0
# SemCKD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill semckd --model_s resnet18 --xai scorecam -c 1 -d 1 -b 400 --trial 0 --gpu_id 0
# SRRL
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill srrl --model_s resnet18 --xai scorecam -c 1 -d 1 -b 1 --trial 0 --gpu_id 0
# SimKD
docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_student_xai.py --path_t /work/project/save/teachers/models/resnet32_vanilla_cifar100_trial_0/resnet32_best.pth --distill simkd --model_s resnet18 --xai scorecam -c 0 -d 0 -b 1 --trial 0 --gpu_id 0

# ImageNets
# python train_student_xai.py --path_t './save/teachers/models/ResNet50_vanilla/ResNet50_best.pth' --batch_size 256  --epochs 120 --dataset imagenet --model_s resnet18 --xai scorecam --distill simkd -c 0 -d 0 -b 1 --learning_rate 0.1 --lr_decay_epochs 30,60,90 --weight_decay 1e-4 --num_workers 32 --gpu_id 0,1,2,3 --dist-url tcp://127.0.0.1:23344 --multiprocessing-distributed --dali gpu --trial 0 
