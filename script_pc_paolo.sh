#!/bin/bash

docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t vgg19bn --distill simkd --model_s vgg8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 150
docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill simkd --model_s vgg8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 10
docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill simkd --model_s vgg8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 50
docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill simkd --model_s vgg8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 100
docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill simkd --model_s vgg8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 150

