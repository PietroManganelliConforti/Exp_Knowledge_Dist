# sample scripts for training vanilla teacher/student models

# CIFAR
#docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="0-7" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet8x4 --trial 0 --gpu_id 0

#docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="0-7" --cpuset-cpus="13" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet32x4 --trial 0 --gpu_id 0

#docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="0-7" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet110 --trial 0 --gpu_id 0

#docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="0-7" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet116 --trial 0 --gpu_id 0

#docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="0-7" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet110x2 --trial 0 --gpu_id 0

#docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="8-14" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model vgg8 --trial 0 --gpu_id 0

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="8-14" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model vgg13 --trial 0 --gpu_id 0

docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="8-14" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model ShuffleV1 --trial 0 --gpu_id 0

docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="8-14" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model ShuffleV2 --trial 0 --gpu_id 0

docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="8-14" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model ShuffleV2_1_5 --trial 0 --gpu_id 0

docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="15-23" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model MobileNetV2 --trial 0 --gpu_id 2

docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="15-23" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model MobileNetV2_1_0 --trial 0 --gpu_id 0

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="15-23" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet56 --trial 0 --gpu_id 2

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="15-23" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet18 --trial 0 --gpu_id 2

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="15-23" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet32 --trial 0 --gpu_id 2

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="24-31" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet44 --trial 0 --gpu_id 7

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="24-31" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model vgg19bn --trial 0 --gpu_id 2

docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="24-31" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model repvgg_a2 --trial 0 --gpu_id 0 ## --> NON VA

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="24-31" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model WResNet10x2 --trial 0 --gpu_id 7

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="24-31" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet38x2 --trial 0 --gpu_id 7

# docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --cpuset-cpus="8-15" --ipc host -u 1001:1001 piemmec/kd_xai /usr/bin/python3 /work/project/train_teacher.py --model resnet38x4 --trial 0 --gpu_id 7



