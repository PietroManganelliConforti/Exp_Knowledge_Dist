run_paolo(){
docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill hint --model_s vgg8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAMpp --w_xai 150
docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill hint --model_s vgg8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAMpp --w_xai 100
docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill hint --model_s vgg8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAMpp --w_xai 50
docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill hint --model_s vgg8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAMpp --w_xai 10
docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill hint --model_s vgg8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
}




run_paolo 