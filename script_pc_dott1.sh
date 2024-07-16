script_dottorandi1_1(){
    
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill simkd --model_s resnet8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 10
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill simkd --model_s resnet8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 50
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill simkd --model_s resnet8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 100

}  

script_dottorandi1_2(){

    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill similarity --model_s vgg8 -c 1 -d 1 -b 3000 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 50
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill similarity --model_s vgg8 -c 1 -d 1 -b 3000 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 150    
    # riga 4 SimKD
    # docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill simkd --model_s resnet8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 10
    # docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill simkd --model_s resnet8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 50
    # docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill simkd --model_s resnet8 -c 0 -d 0 -b 1 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 100

}



run_all()
{
    script_dottorandi1_1 &
    script_dottorandi1_2
}



run_all 