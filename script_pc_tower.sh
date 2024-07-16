# RIGA DI TESTING:
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill kd --model_s resnet8 -c 1 -d 1 -b 0 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette --xai GradCAM  --w_xai 11 --debug

#nohup sh script_for_multiple_runs.sh  > /dev/null 2>&1 &

#ps aux | grep script and then kill, if needed
#kill dockers


script_tower_1()  
{
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t vgg8 --distill kd --model_s resnet8 -c 1 -d 1 -b 0 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette --xai noXAI --w_xai 0  
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill hint --model_s resnet8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 1 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 100
}

script_tower_2()  
{   
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill hint --model_s resnet8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 1 --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0    
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill hint --model_s resnet8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 1 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 10
}

script_tower_3()  
{    
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill hint --model_s resnet8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 1 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 150
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill hint --model_s resnet8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 1 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 50
}

script_tower_4()  
{
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t vgg19bn --distill hint --model_s vgg8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 2 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 150
    docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet56 --distill hint --model_s vgg8 -c 1 -d 1 -b 100 --trial 0 --gpu_id 3 --batch_size 8 --dataset imagenette  --xai GradCAM --w_xai 100
}






run_all()
{
    script_tower_1 &
    script_tower_2 &
    script_tower_3 & 
    script_tower_4 
}



run_all 


# nohup sh   > /dev/null 2>&1 &


# # KD
# # docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai noXAI --w_xai 0  
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 10
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 50
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 100
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 150
    
# # FitNet
# # docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0    
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
# # AT
# # docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0    
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
# # SP
# # docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
# # VID
# # docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
# docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
