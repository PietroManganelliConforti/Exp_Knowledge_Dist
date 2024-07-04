# RIGA DI TESTING:
# docker run -v $PWD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill kd --model_s resnet8 -c 1 -d 1 -b 0 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette --xai GradCAM  --w_xai 11 --debug

#nohup sh script_for_multiple_runs.sh  > /dev/null 2>&1 &

#ps aux | grep script and then kill, if needed
#kill dockers



run_a_table_column()  
{
    # OSS 25/7 23:22 C'ERA W_XAI A 150 SOLO SU ALCUNI E ALTRI CON $5. $0 a trial? perchè?

    local gpu_id=$4
    local script_name="train_student_xai.py"

    local testa_e_butta=$0

    ## CRD
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    
    # SemCKD
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    
    # SimKD
    # docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v $PWD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
}






run_all()
{
    run_a_table_column resnet18 resnet8 GradCAM 0 &
    run_a_table_column resnet32x4 resnet8x4 GradCAM 0 &
    run_a_table_column vgg19bn vgg8 GradCAM 0 &
    run_a_table_column resnet56 vgg8 GradCAM 0 &
    run_a_table_column vgg8 resnet8 GradCAM 0 
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
