# RIGA DI TESTING:
# docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t resnet18 --distill kd --model_s resnet8 -c 1 -d 1 -b 0 --trial 0 --gpu_id 0 --batch_size 8 --dataset imagenette --xai GradCAM  --w_xai 11 --debug

#nohup sh script_for_multiple_runs.sh  > /dev/null 2>&1 &

#ps aux | grep script and then kill, if needed
#kill dockers


train_teacher()
{
    local gpu_id=$2
    local script_name="/train_student_xai.py"


    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all \
        --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name \
        --model_t resnet18 --distill kd --model_s $1 -c 1 -d 0 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8\
        --dataset imagenette --xai noXAI --w_xai 0  
}



train_teacher_debug()
{
    local gpu_id=$2
    local script_name="/train_student_xai.py"


    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all \
        --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name \
        --model_t resnet18 --distill kd --model_s $1 -c 1 -d 0 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8\
        --dataset imagenette --xai noXAI --w_xai 0  --debug
    
}


find_xai_weight()
{
    local gpu_id=$4
    local script_name="train_student_xai.py"

    # W 
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3  --w_xai 10  
    
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3  --w_xai 50 

    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3  --w_xai 100 

    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3  --w_xai 150 
}



run_all_distillations()  
{
    # OSS 25/7 23:22 C'ERA W_XAI A 150 SOLO SU ALCUNI E ALTRI CON $5

    local gpu_id=$4
    local script_name="train_student_xai.py"

    # KD
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3  --w_xai $5  
    # FitNet
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    # AT
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    # SP
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial $0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    # VID
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    ## CRD
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5
    # SemCKD
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    # SRRL
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill srrl --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5
    # SimKD
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
}

run_all_distillations_mini(){

    local gpu_id=$4
    local script_name="train_student_xai.py"

    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial $0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 

}

run_all_distillations_butta(){  # funzione da cancellare

    local gpu_id=$4
    local script_name="train_student_xai.py"

    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial $0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 
    
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai $5 

}

run_all_distillations_debug()
{
    local gpu_id=$4
    local script_name="train_student_xai.py"

    # KD
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3  --w_xai 150  --debug
    #FitNet
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150 --debug
}




run_a_table_column()  
{
    # OSS 25/7 23:22 C'ERA W_XAI A 150 SOLO SU ALCUNI E ALTRI CON $5. $0 a trial? perchè?

    local gpu_id=$4
    local script_name="train_student_xai.py"

    local testa_e_butta=$0

    # KD
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai noXAI --w_xai 0  
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill kd --model_s $2 -c 1 -d 1 -b 0 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette --xai $3 --w_xai 150
     
    # FitNet
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0    
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/  --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill hint --model_s $2 -c 1 -d 1 -b 100 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    # AT
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0    
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill attention --model_s $2 -c 1 -d 1 -b 1000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    # SP
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill similarity --model_s $2 -c 1 -d 1 -b 3000 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    # VID
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill vid --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    ## CRD
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/train_student_xai.py --model_t $1 --distill crd --model_s $2 -c 1 -d 1 -b 0.8 --trial 0 --gpu_id $4 --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    # SemCKD
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill semckd --model_s $2 -c 1 -d 1 -b 400 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    # SRRL
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill srrl --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill srrl --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill srrl --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill srrl --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill srrl --model_s $2 -c 1 -d 1 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
    # SimKD
    # docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai noXAI --w_xai 0
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 10
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 50
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 100
    docker run -v /home/pietro/Research/XAI4KD/torchcam_lib:/work/project/   --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/$script_name --model_t $1 --distill simkd --model_s $2 -c 0 -d 0 -b 1 --trial 0 --gpu_id $gpu_id --batch_size 8 --dataset imagenette  --xai $3 --w_xai 150
}




#DOMANDA SUL TESTNAME TODO

# train_teacher resnet8 0 

# run_all_distillations resnet32x4 resnet8x4 GradCAM 0


#runall dist resnet18 resnet8 GradCAM gpu : 0 w_XAI: 50

run_sequentially_on_gpu_0()
{
    # run_a_table_column resnet56 vgg8 GradCAMpp 0 &
    # run_a_table_column vgg19bn vgg8 GradCAMpp 0 &
    run_a_table_column vgg8 resnet8 GradCAM 0

    #find_xai_weight resnet18 resnet8 GradCAM 0  

    # run_all_distillations_butta resnet18 resnet8 GradCAM 0 50

    # run_all_distillations resnet18 resnet8 GradCAM

    # train_teacher resnet8 0

    }

run_sequentially_on_gpu_1()
{

    run_a_table_column resnet56 vgg8 GradCAM 1


    # train_teacher resnet32x4 1

    # find_xai_weight resnet32x4 resnet8x4 GradCAM 1

    # run_all_distillations resnet32x4 resnet8x4 noXAI 1

    # train_teacher resnet8x4 1

    }

run_sequentially_on_gpu_2() 
{   

    run_a_table_column vgg19bn vgg8 GradCAM 2


    # find_xai_weight vgg8 resnet8 GradCAM 2

    # run_all_distillations_mini vgg8 resnet8 noXAI 2 0

    # run_all_distillations resnet32x4 resnet8x4 noXAI

    # train_teacher vgg19bn 2 ## CORRETTO CON vgg19bn -> RILANCIARE

    # find_xai_weight vgg19bn vgg8 GradCAM 2

    # run_all_distillations vgg19bn vgg8 noXAI 2

    # train_teacher vgg8 2

    }

run_sequentially_on_gpu_3()
{

    run_a_table_column vgg8 resnet8 GradCAM 3
    
    # train_teacher resnet56 3
    
    # find_xai_weight resnet56 vgg8 GradCAM 3

    # run_all_distillations resnet56 vgg8 noXAI 3

    }


# train_teacher_debug resnet32 0 


#OSS: 
# le gpu non vanno

#DA CONTROLLARE, UNO PER VOLTA decommentando la riga/le righe e poi lanciando questo file nel terminale:
# sh run_all_script.sh (fuori da docker, da ~/Research/XAI4KD)

# 1
#train_teacher_debug resnet32 3 #va fixato o il parser del train teacher o messi i param del train student xai

# 2
# run_sequentially_on_gpu_0 #a me va

#3 TUTTI INSIEME. NON PRINTANO SU TERMINALE, da vedere in base alle cartelle dei risultati

run_all()
{
    run_sequentially_on_gpu_0 
}





run_all 


# nohup sh   > /dev/null 2>&1 &
# nohup sh   > /dev/null 2>&1 &

#4 LANCIARE QUELLI VERI, SENZA DEBUG DENTRO, quindi run_sequentially_on_gpu_*

# Run functions in parallel
# run_sequentially_on_gpu_0 &
# run_sequentially_on_gpu_1 &
# run_sequentially_on_gpu_2 &
# run_sequentially_on_gpu_3 &

# # Wait for all background processes to finish
# wait