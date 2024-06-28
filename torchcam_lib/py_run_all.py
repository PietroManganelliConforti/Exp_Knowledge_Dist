import subprocess
from threading import Thread

def run_command(command):
    process = subprocess.Popen(command, shell=True)
    process.communicate()

def train_teacher(model_s, gpu_id):
    script_name = "/train_student_xai.py"
    command = f"""docker run -v /home/lorenzo/Research/XAI4KD/torchcam_lib:/work/project/ --gpus all \
    --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/{script_name} \
    --model_t resnet18 --distill kd --model_s {model_s} -c 1 -d 0 -b 0 --trial 0 --gpu_id {gpu_id} --batch_size 8 \
    --dataset imagenette --xai noXAI --w_xai 0"""
    run_command(command)

def train_teacher_debug(model_s, gpu_id):
    script_name = "/train_student_xai.py"
    command = f"""docker run -v /home/lorenzo/Research/XAI4KD/torchcam_lib:/work/project/ --gpus all \
    --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/{script_name} \
    --model_t resnet18 --distill kd --model_s {model_s} -c 1 -d 0 -b 0 --trial 0 --gpu_id {gpu_id} --batch_size 8 \
    --dataset imagenette --xai noXAI --w_xai 0 --debug"""
    run_command(command)

def run_all_distillations(model_t, model_s, xai, gpu_id):
    script_name = "train_student_xai.py"
    distillations = [
        ("kd", 0, 0),
        ("hint", 1, 100),
        ("attention", 1, 1000),
        ("similarity", 1, 3000),
        ("vid", 1, 1),
        ("semckd", 1, 400),
        ("srrl", 1, 1),
        ("simkd", 0, 1)
    ]
    
    for distill, c, b in distillations:
        command = f"""docker run -v /home/lorenzo/Research/XAI4KD/torchcam_lib:/work/project/ --gpus all  \
        --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/{script_name} \
        --model_t {model_t} --distill {distill} --model_s {model_s} -c {c} -d 1 -b {b} --trial 0 \
        --gpu_id {gpu_id} --batch_size 8 --dataset imagenette --xai {xai} --w_xai 150"""
        run_command(command)

def run_all_distillations_debug(model_t, model_s, xai, gpu_id):
    script_name = "train_student_xai.py"
    distillations = [
        ("kd", 0, 0),
        ("hint", 1, 100),
    ]
    
    for distill, c, b in distillations:
        command = f"""docker run -v /home/lorenzo/Research/XAI4KD/torchcam_lib:/work/project/ --gpus all --ipc host -u 1001:1001 piemmec/xai4kd_2:2 /usr/bin/python3 /work/project/{script_name} --model_t {model_t} --distill {distill} --model_s {model_s} -c {c} -d 1 -b {b} --trial 0 --gpu_id {gpu_id} --batch_size 8 --dataset imagenette --xai {xai} --w_xai 150 --debug"""
        run_command(command)

def run_sequentially_on_gpu_0():
    train_teacher("resnet110", 0)
    run_all_distillations("resnet32x4", "resnet8", "GradCAM", 0)
    run_all_distillations("resnet32x4", "resnet8", "noXAI", 0)
    train_teacher("wide_resnet10_2", 0)
    train_teacher("mobilenet_v2", 0)

def run_sequentially_on_gpu_1():
    train_teacher("resnet8", 1)
    run_all_distillations("resnet18", "resnet8", "GradCAM", 1)
    run_all_distillations("resnet18", "resnet8", "noXAI", 1)
    train_teacher("wide_resnet18_2", 1)
    train_teacher("resnet14x4", 1)

def run_sequentially_on_gpu_2():
    train_teacher("resnet116", 2)
    run_all_distillations("resnet18", "resnet8x4", "GradCAM", 2)
    run_all_distillations("resnet18", "resnet8x4", "noXAI", 2)
    train_teacher("wide_resnet34_2", 2)
    train_teacher("resnet38x4", 2)

def run_sequentially_on_gpu_3():
    train_teacher("resnet110x2", 3)
    run_all_distillations("resnet110x2", "resnet110", "GradCAM", 3)
    run_all_distillations("resnet110x2", "resnet110", "noXAI", 3)
    train_teacher("wide_resnet50_2", 3)
    train_teacher("wide_resnet101_2", 3)

def run_all():
    threads = [
        Thread(target=run_sequentially_on_gpu_0),
        Thread(target=run_sequentially_on_gpu_1),
        Thread(target=run_sequentially_on_gpu_2),
        Thread(target=run_sequentially_on_gpu_3)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    run_all()
