
import argparse,os

import torch

def parse_option():

    split_symbol = '~' if os.name == 'nt' else ':'

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='-1', help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--seed', type=int, default=12345, help='insert seed for randomicity')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    # parser.add_argument('--lr_decay_epochs', type=str, default='75, 100, 115', help='where to decay lr, can be a list') # 75, 100, 115
    # parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    # parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    # parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'imagenette'], help='dataset')
    parser.add_argument('--model_s', type=str, default='resnet8x4')
    parser.add_argument('--model_t', type=str, default=None, help='teacher model snapshot')
    # ---------------- POST TRAIN
    parser.add_argument('--path_s', type=str, default=None, help='studentKD model snapshot')
    

    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 'vid',
                                                                      'crd', 'semckd','srrl', 'simkd'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('-f', '--factor', type=int, default=2, help='factor size of SimKD')
    parser.add_argument('-s', '--soft', type=float, default=1.0, help='attention scale of SemCKD')

    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # XAI
    parser.add_argument('--xai', default='noXAI', required=True, choices=['noXAI', 'CAM' , 'GradCAM', 'GradCAMpp', 'SmoothGradCAMpp', 'ScoreCAM', 'SSCAM', 'ISCAM' ,'xGradCAM', 'LayerCAM' ])
    parser.add_argument('--w_xai', type=float, default=1.0, help='weight balance for XAI loss')

    parser.add_argument('--debug', action='store_true', help="debug flag")

    
    opt = parser.parse_args()


    if opt.debug: opt.epochs = 1
    
    torch.autograd.set_detect_anomaly(True)

    # set different learning rates for these MobileNet/ShuffleNet models
    # if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
    #     opt.learning_rate = 0.01

    # set the path of model and tensorboard

    if opt.debug:
        opt.model_path = '/work/project/save_imagenette/debug_butta/'#XAI_noGradinLib/'
    elif opt.div==0 and opt.beta==0:
        opt.model_path = '/work/project/save_imagenette/teachers/' 
    elif opt.xai == 'noXAI':
        opt.model_path = '/work/project/save_imagenette/student_noXAI_noGradinLib/'
    else:
        opt.model_path = '/work/project/save_imagenette/student_XAI_noGradinLib/'

    # iterations = opt.lr_decay_epochs.split(',')
    # opt.lr_decay_epochs = list([])
    # for it in iterations:
    #     opt.lr_decay_epochs.append(int(it))

    #opt.model_t =  get_teacher_name(opt.path_t)
    # ---------------- POST TRAIN
    # opt.model_s = get_teacher_name(opt.path_s)


    if opt.div==0 and opt.beta==0:
        opt.model_name = '{}_vanilla_{}_trial_{}'.format(opt.model_s, opt.dataset, opt.trial)
    else:
        model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_r', '{}_a', '{}_b', '{}_xai', '{}_Wxai', '{}_seed', '{}_{}'])
        opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, opt.distill, opt.cls, opt.div, opt.beta, opt.xai, opt.w_xai, opt.seed, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt