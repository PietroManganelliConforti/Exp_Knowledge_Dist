import json
from parser import parse_option

import os, re, time
import numpy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import ConvReg, SelfA, SRRL, SimKD

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader,  get_dataloader_sample
from dataset.imagenette import get_imagenette_dataloaders

from helper.loops import train_distill as train, validate_vanilla, validate_distill, train_distill_xai, train_distill_Noxai
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate

from crd.criterion import CRDLoss
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, VIDLoss, SemCKDLoss

split_symbol = '~' if os.name == 'nt' else ':'

def get_teacher_path(model_name):

    base_path = "/work/project/save_imagenette/teachers/"
    
    middle_path = "_vanilla_imagenette_trial_0/"

    end_path = "_best.pth"

    return base_path+model_name+middle_path+model_name+end_path




def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_t, n_cls):
    print('==> loading teacher model')
    #model_t = get_teacher_name(model_path)
    model_path = get_teacher_path(model_t)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0')['model'])
    print('==> done')
    return model


best_acc = 0
total_time = time.time()

def main():

    opt = parse_option()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    print(os.environ['CUDA_VISIBLE_DEVICES'],"\n\n")
    
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node

    gpu = None if ngpus_per_node > 1 else opt.gpu_id

    main_worker(gpu, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    torch.manual_seed(opt.seed)
    cudnn.deterministic = True # TODO -> il seed bloccato come sempre?
    cudnn.benchmark = False
    numpy.random.seed(opt.seed)
    os.environ["PYTHONHASHSEED"] = str(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    # Models loading
    n_cls = {
        'imagenette': 10, 
        'cifar100': 100,
        'imagenet': 1000,
    }.get(opt.dataset, None)
    
    model_t = load_teacher(opt.model_t, n_cls)
    
    # ---------------- STANDARD
    try:
        model_s = model_dict[opt.model_s](num_classes=n_cls)
        # model_t = model_dict[opt.model_s](num_classes=n_cls) # DA RIMUOVERE!!!
    except KeyError:
        print("This model is not supported. Da vedere models/__init__.py se contiene il modello")
        exit(0)
    # ---------------- POST TRAIN
    # model_s = load_teacher(opt.path_s, n_cls)



    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)
    elif opt.dataset == 'imagenette':
        data = torch.randn(2, 3, 224, 224)


    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    # Loss Functions
    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        print('\n Opt-distill:' + opt.distill)
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        print('\n Opt-distill:' + opt.distill)
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'attention':
        print('\n Opt-distill:' + opt.distill)
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        print('\n Opt-distill:' + opt.distill)
        criterion_kd = Similarity()
    elif opt.distill == 'vid':
        print('\n Opt-distill:' + opt.distill)
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'crd':
        print('\n Opt-distill:' + opt.distill)
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        if opt.dataset == 'cifar100':
            opt.n_data = 50000
        else:
            opt.n_data = 1281167
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'semckd':
        print('\n Opt-distill:' + opt.distill)
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = SemCKDLoss()
        self_attention = SelfA(opt.batch_size, s_n, t_n, opt.soft)    
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    elif opt.distill == 'srrl':
        print('\n Opt-distill:' + opt.distill)
        s_n = feat_s[-1].shape[1]
        t_n = feat_t[-1].shape[1]
        model_fmsr = SRRL(s_n= s_n, t_n=t_n)
        criterion_kd = nn.MSELoss()
        module_list.append(model_fmsr)
        trainable_list.append(model_fmsr)
    elif opt.distill == 'simkd':
        print('\n Opt-distill:' + opt.distill)
        s_n = feat_s[-2].shape[1]
        t_n = feat_t[-2].shape[1]
        model_simkd = SimKD(s_n= s_n, t_n=t_n, factor=opt.factor)
        criterion_kd = nn.MSELoss()
        module_list.append(model_simkd)
        trainable_list.append(model_simkd)
    else:
        raise NotImplementedError(opt.distill)
    
    # XAI Loss
    criterion_xai = nn.MSELoss() # L2 loss function
    # criterion_xai = nn.KLDivLoss()

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss
    criterion_list.append(criterion_xai)    # added XAI loss

    module_list.append(model_t)
    
    # Optimizer
    # optimizer = optim.SGD(trainable_list.parameters(),
    #                       lr=opt.learning_rate,
    #                       momentum=opt.momentum,
    #                       weight_decay=opt.weight_decay
    #                       )
    optimizer = optim.Adam(trainable_list.parameters(),
                           lr=opt.learning_rate)

    if torch.cuda.is_available():
        criterion_list.cuda()
        module_list.cuda()

    # Get dataset 
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size, num_workers=opt.num_workers, k=opt.nce_k, mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'imagenette':
        if opt.distill in ['crd']:
            print('Not Implemented yet') # train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size, num_workers=opt.num_workers, k=opt.nce_k, mode=opt.mode)
            exit(0)
        else:
            train_loader, val_loader = get_imagenette_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data, _, train_sampler = get_dataloader_sample(dataset=opt.dataset, batch_size=opt.batch_size,  num_workers=opt.num_workers, is_sample=True, k=opt.nce_k)
        else:
            train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)
    
    # validate teacher accuracy
    teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)
    
    # routine
    for epoch in range(1, opt.epochs + 1):

        torch.cuda.empty_cache()

        # adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")
        
        # Training
        time1 = time.time()

        if opt.xai != 'noXAI':
            train_acc, train_acc_top5, l_base, l_kd, l_xai = train_distill_xai(epoch, train_loader, module_list, criterion_list, optimizer, opt) # Si XAI
        else: 
            train_acc, train_acc_top5, l_base, l_kd, l_xai = train_distill_Noxai(epoch, train_loader, module_list, criterion_list, optimizer, opt) # No XAI
        
        time2 = time.time()
        print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, opt.gpu, train_acc, train_acc_top5, time2 - time1))

        train_merics = {'base_loss': l_base,
                        'kd_loss': l_kd,
                        'xai_loss': l_xai,
                        'trian_acc': train_acc,
                        'train_acc_top5': train_acc_top5}
        save_dict_to_json(train_merics, os.path.join(opt.save_folder, "train_best_metrics.json"))

        # Validation
        test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)        
        print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            
        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            if opt.distill == 'simkd':
                state['proj'] = trainable_list[-1].state_dict() 
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            
            test_merics = {'test_loss': test_loss,
                            'test_acc': test_acc,
                            'test_acc_top5': test_acc_top5,
                            'epoch': epoch}
            
            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))

            print('saving the best model!')
            torch.save(state, save_file)
            
    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_acc)
    
    # save parameters
    save_state = {k: v for k, v in opt._get_kwargs()}
    # No. parameters(M)
    num_params = (sum(p.numel() for p in model_s.parameters())/1000000.0)
    save_state['Total params'] = num_params
    save_state['Total time'] =  (time.time() - total_time)/3600.0
    params_json_path = os.path.join(opt.save_folder, "parameters.json") 
    save_dict_to_json(save_state, params_json_path)


    new_entry = str(test_merics['test_acc']) +" <--- "+ opt.save_folder + '\n'

    # Append the string to the file
    print(new_entry)
    with open('work/project/return_of_everything.txt', 'a') as file: 
        file.write(new_entry)
        file.close()




if __name__ == '__main__':

    #print(torch.cuda.is_available())  #da lasciare commentata e da non mettere prima di os.environ

    main()
