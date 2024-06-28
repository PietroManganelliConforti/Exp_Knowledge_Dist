"""
Training a single model (student or teacher)
"""

import os, argparse, time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from models import model_dict
from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloader
from dataset.imagenette import get_imagenette_dataloaders
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate
from helper.loops import train_vanilla as train, validate_vanilla

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    # baisc
    parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--epochs', type=int, default=125, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='75,100,115', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='resnet32x4')
    parser.add_argument('--dataset', type=str, default='imagenette', choices=['cifar100', 'imagenet', 'imagenette'], help='dataset')
    parser.add_argument('-t', '--trial', type=str, default='0', help='the experiment id')
    
    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard 
    opt.model_path = '/work/project/save_imagenette/teachers'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    # set the model name
    opt.model_name = '{}_vanilla_{}_trial_{}'.format(opt.model, opt.dataset, opt.trial)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

best_acc = 0
total_time = time.time()
def main():
    opt = parse_option()
    
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)
    
    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    # Model
    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
        'imagenette': 10
    }.get(opt.dataset, None)
    
    try:
        model = model_dict[opt.model](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    # Optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # Loss Function
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()


    cudnn.benchmark = True

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset = opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, multiprocessing_distributed=False)
    elif opt.dataset == 'imagenette':
        train_loader, val_loader = get_imagenette_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers)
    else:
        raise NotImplementedError(opt.dataset)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_acc_top5, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()


        print(' * Epoch {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, train_acc, train_acc_top5, time2 - time1))

        test_acc, test_acc_top5, test_loss = validate_vanilla(val_loader, model, criterion, opt)
        print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))
            
        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model': model.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            
            test_merics = { 'test_loss': float('%.2f' % test_loss),
                            'test_acc': float('%.2f' % test_acc),
                            'test_acc_top5': float('%.2f' % test_acc_top5),
                            'epoch': epoch}
            
            save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
            
            print('saving the best model!')
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    print('best accuracy:', best_acc)

    # save parameters
    state = {k: v for k, v in opt._get_kwargs()}

    # No. parameters(M)
    num_params = (sum(p.numel() for p in model.parameters())/1000000.0)
    state['Total params'] = num_params
    state['Total time'] =  float('%.2f' % ((time.time() - total_time) / 3600.0))
    params_json_path = os.path.join(opt.save_folder, "parameters.json") 
    save_dict_to_json(state, params_json_path)
    
if __name__ == '__main__':
    main()
