from cProfile import label

import sys, time, torch
from .util import AverageMeter, accuracy, reduce_tensor
# from xai import return_cam_from_model
from torchcam import methods
import torch.nn.functional as F



def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        images, labels = batch_data
        
        if opt.gpu is not None:
            images = images.cuda(0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(0, non_blocking=True)

        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def get_target_layer(model_name):

    layer_name = ""  #Da lanciare resnet.py e vgg.py con -it per printare le reti listate

    if model_name == "resnet18":  #layer4 - layer4.conv1.conv2

        layer_name = "layer4"

    if model_name == "resnet32x4": #layer3 - layer3.4.conv2

        layer_name = "layer3"
    
    if model_name == "resnet32": #layer3 - layer3.4.conv2 #DA CONTROLLARE, MESSO A MANO

        layer_name = "layer3"

    if model_name == "resnet8x4": #layer3 - layer3. 0. conv 2 (prima del downsample)

        layer_name = "layer3"

    if model_name == "resnet8": #layer3 - layer3.0.conv2

        layer_name = "layer3"

    if model_name == "resnet110": #layer3 - layer3.17.conv2

        layer_name = "layer3"
    
    if model_name == "resnet110x2": #layer3 - layer3.17.conv2

        layer_name = "layer3"

    if model_name == "resnet44": #layer3 - layer3.6.conv2

        layer_name = "layer3"

    if model_name == "resnet56": #layer3 - layer3.8.conv2

        layer_name = "layer3"

    if model_name == "vgg8":   #block4.0

         layer_name = "block4"       

    if model_name == "vgg19bn":  #block4.6

         layer_name = "block4"  

    assert(layer_name!="")

    return layer_name

def train_distill_xai(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    criterion_xai = criterion_list[3]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses_tot = AverageMeter()
    losses_cls_div = AverageMeter()
    losses_kd = AverageMeter()
    losses_xai = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)
    end = time.time()
    for idx, data in enumerate(train_loader):

        if opt.distill in ['crd']:
            images, labels, index, contrast_idx = data
        else:
            images, labels = data
        
        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.clone().detach() for f in feat_t]  # Avoid in-place operations

        cls_t = model_t.get_feat_modules()[-1]
        
        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)
        
        # other kd loss
        if opt.distill == 'kd':
            loss_kd = torch.zeros(1).cuda(0, non_blocking=True)
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)                                                 
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        else:
            raise NotImplementedError(opt.distill) 

        # ----------------------- CUSTOM LOSS + CAM by PMC

        def cam_extractor_fn(model, extractor, img_tensor):
            extractor._hooks_enabled = True
            model.zero_grad()
            img_tensor = img_tensor.clone().detach().requires_grad_(True)
            scores = model(img_tensor)
            class_idx = torch.argmax(scores, dim=1).detach().cpu().tolist()
            activation_map = extractor(class_idx, scores)[0]
            return activation_map


        cam_name = opt.xai

        assert(cam_name != 'noXAI')

        target_layer_s = get_target_layer(opt.model_s)    
        target_layer_t = get_target_layer(opt.model_t)

        extractor_s = methods.__dict__[cam_name](model_s, target_layer=target_layer_s, enable_hooks=False)
        cam_s = cam_extractor_fn(model_s, extractor_s, images)
        extractor_s.remove_hooks()
        extractor_s._hooks_enabled = False

        extractor_t = methods.__dict__[cam_name](model_t, target_layer=target_layer_t, enable_hooks=False)
        cam_t = cam_extractor_fn(model_t, extractor_t, images)
        extractor_t.remove_hooks()
        extractor_t._hooks_enabled = False


        if cam_t.shape[-1] > cam_s.shape[-1]:
            if epoch == 0:
                print(cam_s.shape,cam_t.shape)
                print("[OSS] Stiamo interpolando per far matchare le shapes delle cam")
            cam_s = F.interpolate(cam_s.unsqueeze(1), size=(cam_t.shape[-2], cam_t.shape[-1]), mode='bilinear', align_corners=False).squeeze(1)

        elif cam_t.shape[-1] < cam_s.shape[-1]:
            if epoch == 0:
                print(cam_s.shape,cam_t.shape)
                print("[OSS] Stiamo interpolando per far matchare le shapes delle cam")
            cam_t = F.interpolate(cam_t.unsqueeze(1), size=(cam_s.shape[-2], cam_s.shape[-1]), mode='bilinear', align_corners=False).squeeze(1)


        loss_xai = criterion_xai(cam_s, cam_t)

        # Total loss   
        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd + opt.w_xai * loss_xai 

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        # ===================Losses=====================
        losses_tot.update(loss.item(), images.size(0))
        losses_cls_div.update((opt.cls * loss_cls + opt.div * loss_div).item(), images.size(0))
        losses_kd.update((opt.beta * loss_kd).item(), images.size(0))
        losses_xai.update((opt.w_xai * loss_xai).item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)     

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(epoch, idx, n_batch, opt.gpu, loss=losses_tot, top1=top1, top5=top5, batch_time=batch_time))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses_cls_div.avg, losses_kd.avg, losses_xai.avg


def train_distill_Noxai(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]
    criterion_xai = criterion_list[3]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses_tot = AverageMeter()
    losses_cls_div = AverageMeter()
    losses_kd = AverageMeter()
    losses_xai = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)
    end = time.time()
    for idx, data in enumerate(train_loader):

        if opt.distill in ['crd']:
            images, labels, index, contrast_idx = data
        else:
            images, labels = data
        
        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        cls_t = model_t.get_feat_modules()[-1]
        
        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)
        
        # other kd loss
        if opt.distill == 'kd':
            loss_kd = torch.zeros(1).cuda(0, non_blocking=True)
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)                                                 
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        else:
            raise NotImplementedError(opt.distill) 

        loss_xai = torch.zeros(1)

        # Total loss   
        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd # Standard LOSS
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()   

        # ===================Losses=====================
        losses_tot.update(loss.item(), images.size(0))
        losses_cls_div.update((opt.cls * loss_cls + opt.div * loss_div).item(), images.size(0))
        losses_kd.update((opt.beta * loss_kd).item(), images.size(0))
        losses_xai.update((opt.w_xai * loss_xai).item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)     

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(epoch, idx, n_batch, opt.gpu, loss=losses_tot, top1=top1, top5=top5, batch_time=batch_time))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses_cls_div.avg, losses_kd.avg, losses_xai.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)
    end = time.time()
    for idx, data in enumerate(train_loader):

        if opt.distill in ['crd']:
            images, labels, index, contrast_idx = data
        else:
            images, labels = data
        
        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        cls_t = model_t.get_feat_modules()[-1]
        
        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)
        
        # other kd loss
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)                                                 
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'simkd':
            trans_feat_s, trans_feat_t, pred_feat_s = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            logit_s = pred_feat_s
            loss_kd = criterion_kd(trans_feat_s, trans_feat_t)
        else:
            raise NotImplementedError(opt.distill)
        
        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()        

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(epoch, idx, n_batch, opt.gpu, loss=losses, top1=top1, top5=top5, batch_time=batch_time))
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            images, labels = batch_data
        
            if opt.gpu is not None:
                images = images.cuda(0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def validate_distill(val_loader, module_list, criterion, opt):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    
    model_s = module_list[0]
    model_t = module_list[-1]
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            images, labels = batch_data

            if opt.gpu is not None:
                images = images.cuda(0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(0, non_blocking=True)

            # compute output
            if opt.distill == 'simkd':
                feat_s, _ = model_s(images, is_feat=True)
                feat_t, _ = model_t(images, is_feat=True)
                feat_t = [f.detach() for f in feat_t]
                cls_t = model_t.get_feat_modules()[-1]
                _, _, output = module_list[1](feat_s[-2], feat_t[-2], cls_t)
            else:
                output = model_s(images)

            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)
            
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
