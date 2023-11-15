from __future__ import print_function, division

import sys
import time
import torch

from .util import AverageMeter, accuracy

import torch.nn as nn
from torch.nn import functional as F

from torch.optim.sgd import SGD
import copy

from .gs import gumbel_softmax

step = 0

class MetaSGD(SGD):
    def __init__(self, net, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net = net

    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):
            parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b
            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def dynamic_kd_loss(student_logits, teacher_logits, temperature=1.0):

    with torch.no_grad():
        student_probs = F.softmax(student_logits, dim=-1)
        student_entropy = - torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1) # student entropy, (bsz, )
        # normalized entropy score by student uncertainty:
        # i.e.,  entropy / entropy_upper_bound
        # higher uncertainty indicates the student is more confusing about this instance
        instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))

    input = F.log_softmax(student_logits / temperature, dim=-1)
    target = F.softmax(teacher_logits / temperature, dim=-1)
    batch_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
    weighted_kld = torch.mean(batch_loss * instance_weight)

    return weighted_kld


def train_distill(epoch, train_loader, meta_loader, module_list, criterion_list, optimizer,opt, meta_net, meta_optimizer):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_div_record = AverageMeter()
    loss_kd_record = AverageMeter()
    weight_record = AverageMeter()
    weight0_record = AverageMeter()
    weight1_record = AverageMeter()
    weight0_var = AverageMeter()
    weight1_var = AverageMeter()
    global step

    end = time.time()

    meta_dataloader_iter = iter(meta_loader)

    
    for idx, data in enumerate(train_loader):
        step += 1
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()


        if (idx + 1) % opt.meta_interval == 0:
            pseudo_net = opt.model_dict[opt.model_s](num_classes=opt.n_cls)
            pseudo_net.load_state_dict(model_s.state_dict())
            pseudo_net.cuda()
            pseudo_net.train()
        
            # ===================forward=====================
            preact = False
            if opt.distill in ['abound']:
                preact = True
            feat_s, logit_s = pseudo_net(input, is_feat=True, preact=preact)

            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

            if opt.distill == 'kd':
                loss_kd = 0
            elif opt.distill == 'hint':
                f_s = module_list[1](feat_s[opt.hint_layer]) # module_list[1] 是辅助网络, train_student 255行
                f_t = feat_t[opt.hint_layer]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'crd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            elif opt.distill == 'attention':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'nst':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'similarity':
                g_s = [feat_s[-2]]
                g_t = [feat_t[-2]]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'rkd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'pkt':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'kdsvd':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'correlation':
                f_s = module_list[1](feat_s[-1])
                f_t = module_list[2](feat_t[-1])
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'vid':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
                loss_kd = sum(loss_group)
            elif opt.distill == 'abound':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'fsp':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'factor':
                factor_s = module_list[1](feat_s[-2])
                factor_t = module_list[2](feat_t[-2], is_factor=True)
                loss_kd = criterion_kd(factor_s, factor_t)
            else:
                raise NotImplementedError(opt.distill)
            
            
            # num_labels = opt.n_cls
            # probs = F.softmax(logit_s, dim=-1)  # [bsz, cls of dataset]
            # entropy = torch.sum(probs * torch.log(probs), dim=1)  # [bsz]
            # avg_prob = 1 / num_labels * torch.ones((1, num_labels))  # [1, cls of dataset]，元素都为1 / num_labels
            # uncer = entropy / torch.sum(avg_prob * torch.log(avg_prob))   # [bs]
            
            
            #input_6
            pseudo_input_logits = torch.cat((logit_s,logit_t), 1)
            
            pseudo_weight = meta_net(pseudo_input_logits)  # [bsz, 2]
            
            # pseudo1
            pseudo_loss_vector  = opt.gamma * loss_cls + (opt.alpha* pseudo_weight[:,0]) * loss_div + (opt.beta* pseudo_weight[:,1]) * loss_kd #bs
            
            # pseudo_loss_vector  = uncer * (opt.gamma * loss_cls + (opt.alpha* pseudo_weight[:,0]) * loss_div + (opt.beta* pseudo_weight[:,1]) * loss_kd) #bs          

            pseudo_loss = torch.mean(pseudo_loss_vector)
            # pseudo_loss = pseudo_loss_vector.sum()

            pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
                        
            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
            delta = len(optimizer.state_dict()['param_groups'][0]['params']) - len(pseudo_optimizer.state_dict()['param_groups'][0]['params'])
            if delta >0:
                load_optim_state_dict = copy.deepcopy(optimizer.state_dict())
                load_optim_state_dict['param_groups'][0]['params'] = optimizer.state_dict()['param_groups'][0]['params'][:-delta]
            elif delta ==0:
                load_optim_state_dict = copy.deepcopy(optimizer.state_dict())

            
            pseudo_optimizer.load_state_dict(load_optim_state_dict)
            pseudo_optimizer.meta_step(pseudo_grads)

            del pseudo_grads

            # 计算 meta-net 的更新
            if opt.dataset == 'cifar100':
                try:
                    meta_inputs, meta_labels,_ = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_loader)
                    meta_inputs, meta_labels,_ = next(meta_dataloader_iter)
            else:
                try:
                    meta_inputs, meta_labels, _ = next(meta_dataloader_iter)
                except StopIteration:
                    meta_dataloader_iter = iter(meta_loader)
                    meta_inputs, meta_labels = next(meta_dataloader_iter)

            meta_inputs, meta_labels = meta_inputs.cuda(), meta_labels.cuda()
            meta_outputs = pseudo_net(meta_inputs)
            # with torch.no_grad():
            #     logit_t = model_t(meta_inputs, is_feat=False, preact=preact)
                
            
            # num_labels = opt.n_cls
            # probs = F.softmax(meta_outputs, dim=-1)  # [bsz, cls of dataset]
            # entropy = torch.sum(probs * torch.log(probs), dim=1)  # [bsz]
            # avg_prob = 1 / num_labels * torch.ones((1, num_labels))  # [1, cls of dataset]，元素都为1 / num_labels
            # uncer = entropy / torch.sum(avg_prob * torch.log(avg_prob))   # [bs]
            
            
            # t = int(opt.batch_size/opt.group_num)
            # loss_restrain = 0.0
            # for i in range(opt.group_num):
            #     loss = (abs(uncer[i*t:(i+1)*t] - uncer[i*t:(i+1)*t].mean())).mean()
                # loss_restrain += loss
            
            # loss_4
            incorrect_index = (torch.argmax(meta_outputs,dim=1)!=meta_labels).view(-1,1)
            meta_loss = (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float()) * (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float()) * incorrect_index.float()
            # meta_loss = (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float()) * (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float())
            # loss计算难度aware
            # meta_loss = meta_loss.mean(dim=1)
            # epsilon = torch.ones_like(meta_loss) * opt.epsilon
            # meta_loss = -meta_loss**opt.factor * torch.log(torch.max(epsilon, 1-meta_loss))
            
            meta_loss = meta_loss.mean()

            # all
            # incorrect_index = (torch.argmax(meta_outputs,dim=1)!=meta_labels).view(-1,1)
            # meta_loss = (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float()) * (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float())
            # meta_loss = meta_loss.mean()

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
        
        # print('#####4#####')
        # 正常 蒸馏 学生
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint': # fitnet
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity': # sp
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation': # cc
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor': # ft
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)
        
        # 搞 meta_net 的输入
        with torch.no_grad():
            pseudo_input_logits = torch.cat((logit_s,logit_t), 1)
            weight = meta_net(pseudo_input_logits)
            weight0 = weight[:,0]
            weight1 = weight[:,1]
            weight0_var_real = torch.var(weight[:,0])
            weight1_var_real = torch.var(weight[:,1])
        # uncer 
        # num_labels = opt.n_cls
        # probs = F.softmax(logit_s, dim=-1)  # [bsz, cls of dataset]
        # entropy = torch.sum(probs * torch.log(probs), dim=1)  # [bsz]
        # avg_prob = 1 / num_labels * torch.ones((1, num_labels))  # [1, cls of dataset]，元素都为1 / num_labels
        # uncer = entropy / torch.sum(avg_prob * torch.log(avg_prob))   # [bs]


        loss_vector  = opt.gamma * loss_cls + (opt.alpha * weight0) * loss_div + (opt.beta * weight1) * loss_kd
        # k1 = 1/(max(loss_vector)-min(loss_vector))
        # t1 = (k1 * (loss_vector-min(loss_vector)))

        # uncer计算难度aware
        # epsilon = torch.ones_like(loss_vector) * opt.epsilon
        # uncer_weight = -uncer**opt.factor * torch.log(torch.max(epsilon, 1-uncer))
        # loss_vector = loss_vector * uncer_weight
        # loss计算难度aware
        # loss_weight = -loss_vector**opt.factor * torch.log(torch.max(epsilon, 1-t1))

        loss = torch.mean(loss_vector)
        # loss_vector  = uncer * (opt.gamma * loss_cls + (opt.alpha * weight0) * loss_div + (opt.beta * weight1) * loss_kd)
        # loss = loss_vector.sum()

        loss_div_record.update(loss_div.mean(), input.size(0))
        loss_kd_record.update(loss_kd.mean(), input.size(0))
        weight0_record.update(weight0.mean(), input.size(0))
        weight1_record.update(weight1.mean(), input.size(0))
        weight0_var.update(weight0_var_real, input.size(0))
        weight1_var.update(weight1_var_real, input.size(0))
        weight_record.update(weight.mean(), input.size(0))
        

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()
        

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'loss_div {loss_div_record.val:.3f} ({loss_div_record.avg:.3f})\t'
                  'loss_kd {loss_kd_record.val:.3f} ({loss_kd_record.avg:.3f})\t'
                  'weight0 {weight0_record.val:.3f} ({weight0_record.avg:.3f})\t'
                  'weight1 {weight1_record.val:.3f} ({weight1_record.avg:.3f})\t'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, loss_div_record=loss_div_record, loss_kd_record=loss_kd_record, weight0_record= weight0_record,weight1_record=weight1_record))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, loss_div_record.avg, loss_kd_record.avg, weight0_record.avg, weight1_record.avg, weight0_var.avg, weight1_var.avg, step


def train_distill_time(epoch, train_loader, meta_loader, module_list, criterion_list, optimizer,opt, meta_net, meta_optimizer, weight_time_list, weight_real_list, now, last):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_div_record = AverageMeter()
    loss_kd_record = AverageMeter()
    weight_record = AverageMeter()
    weight0_record = AverageMeter()
    weight1_record = AverageMeter()
    weight0_var = AverageMeter()
    weight1_var = AverageMeter()
    global step

    end = time.time()

    meta_dataloader_iter = iter(meta_loader)

    
    for idx, data in enumerate(train_loader):
        step += 1
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()

            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()


        if (idx + 1) % opt.meta_interval == 0:
            pseudo_net = opt.model_dict[opt.model_s](num_classes=opt.n_cls)
            pseudo_net.load_state_dict(model_s.state_dict())
            pseudo_net.cuda()
            pseudo_net.train()
        
            # ===================forward=====================
            preact = False
            if opt.distill in ['abound']:
                preact = True
            feat_s, logit_s = pseudo_net(input, is_feat=True, preact=preact)

            with torch.no_grad():
                feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            loss_cls = criterion_cls(logit_s, target)
            loss_div = criterion_div(logit_s, logit_t)

            if opt.distill == 'kd':
                loss_kd = 0
            elif opt.distill == 'hint':
                f_s = module_list[1](feat_s[opt.hint_layer]) # module_list[1] 是辅助网络, train_student 255行
                f_t = feat_t[opt.hint_layer]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'crd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
            elif opt.distill == 'attention':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'nst':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'similarity':
                g_s = [feat_s[-2]]
                g_t = [feat_t[-2]]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'rkd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'pkt':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'kdsvd':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'correlation':
                f_s = module_list[1](feat_s[-1])
                f_t = module_list[2](feat_t[-1])
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'vid':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
                loss_kd = sum(loss_group)
            elif opt.distill == 'abound':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'fsp':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'factor':
                factor_s = module_list[1](feat_s[-2])
                factor_t = module_list[2](feat_t[-2], is_factor=True)
                loss_kd = criterion_kd(factor_s, factor_t)
            else:
                raise NotImplementedError(opt.distill)
            
            #input_6
            pseudo_input_logits = torch.cat((logit_s,logit_t), 1)
            
            pseudo_weight = meta_net(pseudo_input_logits)  # [bsz, 2]
            
            # if epoch == 1:
            #     weight_pseudo_list[index] = pseudo_weight.detach()
            # else:
            #     pseudo_weight = last * weight_pseudo_list[index] + now * pseudo_weight
            #     weight_pseudo_list[index] = pseudo_weight.detach()
            
            # pseudo1
            pseudo_loss_vector  = opt.gamma * loss_cls + (opt.alpha* pseudo_weight[:,0]) * loss_div + (opt.beta* pseudo_weight[:,1]) * loss_kd #bs           

            pseudo_loss = torch.mean(pseudo_loss_vector)

            pseudo_grads = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph=True)

            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                break
                        
            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
            delta = len(optimizer.state_dict()['param_groups'][0]['params']) - len(pseudo_optimizer.state_dict()['param_groups'][0]['params'])
            if delta >0:
                load_optim_state_dict = copy.deepcopy(optimizer.state_dict())
                load_optim_state_dict['param_groups'][0]['params'] = optimizer.state_dict()['param_groups'][0]['params'][:-delta]
            elif delta ==0:
                load_optim_state_dict = copy.deepcopy(optimizer.state_dict())

            
            pseudo_optimizer.load_state_dict(load_optim_state_dict)
            pseudo_optimizer.meta_step(pseudo_grads)

            del pseudo_grads

            # 计算 meta-net 的更新
            try:
                meta_inputs, meta_labels,_ = next(meta_dataloader_iter)
            except StopIteration:
                meta_dataloader_iter = iter(meta_loader)
                meta_inputs, meta_labels,_ = next(meta_dataloader_iter)

            meta_inputs, meta_labels = meta_inputs.cuda(), meta_labels.cuda()
            meta_outputs = pseudo_net(meta_inputs)
            
            # incorrect
            incorrect_index = (torch.argmax(meta_outputs,dim=1)!=meta_labels).view(-1,1)
            meta_loss = (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float()) * (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float()) * incorrect_index.float()
            meta_loss = meta_loss.mean()
            # all
            # meta_loss = (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float()) * (meta_outputs-F.one_hot(meta_labels,num_classes=meta_outputs.shape[1]).float())
            # meta_loss = meta_loss.mean()

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
        
        # print('#####4#####')
        # 正常 蒸馏 学生
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint': # fitnet
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity': # sp
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation': # cc
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor': # ft
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)
        
        # 搞 meta_net 的输入
        with torch.no_grad():
            pseudo_input_logits = torch.cat((logit_s,logit_t), 1)
            weight = meta_net(pseudo_input_logits)

            num_labels = opt.n_cls
            probs = F.softmax(logit_s, dim=-1)  # [bsz, cls of dataset]
            entropy = torch.sum(probs * torch.log(probs), dim=1)  # [bsz]
            avg_prob = 1 / num_labels * torch.ones((1, num_labels))  # [1, cls of dataset]，元素都为1 / num_labels
            uncer = entropy / torch.sum(avg_prob * torch.log(avg_prob))   # [bs]
            
            if epoch == 1:
                weight_real_list[index] = weight
            else:
                # 对于不确定度低的样本，认为学的已经够充分，线索权重不宜发生过大变化，所以对权重进行时序集成
                idx_batch = (uncer < opt.threshold).nonzero().view(-1)
                idx_1ist = index[idx_batch].view(-1)
                weight_uncer = last * weight_real_list[idx_1ist] + now * weight[idx_batch]
                weight_real_list[idx_1ist] = weight_uncer
                weight[idx_batch] = weight_uncer

            
            weight0 = weight[:,0]
            weight1 = weight[:,1]
            weight0_var_real = torch.var(weight[:,0])
            weight1_var_real = torch.var(weight[:,1])

        loss_vector  = opt.gamma * loss_cls + (opt.alpha * weight0) * loss_div + (opt.beta * weight1) * loss_kd
        loss = torch.mean(loss_vector)

        loss_div_record.update(loss_div.mean(), input.size(0))
        loss_kd_record.update(loss_kd.mean(), input.size(0))
        weight0_record.update(weight0.mean(), input.size(0))
        weight1_record.update(weight1.mean(), input.size(0))
        weight0_var.update(weight0_var_real, input.size(0))
        weight1_var.update(weight1_var_real, input.size(0))
        weight_record.update(weight.mean(), input.size(0))
        

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()
        

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'loss_div {loss_div_record.val:.3f} ({loss_div_record.avg:.3f})\t'
                  'loss_kd {loss_kd_record.val:.3f} ({loss_kd_record.avg:.3f})\t'
                  'weight0 {weight0_record.val:.3f} ({weight0_record.avg:.3f})\t'
                  'weight1 {weight1_record.val:.3f} ({weight1_record.avg:.3f})\t'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, loss_div_record=loss_div_record, loss_kd_record=loss_kd_record, weight0_record= weight0_record,weight1_record=weight1_record))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg, loss_div_record.avg, loss_kd_record.avg, weight0_record.avg, weight1_record.avg, weight0_var.avg, weight1_var.avg, step


def validate(val_loader, model, criterion, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        if opt.dataset == 'cifar100':
            for idx, (input, target) in enumerate(val_loader):
                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                # compute output
                output = model(input)
                loss = criterion(output, target).mean()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % opt.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            idx, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))
        else:
            for idx, data in enumerate(val_loader):
                input, target, _ = data
                input = input.float()
                if torch.cuda.is_available():
                    input = input.cuda()
                    target = target.cuda()

                # compute output
                output = model(input)
                loss = criterion(output, target).mean()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if idx % opt.print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            idx, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg
