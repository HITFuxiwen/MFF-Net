import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist

from dataset import get_loader
import math
from ImageDepthNet5 import ImageDepthNet5
from ImageDepthNet6 import ImageDepthNet6
from ImageDepthNet7 import ImageDepthNet7
import os
import Testing
import eva
import pytorch_iou
from utils import clip_gradient, adjust_lr
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):
    # mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))
    num_gpus = 1
    main(1,num_gpus,args)

def main(local_rank, num_gpus, args):

    net = ImageDepthNet7(args)

    net.cuda()
    # model_path = args.save_model_dir + '163RGB_VST_rcmt_d2.pth'
    # model_path = args.save_model_dir + '121RGB_VST_rcmt_d40.pth'
    # args.lr = args.lr * 0.1
    # state_dict = torch.load(model_path)
    # # load params
    # net.load_state_dict(state_dict)
    net.train()

    # del state_dict

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]

    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])

    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')

    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1,
                                               # pin_memory=True,
                                               # sampler=sampler,
                                               drop_last=True,
                                               )

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    loss_weights = [1, 0.8, 0.8, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    IOU = pytorch_iou.IOU(size_average = True)

    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    # for epoch in range(119, args.epochs):
    for epoch in range(args.epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch + 1, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, label_224, label_14, label_28, label_56, label_112 = data_batch

            images, label_224 = Variable(images.cuda(non_blocking=True)), \
                                        Variable(label_224.cuda(non_blocking=True))

            label_14, label_28, label_56, label_112 = Variable(label_14.cuda()), Variable(label_28.cuda()),\
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())

            outputs_saliency = net(images)

            mask_1_16, mask_1_8, mask_1_4, mask_1_1, m_16, m_8, m_4, m_1 = outputs_saliency

            loss5 = criterion(mask_1_16, label_14)
            loss4 = criterion(mask_1_8, label_28)
            loss3 = criterion(mask_1_4, label_56)
            loss1 = criterion(mask_1_1, label_224)

            loss5_2 = IOU(m_16, label_14)
            loss4_2 = IOU(m_8, label_28)
            loss3_2 = IOU(m_4, label_56)
            loss1_2 = IOU(m_1, label_224)

            img_total_loss = loss_weights[0] * loss1 + loss_weights[2] * loss3 + loss_weights[3] * loss4 + loss_weights[4] * loss5 
            IOU_loss = loss_weights[0] * loss1_2 + loss_weights[2] * loss3_2 + loss_weights[3] * loss4_2 + loss_weights[4] * loss5_2 

            total_loss = img_total_loss + IOU_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += loss1.cpu().data.item()
            if whole_iter_num % 40 == 0:
                print(
                    'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- img_loss: {3:.6f}  --- saliency loss: {4:.6f} --- IOU_loss:{5:.6f}'.format(
                        (whole_iter_num + 1),
                        (i + 1) * args.batch_size / N_train, total_loss.item(), img_total_loss.item(), loss1.item(), IOU_loss.item() ))


            optimizer.zero_grad()

            total_loss.backward()

            clip_gradient(optimizer, 0.5)

            optimizer.step()
            whole_iter_num += 1

            
            if whole_iter_num == args.train_steps:
                return 0

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)                                                                                                         
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        if  epoch%5 == 1 and epoch > 20:
            # torch.save(net.state_dict(), args.save_model_dir + str(epoch) + 'RGB_VST_rcmt_d22.pth')
            torch.save(net.state_dict(), args.save_model_dir + str(epoch) + 'RGB_VST_rcmt_newswam4.pth')
            path = args.save_model_dir + str(epoch) + 'RGB_VST_rcmt_newswam4.pth'
            Testing.test_net(args, path)
            eva.evaluate(args, True)   

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss/iter_num, epoch+1)




