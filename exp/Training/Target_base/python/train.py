import os
import sys
import time
import numpy as np
import random
import argparse
import ast
from tqdm import trange
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as ttransforms

from tensorboardX import SummaryWriter

from data.gta5 import gta5_dataset, IterLoader
from data.cityscapes import cityscapes_dataset_test, cityscapes_dataset_training, IterLoader
from model.img_model import deeplabv2_img_model
from utils.metrics import runningScore
from utils.utils import dynamic_copy_paste
from utils.augmentations import get_augmentation, augment

ignore_index = -1
label_name=["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"]
NUM_CLASS = 19

all_class = [i for i in range(NUM_CLASS)]
easy_class = [0, 1, 2, 8, 10]
hard_class = [3, 6, 7, 12, 14, 15, 16, 17, 18]
target_cat = list(set(all_class)-set(easy_class))

def get_arguments():
    parser = argparse.ArgumentParser(description="Train SCNet")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--weight_res101", type=str, help="path to resnet18 pretrained weight")
    parser.add_argument("--al_SDA", action="store_true", help="use al strong data augmantation or not.")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--source_batch_size", type=int)
    parser.add_argument("--target_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--train_num_workers", type=int)
    parser.add_argument("--test_num_workers", type=int)
    parser.add_argument("--train_iterations", type=int)
    parser.add_argument("--early_stop", type=int)
    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--val_interval", type=int)
    parser.add_argument("--work_dirs", type=str)

    return parser.parse_args()

def train():


    args = get_arguments()
    print(args)
    if not os.path.exists(os.path.join(args.work_dirs, args.exp_name)):
        os.makedirs(os.path.join(args.work_dirs, args.exp_name))
    tblogger = SummaryWriter(os.path.join(args.work_dirs, args.exp_name))

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    fh = logging.FileHandler(os.path.join(args.work_dirs, args.exp_name, '{}.log'.format(rq)), mode='w')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('random seed:{}'.format(random_seed))

    # network
    net = deeplabv2_img_model(weight_res101=args.weight_res101).cuda()

    # optimizer
    params = net.segnet.optim_parameters(lr=args.lr)
    optimizer = optim.SGD(params=params, lr=args.lr, weight_decay=0.0005, momentum=0.9)

    # dataset
    source_data = gta5_dataset(split='train')
    source_loader = DataLoader(
        source_data,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    source_loader = IterLoader(source_loader)

    label_data = cityscapes_dataset_training(split='train', sp_method='superpixel_acitve_label')
    label_loader = DataLoader(
        label_data,
        batch_size=args.source_batch_size,
        shuffle=True,
        num_workers=args.train_num_workers,
        drop_last=True,
        pin_memory=True
    )
    label_loader = IterLoader(label_loader)


    print("label data:", len(label_loader))
    # print("unlabeled data:", len(unlabel_loader))

    test1_data = cityscapes_dataset_test(split='val')
    test1_loader = DataLoader(
        test1_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        drop_last=False,
        pin_memory=True
    )

    miou_cal = runningScore(n_classes=NUM_CLASS)
    if args.al_SDA:
        augmentations = get_augmentation()
    best_city_miou = 0.0

    for step in trange(args.early_stop):
        net.train()

        # reset optimizers
        optimizer.zero_grad()

        # adapt LR if needed
        lr = poly_lr_scheduler(args, optimizer, iter=step)
        
        # init loss
        total_loss = 0

        # train on source
        image_s, gt_s = next(source_loader)

        # train on target
        image_t, _, gt_t = next(label_loader)

        gt_s = gt_s.cuda()
        gt_t = gt_t.cuda()

        loss = net(image_t.cuda(), gt=gt_t.cuda())
        ce_loss = loss.mean()
        ce_loss.backward()


        # train on unlabeled target
        with torch.no_grad():
            # Run non-augmented image though model to get predictions
            image_u, _, _ = next(label_loader)
            
            logits_u = net(image_u.cuda())
            pred = F.softmax(logits_u.detach(), dim=1)
            max_probs, pseudo_label = torch.max(pred.detach(), dim=1)

        query_class0 = []
        query_class1 = []
        num_class_gt0 = torch.unique(gt_s[0])
        num_class_gt0 = num_class_gt0[num_class_gt0!=ignore_index]
        len_gt0 = len(num_class_gt0)
        num_class_gt1 = torch.unique(gt_s[1])
        num_class_gt1 = num_class_gt1[num_class_gt1!=ignore_index]
        len_gt1 = len(num_class_gt1)

        class0= np.random.choice(num_class_gt0.cpu(), int((len_gt0+len_gt0%2)/2), replace=False)
        for k in range(len(class0)):
            query_class0.append(class0[k])
            
        class1 = np.random.choice(num_class_gt1.cpu(), int((len_gt1+len_gt1%2)/2), replace=False)
        for k in range(len(class1)):
            query_class1.append(class1[k])

        image_mix1, label_mix1, _ = dynamic_copy_paste(image_u[0].cuda(), pseudo_label[0].long(), image_s[0].cuda(), gt_s[0].long(), query_class0, return_mask=True) 
        image_mix2, label_mix2, _ = dynamic_copy_paste(image_u[1].cuda(), pseudo_label[1].long(), image_s[1].cuda(), gt_s[1].long(), query_class1, return_mask=True) 

        image_mix = torch.cat((image_mix1, image_mix2))
        label_mix = torch.cat((label_mix1, label_mix2)).long()

        if args.al_SDA:
            image_mix, label_mix = augment(image_mix.cpu(), label_mix.detach().cpu(), augmentations)
            label_mix = label_mix.cuda()

        loss = net(image_mix.cuda(), gt=label_mix.cuda())
        domain_mix_loss = loss.mean()

        domain_mix_loss.backward()



        total_loss = ce_loss + domain_mix_loss 



        # total_loss.backward()
        optimizer.step()

        if (step + 1) % args.log_interval == 0:

            print('iter:{}/{} lr:{:.6f} ce_loss:{:.6f} domain_mix_loss:{:.6f} total_loss:{:.6f}'.format(step + 1, args.train_iterations, lr, ce_loss.item(), domain_mix_loss.item(), total_loss.item()))
            logger.info('iter:{}/{} lr:{:.6f} ce_loss:{:.6f} domain_mix_loss:{:.6f} total_loss:{:.6f}'.format(step + 1, args.train_iterations, lr, ce_loss.item(), domain_mix_loss.item(), total_loss.item()))
            tblogger.add_scalar('lr', lr, step + 1)
            tblogger.add_scalar('ce_loss', ce_loss.item(), step + 1)
            tblogger.add_scalar('domain_mix_loss', domain_mix_loss.item(), step + 1)
            tblogger.add_scalar('total_loss', total_loss.item(), step + 1)


        if (step + 1) % args.val_interval == 0:
            print('begin validation')
            logger.info('begin validation')

            net.eval()

            # test city
            test_loader_iter = iter(test1_loader)
            with torch.no_grad():
                i = 0
                for data in test_loader_iter:
                    i += 1
                    if i % 50 ==0 :
                        print("test {}/{}".format(i,len(test1_loader)))
                    image, gt = data
                    pred = net(image.cuda())
                    out = torch.argmax(pred, dim=1)
                    out = out.squeeze().cpu().numpy()
                    gt = gt.squeeze().cpu().numpy()
                    miou_cal.update(gt, out)
                miou, miou_class = miou_cal.get_scores(return_class=True)
                miou_cal.reset()

            tblogger.add_scalar('miou_city/miou', miou, step + 1)
            for i in range(int(NUM_CLASS)):
                tblogger.add_scalar('miou_city/{}_miou'.format(label_name[i]), miou_class[i], step + 1)
            if miou > best_city_miou:
                best_city_miou = miou
                save_path = os.path.join(args.work_dirs, args.exp_name, 'best_city.pth')
                torch.save(net.state_dict(), save_path)
            print('step:{} current city miou:{:.4f} best city miou:{:.4f}'.format(step + 1, miou, best_city_miou))
            logger.info('step:{} current city miou:{:.4f} best city miou:{:.4f}'.format(step + 1, miou, best_city_miou))
            print('step:{} current city miou{}'.format(step + 1, miou_class))
            logger.info('step:{} current city miou{}'.format(step + 1, miou_class))

def poly_lr_scheduler(args, optimizer, init_lr=None, iter=None, max_iter=None, power=None):
    init_lr = args.lr
    max_iter = args.train_iterations
    power = 0.9
    new_lr = init_lr * (1 - float(iter) / max_iter) ** power
    optimizer.param_groups[0]["lr"] = new_lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]["lr"] = 10 * new_lr
    return new_lr

if __name__ == '__main__':
    train()
