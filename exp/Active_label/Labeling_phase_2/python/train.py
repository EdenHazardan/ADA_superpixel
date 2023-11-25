import os
import numpy as np
import random
import argparse
from tqdm import trange
import logging
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import Counter
import pickle

from data.cityscapes import cityscapes_dataset_superpixel_AL_phase2
from model.img_model import deeplabv2_img_model
ignore_index = -1
label_name=["road", "sidewalk", "building", "wall", "fence", "pole", "light", "sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motocycle", "bicycle"]
NUM_CLASS = 19



def get_arguments():
    parser = argparse.ArgumentParser(description="Active Label")
    ###### general setting ######
    parser.add_argument("--exp_name", type=str, help="exp name")

    ###### training setting ######
    parser.add_argument("--model_name", type=str, help="name for the training model")
    parser.add_argument("--weight_res101", type=str, help="path to resnet18 pretrained weight")
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

def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])
    return cmap

def JensenShannonDivergence(p, q):
    M = (p + q)/2
    return 0.5 * torch.sum(p*torch.log(p/M)) + 0.5 * torch.sum(q*torch.log(q/M))


def superpixel_labeling_phase_two(Y1, gt, superpixel, label_info, uda_logit, tgt_logit, budget):
    """
    HUSP selective (domain differences) annotation

    Args:
        input: superpixel (H, W), Active Label Y1 (H, W), gt (H, W), uda_logit (C, H, W), tgt_logit (C, H, W)
        output: ground-truth (H, W)
    """
    #################################################################
    ### HUSP selective annotation according to domain differences ###
    #################################################################
    uda_pred_softmax = F.softmax(uda_logit, dim=1)
    tgt_pred_softmax = F.softmax(tgt_logit, dim=1)
    HUSP = label_info['HUSP']
    score_record = []
    ones = torch.ones_like(superpixel)
    zeros = torch.zeros_like(superpixel)
    for i in range(len(HUSP)):
        mask = superpixel == HUSP[i]
        mask1 = torch.where(mask, ones, zeros)
        uda_proto = (uda_pred_softmax * mask1).sum(2).sum(2)/mask1.sum() 
        tgt_proto = (tgt_pred_softmax * mask1).sum(2).sum(2)/mask1.sum() 
        DDscore = (1 - torch.cosine_similarity(uda_proto, tgt_proto, dim=1)).item() # compute domain difference score (DDscore)
        score_record.append(DDscore)

    score_record = np.array(torch.tensor(score_record, device='cpu'))
    up_index = np.argsort(score_record)
    up_index = up_index[::-1]

    HUSP_select_id =[]
    for i in range(0,len(HUSP)):
        HUSP_select_id.append(HUSP[up_index[i]]) ## descending order by domain differences

    ones=torch.ones_like(superpixel)
    cost = 0
    Y2 = Y1.clone() # initialize Y2
    
    ##### dominant class labeling for superpixel #####
    gt = gt.cpu().numpy()
    for uid in HUSP_select_id: ## select and annotate the HUSP with largest domain differences
        if cost<budget:
            mask = superpixel == uid
            result = Counter(gt[mask.cpu()]).most_common(1)
            dominant_class = torch.tensor(result[0][0]).cuda()
            Y2 = torch.where(mask, dominant_class, Y2)
            cost += 1
    return Y2




def train():

    args = get_arguments()
    print(args)

    random_seed = 0
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print('random seed:{}'.format(random_seed))

    # network
    UDA_base = deeplabv2_img_model(weight_res101=args.weight_res101).cuda()
    UDA_base_weight = torch.load('/home/gaoy/ADA_superpixel/pretrained/dacs.pth')  # we DACS as UDA_base 
    UDA_base.load_state_dict(UDA_base_weight, strict=False)
    UDA_base.eval()

    Target_base = deeplabv2_img_model(weight_res101=args.weight_res101).cuda()
    Target_base_weight = torch.load('/data/gaoy/ADA_superpixel/work_dirs/Target_base/best.pth')
    Target_base.load_state_dict(Target_base_weight, strict=False)
    Target_base.eval()

    test1_data = cityscapes_dataset_superpixel_AL_phase2(split='train',sp='SSN_city', label_save='superpixel_acitve_label', return_name=1)
    test1_loader = DataLoader(
        test1_data,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        drop_last=False,
        pin_memory=True
    )

    cmap = color_map('cityscapes')

    save_path = '/data/gaoy/ADA_superpixel/work_dirs/Active_label/superpixel_acitve_label'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('active labeling')

    real_cost=0
    all_cost=0

    test_loader_iter = iter(test1_loader)
    with torch.no_grad():
        i = 0
        for data in test_loader_iter:
            i += 1
            if i % 50 ==0 :
                print("test {}/{}".format(i,len(test1_loader)))
            image, Y1, gt, superpixel, label_info_file, img_name = data
            with open(label_info_file[0], "rb") as file:
                label_info = pickle.load(file)

            budget = int((160 - label_info['cost']))  # set total budget as 160
            uda_pred = UDA_base(image.cuda())
            tgt_pred = Target_base(image.cuda())

            ##### HUSP selective annotation #####
            Y2 = superpixel_labeling_phase_two(Y1.cuda(), gt.cuda(), superpixel.cuda(), label_info, uda_pred, tgt_pred, budget)

            ##### save the active label Y2 #####
            Y2 = Y2.cpu().numpy()
            Y2 = Y2.squeeze()
            Y2 = Image.fromarray(Y2.astype(np.uint8), mode='P')
            Y2.putpalette(cmap)

            filename = os.path.basename(img_name[0].split('/')[0])
            if not os.path.exists(os.path.join(save_path,filename)):
                os.mkdir(os.path.join(save_path,filename))
            Y2.save(os.path.join(save_path, img_name[0]+'.png'))

        print("real_cost = ",real_cost/len(test1_loader))
        logger.info("real_cost = {}".format(real_cost/len(test1_loader)))


if __name__ == '__main__':
    train()
