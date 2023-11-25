import os
import math
import numpy as np
from PIL import Image
from collections import Counter
import random
import argparse
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.cityscapes import cityscapes_dataset_superpixel_AL_phase1
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

def superpixel_labeling_phase_one(superpixel, label, logit):
    """
    Superpixel Fusion (Low-uncertainty Superpixel Fusion) & MLUSP priority annotation

    Args:
        input: superpixel (H, W), label (H, W), logit (C, H, W)
        output: Active Label Y1 (H, W)
    """
    ################################################
    ### Low-uncertainty Superpixel Fusion Module ###
    ################################################
    pred_softmax = F.softmax(logit, dim=1)
    p = pred_softmax.squeeze(dim=0)
    entropy_map = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(dim=0) / math.log(
        19)  # [1, 1, h, w]
    unique_id = torch.unique(superpixel)

    ##### divide superpixel into LUSP or HUSP #####
    HUSP = [] # high uncertainty superpixel: superpixel with high entropy
    LUSP = [] # low uncertainty superpixel: superpixel with low entropy
    LUSP_predictions = [] # the averaged class prediction f(s) of LUSP s
    ones = torch.ones_like(superpixel)
    zeros = torch.zeros_like(superpixel)
    for uid in unique_id:
        mask = superpixel == uid
        mask1 = torch.where(mask, ones, zeros)
        ent = entropy_map[mask]
        ent = ent.mean()
        if ent < 0.05: 
            LUSP.append(uid.item())
            avg_prediction = (pred_softmax * mask1).sum(2).sum(2)/mask1.sum() # calculate the average predictions of LUSP for the next step of fusion
            LUSP_predictions.append(avg_prediction)
        else:
            HUSP.append(uid.item())

    ##### LUSP fusion #####
    index = len(LUSP)
    new_id = len(unique_id) + 1 # assign new id for merged LUSP
    MLUSP = torch.ones_like(superpixel) * -1 # initialize merged LUSP (MLUSP)

    while index > 0:
        new_id += 1 
        root_sp_id = LUSP.pop(0)
        root_proto = LUSP_predictions.pop(0)
        index -= 1 
        # semantic_root_l = obtain_sp_label(root_sp_id,superpixel,pseudo)
        MLUSP = torch.where(superpixel==root_sp_id, new_id, MLUSP)
        delete_list = []
        for i in range(1,len(LUSP),1):
            current_proto = LUSP_predictions[i]
            score = JensenShannonDivergence(root_proto, current_proto)
            if score < 0.10: # merge two LUSPs when their JS divergence is lower than 0.10
                MLUSP = torch.where(superpixel==LUSP[i], new_id, MLUSP)
                delete_list.append(i)
                index -= 1 
        delete_list.reverse()
        for j in delete_list:
            LUSP.pop(j)
            LUSP_predictions.pop(j)


    #################################
    ### MLUSP priority annotation ###
    #################################
    label = label.cpu().numpy()
    MLUSP = MLUSP.cpu().numpy()
    active_label_Y1 = np.ones_like(label) * ignore_index

    ##### dominant class labeling for superpixel #####
    for uid in np.unique(MLUSP):
        if uid!=ignore_index:
            mask = MLUSP == uid

            result = Counter(label[mask]).most_common(1) 
            dominant_class = result[0][0]
            active_label_Y1 = np.where(mask, dominant_class, active_label_Y1) # only one click for a superpixel annotation

    cost = len(np.unique(MLUSP)) - 1 # record the annotation cost of labeling all MLUSPs
    print("cost = ",cost)
    return cost, HUSP, active_label_Y1


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
    UDA_merge = deeplabv2_img_model(weight_res101=args.weight_res101).cuda()
    UDA_merge_weight = torch.load('/home/gaoy/ADA_superpixel/pretrained/dacs.pth')  # we can use any UDA model as UDA-merge, and for simplicity, we use DACS here 
    UDA_merge.load_state_dict(UDA_merge_weight, strict=False)
    UDA_merge.eval()

    test1_data = cityscapes_dataset_superpixel_AL_phase1(split='train',sp='SSN_city', return_name=1)
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
        os.mkdir(save_path)

    print('active labeling')


    # test city
    test_loader_iter = iter(test1_loader)
    with torch.no_grad():
        i = 0
        for data in test_loader_iter:
            i += 1
            if i % 50 ==0 :
                print("test {}/{}".format(i,len(test1_loader)))
            image, gt, superpixel, img_name = data
            pred = UDA_merge(image.cuda())

            ##### Superpixel Fusion & MLUSP priority annotation #####
            cost, HUSP, Y1 = superpixel_labeling_phase_one(superpixel.cuda(), gt.cuda(), pred)

            ##### record the cost of labeling all MLUSPs and the remaining HUSPs #####
            sp_label_inf = {}
            sp_label_inf['cost'] = cost
            sp_label_inf['HUSP'] = HUSP
            print("HUSP = ",HUSP)
            filename = os.path.basename(img_name[0].split('/')[0])
            if not os.path.exists(os.path.join(save_path,filename)):
                os.mkdir(os.path.join(save_path,filename))
            with open(os.path.join(save_path, img_name[0]+'.pkl'), "wb") as file:
                pickle.dump(sp_label_inf, file)

            ##### save the active label Y1 #####
            Y1 = Y1.squeeze()
            Y1 = Image.fromarray(Y1.astype(np.uint8), mode='P')
            Y1.putpalette(cmap)
            Y1.save(os.path.join(save_path, img_name[0]+'.png'))

if __name__ == '__main__':
    train()
