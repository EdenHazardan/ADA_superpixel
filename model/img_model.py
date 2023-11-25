import torch
import torch.nn as nn
import torch.nn.functional as F
from model.deeplabv2 import get_deeplab_v2 as deeplabv2


class deeplabv2_img_model(nn.Module):
    def __init__(self, num_classes=19, weight_res101=None, multi_level=False):
        super(deeplabv2_img_model, self).__init__()
        self.multi_level = multi_level
        self.segnet = deeplabv2(num_classes=num_classes, multi_level=self.multi_level)
        self.num_classes = num_classes
        self.weight_init(weight_res101)
        
        self.criterion_semantic = nn.CrossEntropyLoss(ignore_index=-1)

    def weight_init(self, weight_res101):
        weight = torch.load(weight_res101, map_location='cpu')

        new_params = self.segnet.state_dict().copy()
        for i in weight:
            i_parts = i.split('.')
            if not i_parts[1] == 'layer5' and not i_parts[1] == 'layer6' and not i_parts[1] == 'weight':
                new_params['.'.join(i_parts[1:])] = weight[i]
        self.segnet.load_state_dict(new_params, True)

        print('pretrained weight loaded')

    def forward(self, image, gt=None, hard_mine=False):
        if self.multi_level:
            pred_c_1, pred_c_2 = self.segnet(image)
            pred_c_1 = F.interpolate(pred_c_1, scale_factor=8, mode='bilinear', align_corners=False)
            pred_c_2 = F.interpolate(pred_c_2, scale_factor=8, mode='bilinear', align_corners=False)

            if gt is not None:
                loss_1 = self.criterion_semantic(pred_c_1, gt)
                loss_2 = self.criterion_semantic(pred_c_2, gt)
                loss_1 = loss_1.unsqueeze(0)
                loss_2 = loss_2.unsqueeze(0)
                loss = loss_1 + loss_2 * 0.1 
                return loss
            else:
                return pred_c_1

        else:
            pred_c = self.segnet(image)
            pred_c = F.interpolate(pred_c, scale_factor=8, mode='bilinear', align_corners=False)

            if gt is not None:
                loss = self.criterion_semantic(pred_c, gt)
                loss = loss.unsqueeze(0)
                # print("loss =",loss.shape)
                # print("gt =",loss.shape)
                return loss
            else:
                return pred_c
