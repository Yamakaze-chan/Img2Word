from lib.DocTr.seg import U2NETP
from lib.DocTr.GeoTr import GeoTr

import torch
import torch.nn as nn
import torch.nn.functional as F
# import skimage.io as io
import numpy as np
import cv2
# import glob
# import os
from PIL import Image
# import argparse


def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        # print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def reload_segmodel(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cpu')
        # print(len(pretrained_dict.keys()))
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


class GeoTr_Seg(nn.Module):
    def __init__(self):
        super(GeoTr_Seg, self).__init__()
        self.msk = U2NETP(3, 1)
        self.GeoTr = GeoTr(num_attn_layers=6)

    def forward(self, x):
        msk, _1, _2, _3, _4, _5, _6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.GeoTr(x)
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm
    
def init_Geotr_model(GeoTr_msk_path, GeoTr_model_path):
    GeoTr_Seg_model = GeoTr_Seg()
    reload_segmodel(GeoTr_Seg_model.msk, GeoTr_msk_path)
    reload_model(GeoTr_Seg_model.GeoTr, GeoTr_model_path)
    return GeoTr_Seg_model

def process_image(input_image, model):
    model.eval()

    im_ori = np.array(input_image)[:, :, :3] / 255.
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288))
    im = im.transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)

    with torch.no_grad():
        bm = model(im)
        bm = bm.cpu()
        bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))
        bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))
        bm0 = cv2.blur(bm0, (3, 3))
        bm1 = cv2.blur(bm1, (3, 3))
        lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)

        out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
        img_geo = ((out[0] * 255).permute(1, 2, 0).numpy()).astype(np.uint8)
        return img_geo

# init_model = init_Geotr_model(r'./model_pretrained/seg.pth', r'./model_pretrained/geotr.pth')
# im = process_image(Image.open(r"C:\Users\ACER\Downloads\1575466545_5.jpg"), init_model)
# im.save("DocTr_rec.png")
# im.show()
