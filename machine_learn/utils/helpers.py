# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import torch
import matplotlib.pyplot as plt
import skimage
from skimage import io
from mpl_toolkits.mplot3d import Axes3D

from PIL import Image
from io import BytesIO 

import sys
def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)

    return x


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
       type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_cspace_views(cspace):
    cspace = cspace.squeeze().__ge__(0.9) # 次元削減＋要素の値を 0 or 1 に変換
    cspace = np.rot90(cspace, -1)#横軸x,縦軸y
    cspace = np.fliplr(cspace)#左右反転
    plt.gray()
    cspace =  1 - cspace #そのままだと、障害物：白、背景：黒なので白黒を反転させる（真っ白＝1, 真っ黒=0）
    plt.imshow(cspace)
    plt.axis([0, 360, 0, 360])
    #plt.axis([0, 72, 0, 72])
    plt.grid(True)
    plt.savefig('./hoge.jpg')
    # 画像ファイルパスから読み込み
    img = np.array(Image.open('./hoge.jpg'))
    return img

def get_probability_cspace_view(probability_cspace):
    probability_cspace = probability_cspace.squeeze()
    probability_cspace = np.rot90(probability_cspace, -1)#横軸x,縦軸y
    probability_cspace = np.fliplr(probability_cspace)#左右反転
    plt.gray()
    probability_cspace =  1 - probability_cspace #そのままだと、障害物：白、背景：黒なので白黒を反転させる （真っ白＝1, 真っ黒=0）
    plt.imshow(probability_cspace)
    plt.axis([0, 360, 0, 360])
    #plt.axis([0, 72, 0, 72])
    #plt.axis([0, 36, 0, 36])
    plt.grid(True)
    plt.savefig('./hoge.jpg')
    # 画像ファイルパスから読み込み
    img = np.array(Image.open('./hoge.jpg'))
    return img


def save_generated_cspace(sample_idx, epoch_idx, cspace):
    
    cspace = cspace.squeeze().__ge__(0.5)
    #np.save('./output/'+str(epoch_idx)+'epoch'+str(sample_idx)+'generated_cspace.npy', cspace)
    cspace = np.rot90(cspace, -1)#横軸x,縦軸y
    cspace = np.fliplr(cspace)#左右反転
    plt.gray()
    cspace =  1 - cspace #そのままだと、障害物：白、背景：黒なので白黒を反転させる（真っ白＝1, 真っ黒=0）
    plt.imshow(cspace)
    plt.axis([0, 360, 0, 360])
    #plt.axis([0, 72, 0, 72])
    #plt.axis([0, 36, 0, 36])
    plt.grid(True)
    plt.savefig('./output/'+str(epoch_idx)+'epoch'+str(sample_idx)+'generated_cspace.jpg')

def get_binarization_cspace(cspace):
    binarization_cspace = cspace.__ge__(0.5).float()
    return binarization_cspace

def save_ground_truth_cspace(sample_idx, epoch_idx, cspace):
    cspace = cspace.squeeze().__ge__(0.5)
    cspace = np.rot90(cspace, -1)#横軸x,縦軸y
    cspace = np.fliplr(cspace)#左右反転
    plt.gray()
    cspace =  1 - cspace #そのままだと、障害物：白、背景：黒なので白黒を反転させる（真っ白＝1, 真っ黒=0）
    plt.imshow(cspace)
    plt.axis([0, 360, 0, 360])
    #plt.axis([0, 72, 0, 72])
    #plt.axis([0, 36, 0, 36])
    plt.grid(True)
    plt.savefig('./output/'+str(epoch_idx)+'epoch'+str(sample_idx)+'ground_truth_cspace.jpg')

def save_probability_cspace(sample_idx, epoch_idx, probability_cspace):
    probability_cspace = probability_cspace.squeeze()
    probability_cspace = np.rot90(probability_cspace, -1)#横軸x,縦軸y
    probability_cspace = np.fliplr(probability_cspace)#左右反転
    plt.gray()
    probability_cspace =  1 - probability_cspace #そのままだと、障害物：白、背景：黒なので白黒を反転させる （真っ白＝1, 真っ黒=0）
    plt.imshow(probability_cspace)
    plt.axis([0, 360, 0, 360])
    #plt.axis([0, 72, 0, 72])
    #plt.axis([0, 36, 0, 36])
    plt.grid(True)
    plt.savefig('./output/'+str(epoch_idx)+'epoch'+str(sample_idx)+'probability_cspace.jpg')