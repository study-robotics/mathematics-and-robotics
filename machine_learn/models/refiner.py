# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool2d(kernel_size=2), # 360->180
            torch.nn.Dropout3d(p=0.5),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool2d(kernel_size=2), # 180 -> 90
            torch.nn.Dropout3d(p=0.5),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool2d(kernel_size=2), # 90 -> 45
            torch.nn.Dropout3d(p=0.5),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=6, padding=3, stride=3), #45 -> 16
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool2d(kernel_size=2), # 16 -> 8
            torch.nn.Dropout3d(p=0.5),
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 512, kernel_size=3, padding=1), 
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),
            torch.nn.MaxPool2d(kernel_size=2), # 8 -> 4
            torch.nn.Dropout3d(p=0.5),
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU()
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
            torch.nn.ReLU()
        )
        self.layer9 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),# 4 -> 8
            torch.nn.BatchNorm2d(128), 
            torch.nn.ReLU()
        )
        self.layer10 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),# 8 -> 16
            torch.nn.BatchNorm2d(64), 
            torch.nn.ReLU()
        )
        self.layer11 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=4, stride=3, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=2), # 16->45
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.layer12 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1), # 45->90
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.layer13 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),# 90->180
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU()
        )
        self.layer14 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),# 180->360
            torch.nn.Dropout3d(p=0.5),
            torch.nn.Sigmoid()
        )

    def forward(self, coars_cspaces):
        cspace_360_l = coars_cspaces.unsqueeze(dim=1)
        cspace_180_l = self.layer1(cspace_360_l)
        cspace_90_l = self.layer2(cspace_180_l)
        cspace_45_l = self.layer3(cspace_90_l)
        cspace_16_l = self.layer4(cspace_45_l)
        cspace_8_l = self.layer5(cspace_16_l)
        cspace_4_l = self.layer6(cspace_8_l)
        flatten_features = self.layer7(cspace_4_l.view(-1, 8192))
        flatten_features = self.layer8(flatten_features)
        cspace_4_r = cspace_4_l + flatten_features.view(-1, 512, 4, 4)
        cspace_8_r = cspace_8_l + self.layer9(cspace_4_r)
        cspace_16_r = cspace_16_l + self.layer10(cspace_8_r)
        #print("cspace_45l", cspace_45_l.size())
        #print("cspace_16r", cspace_16_r.size())
        cspace_45_r = cspace_45_l + self.layer11(cspace_16_r)
        cspace_90_r = cspace_90_l + self.layer12(cspace_45_r)
        cspace_180_r = cspace_180_l + self.layer13(cspace_90_r)
        cspace_360_r = (cspace_360_l + self.layer14(cspace_180_l))*0.5
        return cspace_360_r.squeeze(dim=1)
        

        
        """
        volumes_32_l = coarse_volumes.unsqueeze(dim=1)
        # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 360, 360])
        volumes_16_l = self.layer1(volumes_32_l)
        # print(volumes_16_l.size())       # torch.Size([batch_size, 32, 180, 180])
        volumes_8_l = self.layer2(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 64, 90, 90])
        volumes_4_l = self.layer3(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 128, 45, 45])
        flatten_features = self.layer4(volumes_4_l.view(-1, 259200))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 8192])
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 45, 45)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 128, 45, 45])
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 64, 90, 90])
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 32, 180, 180])
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # torch.Size([batch_size, 1, 360, 360])
        return volumes_32_r.squeeze(dim=1)
        """
