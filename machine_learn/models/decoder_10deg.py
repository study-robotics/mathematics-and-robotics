# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch


class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(2048, 512, kernel_size=5, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1), # 4 -> 9
            torch.nn.InstanceNorm2d(512), 
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),# 9 -> 18
            torch.nn.InstanceNorm2d(256),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1), # 18 -> 36
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 32, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.InstanceNorm2d(32),
            torch.nn.ReLU()
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS), 
            #torch.nn.Dropout2d(p=0.5),
            torch.nn.Sigmoid()
        )

    def forward(self, voxel_features):
        voxel_features = voxel_features.permute(1, 0, 2, 3, 4, 5).contiguous()
        voxel_features = torch.split(voxel_features, 1, dim=0)
        Cspaces = []

        for features in voxel_features:

            Cspace = features.view(-1, 2048, 4, 4)
            #print(Cspace.size())
            Cspace = self.layer1(Cspace)
            #Cspace = torch.Tensor(Cspaces)
            #print(Cspace.size())
            Cspace = self.layer2(Cspace)
            #print(Cspace.size())
            Cspace = self.layer3(Cspace)
            #print(Cspace.size())
            Cspace = self.layer4(Cspace)
            #print(Cspace.size())
            Cspace = self.layer5(Cspace)
            #print(Cspace.size())
            Cspaces.append(torch.squeeze(Cspace, dim=1))

        Cspaces = torch.stack(Cspaces).permute(1, 0, 2, 3).contiguous()
        #print(Cspaces.size())
        return Cspaces
