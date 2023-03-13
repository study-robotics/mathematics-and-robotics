# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models
import torchvision.transforms
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 64, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(64),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 64, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2),
            #torch.nn.Dropout3d(p=0.5),
        )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(128),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2),
            #torch.nn.Dropout3d(p=0.5),
        )
        
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(256),
            torch.nn.ReLU(),
        )

        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 256, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(256),
            torch.nn.ReLU(),
        )       

        self.layer7 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 256, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d(kernel_size=2),
            #torch.nn.Dropout3d(p=0.5),
        )    

        self.layer8 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(512),
            torch.nn.ReLU(),
        )     

        self.layer9 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 512, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(512),
            torch.nn.ReLU(),
        )  

        self.layer10 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 512, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(512),
            torch.nn.ReLU(),
        )  


    def forward(self, volumes):
        volumes = torch.split(volumes, self.cfg.CONST.BATCH_SIZE, dim=0)
        voxel_features = []

        for vox in volumes:
            # -------------------check input vox (debug) -------------------------
            """
            vox = vox[0]
            vox = np.squeeze(vox)
            print(vox.size())
            fig = plt.figure()
            ax = fig.gca(projection=Axes3D.name)
            ax.set_aspect('equal')
            ax.voxels(vox, edgecolor="k")

            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3, ))

            plt.draw()
            plt.savefig("hoge_vox.png")
            """
            # -----------------------------------------------------
            features = self.layer1(vox)
            #print(features.size())
            features = self.layer2(features)
            #print(features.size())
            features = self.layer3(features)
            #print(features.size())
            features = self.layer4(features)
            #print(features.size())
            features = self.layer5(features)
            #print(features.size())
            features = self.layer6(features)
            #print(features.size())
            features = self.layer7(features)
            #print(features.size())
            features = self.layer8(features)
            #print(features.size())
            features = self.layer9(features)
            #print(features.size())
            features = self.layer10(features)
            #print(features.size())
            voxel_features.append(features)

        voxel_features = torch.stack(voxel_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        return voxel_features # will change
