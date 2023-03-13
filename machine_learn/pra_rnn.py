# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

'''ライブラリの準備'''
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import numpy as np
import matplotlib.pyplot as plt

class Encoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=3, padding=1),
            torch.nn.InstanceNorm3d(64),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            #torch.nn.Dropout3d(p=0.5),
        )
    def forward(self, voxel):
    

        feature = self.layer1(voxel)
        #print(features.size())
        feature = self.layer2(feature)
        #print(features.size())

        return feature 

    '''
    def forward(self, volumes):
    
        volumes = torch.split(volumes, self.cfg.CONST.BATCH_SIZE, dim=0)
        voxel_features = []

        for vox in volumes:
            features = self.layer1(vox)
            #print(features.size())
            features = self.layer2(features)
            #print(features.size())

        voxel_features = torch.stack(voxel_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        return voxel_features # will change
    '''

class GRU(torch.nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = torch.nn.RNN(1, 64, batch_first=True)
        #self.fc = torch.nn.Linear(64, 1)
    def forward(self, voxel_time_feature):
        #batch_size = x.size(0)
        output, hidden = self.gru(voxel_time_feature, None)
        return output
model = GRU()





'''GPUチェック'''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


x = np.linspace(0, 4*np.pi)
sin_x = np.sin(x) + np.random.normal(0, 0.3, len(x))
plt.plot(x, sin_x)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()

'''ハイパーパラメータ'''
n_time = 10
n_sample = len(x) - n_time

'''データを格納する空の配列を準備'''
input_data = np.zeros((n_sample, n_time, 1))
correct_data = np.zeros((n_sample, 1))

'''前処理'''
for i in range(n_sample):
    input_data[i] = sin_x[i:i+n_time].reshape(-1, 1)
    correct_data[i] = [sin_x[i+n_time]]
input_data = torch.FloatTensor(input_data)
correct_data = torch.FloatTensor(correct_data)

'''バッチデータの準備'''
dataset = TensorDataset(input_data, correct_data)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

'''最適化手法の定義'''
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

'''誤差(loss)を記録する空の配列を用意'''
record_loss_train = []

for i in range(201):
  model.train()
  loss_train = 0
  for j, (x, t) in enumerate(train_loader):
    print(x.size())
    print("x[0]", x[0])
    #x00 = x[0][0]
    #x01 = x[0][1]
    #x02 = x[0][2]
    #x03 = x[0][3]
    #x04 = x[0][4]
    #x05 = x[0][5]
    #x06 = x[0][6]
    #x07 = x[0][7]
    #x08 = x[0][8]
    #x09 = x[0][9]
    time_num_length = 10
    not_time_xs = []

    # 時間要素を剥奪
    for time in range(time_num_length):

        not_time_x = x[:, time, :]
        print(x.size())
        print(not_time_x.size())
        not_time_xs.append(not_time_x)

    # 各時刻のボクセルをエンコーダに入力 -> 時刻分だけ特徴量を抽出
    features = []
    for time in range(time_num_length):
        feature = not_time_xs[time]
        features.append(feature)
    # for i in range(time_num_length):
    #   feature = encoder(not_time_xs[time])
    #   features.append(feature)
       
    # 特徴量をGRUに入力するために，時系列順に並べる（pytorchの書き方で）
    bathch_size = 8
    input_size = 1

    # torch.zeros((bathch_size, time_num_length, channel, x, y, z))
    time_features = torch.zeros((bathch_size, time_num_length, input_size)) 
    for time in range(time_num_length):
        time_features[:, time, :] = features[time]
    
    print(x)
    print(time_features)
    print(x.size())
    print(time_features.size())












