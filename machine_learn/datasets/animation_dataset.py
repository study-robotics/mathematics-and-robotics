#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sqlalchemy import Interval
import binvox_rw
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import skimage


fig = plt.figure(figsize=(10,4))

ax2 = fig.add_subplot(122)
RESOLUTION = 1
ax2.set_xlim(0, 360/RESOLUTION)
ax2.set_ylim(0, 360/RESOLUTION)
ax2.set_xlabel("theta1(deg)")
ax2.set_ylabel("theta2(deg)")
ax2.set_title("voxel (simulation space)")

ax1 = fig.add_subplot(121, projection="3d")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax2.set_title("Configuration space")

# https://qiita.com/AnchorBlues/items/3acd37331b12e844d259
def update(time_num, scene_num):
    plt.cla()
    cspace = np.load("./Cspaces/"+str(scene_num)+"_"+str(time_num)+".npy")
    #このままだと、縦軸x,横軸yで原点が左下の図が表示されるので直す
    cspace = np.rot90(cspace, -1)#横軸x,縦軸y
    cspace = np.fliplr(cspace)#左右反転
    cspace = np.invert(cspace)#そのままだと、障害物：白、背景：黒なので白黒を反転させる
    ax2.imshow(cspace, cmap="gray")

    with open('./voxels/'+str(scene_num)+'_'+str(time_num)+'.binvox', 'rb') as f:
        voxel = binvox_rw.read_as_3d_array(f, fix_coords=False)
    ax1.voxels(voxel.data, edgecolor='k')
    fig.suptitle("time"+str(time_num))

# 0_0.npy〜0_80.npyまである場合，scene_num = 0, time_num = 80にすると1シーン分アニメーション化
scene_num = 45
time_num = 89
ani = animation.FuncAnimation(fig, update, fargs = (scene_num,), frames = time_num, interval=100)
#plt.show()


ani.save(str(scene_num)+'_'+str(time_num)+'.mp4', writer="ffmpeg",dpi=100)
