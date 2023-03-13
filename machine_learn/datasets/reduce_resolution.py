from time import time
from traceback import print_tb
import binvox_rw
#from mpl_toolkits.mplot3d import axes3d, Axes3D
#import matplotlib.pyplot as plt
import numpy as np
import torch

def reduce_resolution(before_voxel, after_resolution):
    before_resolution = len(before_voxel)
    kernel_size = before_resolution/after_resolution
    before_voxel = before_voxel[:, :, :, np.newaxis, np.newaxis] #pytorchが扱えるように次元を増加
    before_voxel = torch.from_numpy(before_voxel).float() # to tensor

    # [x, y, z, batch_size, channel] -> [batch_size, channel, x, y, z]（pytorchが正しく認識する形にするため，次元の順番を入れ替え）
    before_voxel = torch.permute(before_voxel, (3,4,0,1,2))

    m = torch.nn.MaxPool3d(int(kernel_size)) # pytorchのmaxpool3Dを利用して，分解能削減
    after_voxel = m(before_voxel)

    after_voxel = np.squeeze(after_voxel) # batch_sizeとchannelの次元を削除
    after_voxel = after_voxel.numpy().copy() # tensor to numpy
    return after_voxel

def main():
    # not time dataset
    """
    for i in range(6000):
        try:
            with open('./voxels/after_name/'+str(i)+'.binvox', 'rb') as f:
                model = binvox_rw.read_as_3d_array(f, fix_coords=False)
            before_voxel = model.data
            after_resolution = 32
            after_voxel = reduce_resolution(before_voxel, after_resolution)

            #fp = open("./voxels/after_name/to32/"+str(after_resolution)+"_"+str(i)+".binvox", 'wb')
            fp = open("./voxels/after_name/to32/"+str(i)+".binvox", 'wb')
            binvox = binvox_rw.Voxels(after_voxel, [after_resolution, after_resolution, after_resolution], [0,0,0], 1, 'xzy')
            binvox_rw.write(binvox, fp)
        except FileNotFoundError:
            print('number'+str(i)+' data is not found, finish.')
            break
    """

    # time dataset
    for scene_num in range(6000):
        for time_num in range(1000):
            try:
                with open('./voxels/'+str(scene_num)+'_'+str(time_num)+'.binvox', 'rb') as f:
                    model = binvox_rw.read_as_3d_array(f, fix_coords=False)
                before_voxel = model.data
                after_resolution = 32
                after_voxel = reduce_resolution(before_voxel, after_resolution)

                #fp = open("./voxels/after_name/to32/"+str(after_resolution)+"_"+str(i)+".binvox", 'wb')
                fp = open('./voxels/to32/'+str(scene_num)+'_'+str(time_num)+'.binvox', 'wb')
                binvox = binvox_rw.Voxels(after_voxel, [after_resolution, after_resolution, after_resolution], [0,0,0], 1, 'xzy')
                binvox_rw.write(binvox, fp)
            except FileNotFoundError:
                #print('number'+str(scene_num)+'_'+str(time_num)+' data is not found.')
                break


if __name__ == "__main__":
    main()