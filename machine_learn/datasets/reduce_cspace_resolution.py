from traceback import print_tb
import binvox_rw
#from mpl_toolkits.mplot3d import axes3d, Axes3D
#import matplotlib.pyplot as plt
import numpy as np
import torch

def reduce_resolution(before_cspace, after_resolution):
    before_resolution = len(before_cspace)
    kernel_size = before_resolution/after_resolution
    before_cspace = before_cspace[:, :, np.newaxis, np.newaxis] #pytorchが扱えるように次元を増加
    before_cspace = torch.from_numpy(before_cspace).float() # to tensor

    # [x, y, z, batch_size, channel] -> [batch_size, channel, x, y, z]（pytorchが正しく認識する形にするため，次元の順番を入れ替え）
    print(len(before_cspace[1]))
    before_cspace = torch.permute(before_cspace, (2, 3, 0, 1))

    after_cspace = torch.max_pool2d(before_cspace, int(kernel_size)) # pytorchのmaxpool3Dを利用して，分解能削減

    after_cspace = np.squeeze(after_cspace) # batch_sizeとchannelの次元を削除
    after_cspace = after_cspace.numpy().copy() # tensor to numpy
    return after_cspace

def main():
    for i in range(1):
        load_cspace_path = './Cspaces/test/test_1deg/'+str(i)+'.npy'
        before_cspace = np.load(load_cspace_path)

        after_resolution = int(360/10)
        after_cspace = reduce_resolution(before_cspace, after_resolution)

        save_cspace_path = "./Cspaces/test/test_1deg/"+str(after_resolution)+"_"+str(i)+".npy"
        np.save(save_cspace_path, after_cspace)

if __name__ == "__main__":
    main()