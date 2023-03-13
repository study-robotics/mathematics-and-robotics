from time import time
from traceback import print_tb
import binvox_rw
import numpy as np
import torch
import shutil

def main():
    num = 0
    for scene_num in range(1000):
        for time_num in range(1000):
            try:
                before_voxel_path = './voxels/'+str(scene_num)+'_'+str(time_num)+'.binvox'
                after_voxel_path = './voxels/after_name/'+str(num)+'.binvox'
                shutil.copy(before_voxel_path, after_voxel_path)
                #after_resolution = 32
                #after_voxel = reduce_resolution(before_voxel, after_resolution)

                before_cspace_path = './Cspaces/'+str(scene_num)+'_'+str(time_num)+'.npy'
                after_cspace_path = './Cspaces/after_name/'+str(num)+'.npy'
                shutil.copy(before_cspace_path, after_cspace_path)
                num += 1
            except FileNotFoundError:
                break

if __name__ == "__main__":
    main()