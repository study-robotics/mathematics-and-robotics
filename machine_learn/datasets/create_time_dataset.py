from tkinter.tix import Tree
import binvox_rw
import numpy as np

def main():
    voxel_time_length = 5
    cspace_time_length = 1
    num = 0 # 統合後のnpyファイルのファイル名
    for scene in range(20):
        start_time = 0
        while True:
            try:
                voxels = []
                cspaces = []

                # 例：start_time=0, voxel_time_length=5の場合
                #   「時刻0〜4」のボクセルを一つにまとめる（seq2seqへの入力）
                for time in range(start_time, start_time+voxel_time_length):
                    with open('./voxels/'+str(scene)+"_"+str(time)+'.binvox', 'rb') as f:
                        print("voxel time", time)
                        model = binvox_rw.read_as_3d_array(f, fix_coords=False)
                        voxel = model.data
                        voxels.append(voxel)

                # 例： start_time=0, voxel_time_length=5, cspace_time_length=2の場合
                #   「時刻5〜6」のCspaceを一つにまとめる（seq2seqからの出力）
                output_time = start_time+voxel_time_length
                for time in range(output_time, output_time+cspace_time_length):
                    print("cspace time", time)
                    cspace = np.load('./Cspaces/'+str(scene)+"_"+str(time)+'.npy')
                    cspaces.append(cspace)

                np.save('./time_dataset/'+str(num)+'.npy', (voxels, cspaces))
                num += 1

                start_time += 1

                
            except FileNotFoundError:
                print(str(scene)+"_"+str(output_time)+'.npy not found')
                break
#https://tomomai.com/python-dataset-npy/
if __name__ == "__main__":
    main()