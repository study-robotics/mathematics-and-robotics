import binvox_rw
import numpy as np

def main():
    for i in range(1):
        cspace = np.load('/workspace/machine_learn/datasets/time_dataset/'+str(i)+'.npy',allow_pickle=True)
        print(cspace)

if __name__ == "__main__":
    main()