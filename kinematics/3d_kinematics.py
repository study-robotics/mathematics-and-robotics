import numpy as np

def Hx(theta):
    return np.array([[1,             0,              0, 0],
                     [0, np.cos(theta), -np.sin(theta), 0],
                     [0, np.sin(theta),  np.cos(theta), 0],
                     [0,             0,              0, 1]])

# y軸周りに回転する同時変換行列
def Hy(theta):
    return np.array([[ np.cos(theta), 0,  np.sin(theta), 0],
                     [             0, 1,              0, 0],
                     [-np.sin(theta), 0,  np.cos(theta), 0],
                     [0,              0,              0, 1]])

# z軸周りに回転する同時変換行列
def Hz(theta):
    return np.array([[np.cos(theta),  -np.sin(theta), 0, 0],
                     [np.sin(theta),   np.cos(theta), 0, 0],
                     [            0,               0, 1, 0],
                     [            0,               0, 0, 1]])

# (x,y,z)方向に並進移動する同時変換行列
def Hp(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]])

