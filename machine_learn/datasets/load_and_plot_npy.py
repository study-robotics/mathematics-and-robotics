#!/usr/bin/env python
# -*- coding: utf-8 -*-
# license removed for brevity
import rospy
from std_msgs.msg import String
from moveit_msgs.msg  import DisplayRobotState, RobotState
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import skimage
import cv2

if __name__ == '__main__':
    all_not_collision_cspace = 0
    RESOLUTION = 1
    for i in range(600):
        print(i)
        
        #obstacles_img= np.load("./Cspaces/test/test_10deg/15.npy")
        obstacles_img= np.load("/workspace/Vox2Cspace/datasets/Cspaces/7_"+str(i)+".npy")
        #obstacleList = np.load("/workspace/Vox2Cspace/datasets/Cspaces/623.npy")
        #print(obstacleList)
        #このままだと、縦軸x,横軸yで原点が左下の図が表示されるので直す
        obstacles_img = np.rot90(obstacles_img, -1)#横軸x,縦軸y
        obstacles_img = np.fliplr(obstacles_img)#左右反転
        #obstacles_img = skimage.img_as_ubyte(obstacles_img)
        #obstacles_img = np.linalg.inv(obstacles_img)#そのままだと、障害物：白、背景：黒なので白黒を反転させる
        #io.imshow(obstacles_img)
        #print("end")
        plt.imshow(obstacles_img)
        plt.axis([0, 360/RESOLUTION, 0, 360/RESOLUTION])
        plt.grid(True)
        plt.show()
    
        
    #print(all_not_collsiion_cspace)
    
