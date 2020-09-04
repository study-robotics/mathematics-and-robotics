import numpy as np
import math
from sympy.geometry import *
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as animation
def now_hand_position(L,theta):
    x = L * math.cos(theta)
    y = L * math.sin(theta)
    return x,y

    
def  Jacobian(L1,theta1, L2, theta2, L3, theta3):#ヤコビアン2×3
    """
    -----------------------説明-------------------------
    J = [[∂x/∂θ1, ∂x/∂θ2, ∂x/θ3],
         [∂y/∂θ1, ∂y/∂θ2, ∂y/θ3]
    J = [[xをθ1で偏微分, xをθ2で偏微分, xをθ3で偏微分],     
         [yをθ1で偏微分, yをθ2で偏微分, yをθ3で偏微分]]
    -----------------------------------------------------
    """
    J = np.array([[-L1*math.sin(theta1) - L2*math.sin(theta1+theta2) -L3*math.sin(theta1+theta2+theta3), -L2*math.sin(theta1+theta2)-L3*math.sin(theta1+theta2+theta3), -L3*math.sin(theta1+theta2+theta3)],
                  [ L1*math.cos(theta1) + L2*math.cos(theta1+theta2) +L3*math.cos(theta1+theta2+theta3),  L2*math.cos(theta1+theta2)+L3*math.cos(theta1+theta2+theta3),  L3*math.cos(theta1+theta2+theta3)]])
    return J


def pseudo_inverse_matrix(mat):#擬似逆行列 numpy関数を仕様
    mat_p_inverse = np.linalg.pinv(mat)
    return mat_p_inverse

def Inverse_kinematics(x, y, goal_x, goal_y, L1, L2, L3, theta1, theta2, theta3):
    j = 0
    for i in range(500):
        J = Jacobian(L1, theta1, L2, theta2, L3, theta3)
        J_p_inverse = pseudo_inverse_matrix(J)
        print("J_p_inverse", J_p_inverse)
        
        """
        if  (J_p_inverse == -np.inf).any() or (J_p_inverse == np.inf).any() :
            theta1 += 0.1
            theta2 += 0.1
            theta3 += 0.1
            continue
        """
        

        
        Q = np.array([[theta1],
                      [theta2],
                      [theta3]])
        dx = goal_x - x
        dy = goal_y - y
        delta_Q = np.dot(J_p_inverse, np.array([[0.1*dx],
                                                [0.1*dy]]))
        Q_new = Q + delta_Q
        theta1 = Q_new[0,0]; theta2 = Q_new[1,0]; theta3 = Q_new[2,0]
        x1, y1 = now_hand_position(L1, theta1)
        x2, y2 = now_hand_position(L2, theta1+theta2)
        x3, y3 = now_hand_position(L3, theta1+theta2+theta3)
        x = x1+x2+x3
        y = y1+y2+y3
        j += 1
        print("number",j,"x",x,"y", y)
        if 2*math.pi < theta1:
            theta1 = 0.0
        if 2*math.pi < theta2:
            theta2 = 0.0
        if 2*math.pi < theta3:
            theta3 = 0.0
        if (goal_x-0.01< x <goal_x+0.01) and (goal_y-0.01 < y <goal_y+0.01):
            break
    if 490 < j:
        print("位置を計算できませんでした（特異点，もしくは実現不可能な座標の可能性があります")
    print("final_x", x ,"final_y", y)
    print("number of loop",j)
    return theta1, theta2, theta3
        



if __name__ == '__main__':
    ################デバック用#############
    """
    mat = np.array([[3.0, 2.0],
                    [1.0, 4.0]])#行列
    trans_mat = trans_matrix(mat)#転置行列
    print("trans_mat", trans_mat)
    inverse_mat = inverse_matrix(mat)#逆行列
    print("inverse_mat", inverse_mat)
    mat_p_inverse = pseudo_inverse_matrix(mat, trans_mat)#擬似逆行列
    print("mat_p_inverse", mat_p_inverse)
    """
    #######################################

    L1 = 2.0    
    deg1 = 60
    rad1 = deg1*math.pi/180
    L2 = 2.0
    deg2 = 60
    rad2 = deg2*math.pi/180
    L3 = 2.0
    deg3 = 40
    rad3 = deg3*math.pi/180
    x1, y1 = now_hand_position(L1, rad1)
    x2, y2 = now_hand_position(L2,rad1+rad2)
    x3, y3 = now_hand_position(L3, rad1+rad2+rad3)
    x = x1+x2+x3
    y = x1+x2+x3
    goal_x = -5.2
    goal_y = 3.0

    print("now hand point of x", x)
    print("now hand point of y", y)
    
    """
    J = Jacobian(L1, rad1, L2, rad2)
    #print("J",J)
    J_trans = trans_matrix(J)
    #print("J_trans",J_trans)
    J_inverse = inverse_matrix(J)
    #print("J_inverse", J_inverse)
    """
    
    final_theta1, final_theta2, final_theta3 = Inverse_kinematics(x, y, goal_x, goal_y, L1, L2, L3, rad1, rad2, rad3)
    deg1, deg2, deg3 = 180*final_theta1/math.pi, 180*final_theta2/math.pi, 180*final_theta3/math.pi
    print("final_theta1,final_theta2, final_theta3",deg1, deg2, deg3)

    
    x0, y0 = L1+L2+L3, L1+L2+L3
    x1, y1 = now_hand_position(L1, final_theta1)
    x1, y1 = x1+x0, y1+y0
    x2, y2 = now_hand_position(L2, final_theta1+final_theta2)
    x2, y2 = x2+x1, y2+y1
    x3, y3 = now_hand_position(L3, final_theta1+final_theta2+final_theta3)
    x3, y3 = x3+x2, y3+y2
    ###########

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1, 1, 1)
    s0 = Segment(Point(x0, y0),Point(x1, y1))
    s1 = Segment(Point(x1, y1),Point(x2, y2))
    s2 = Segment(Point(x2, y2),Point(x3, y3))
    segments = [s0, s1, s2]
    circle1 = pat.Circle(xy = (x0, y0), radius= L1+L2+L3, color="blue")
    ax1.add_patch(circle1)
    # グラフ描画
    for i,s in enumerate(segments) :
        plt.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y])
    plt.xlim(0,(L1+L2+L3)*2)
    plt.ylim(0,(L1+L2+L3)*2)
    
    plt.show()
