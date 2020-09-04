##############
"""
線分×円の衝突判定を使用(circle_collision.py)
逆運動学を使用(inversr_kinematics.py)
衝突判定とCspaceの構築を同時に行うプログラム
（デバック用にCspaceを完全に構築した後にこのプログラムを行ったプログラム
はDebug_Cspace_and_collision.pyとする）
start_deg1, start_deg2 = 80, 10
goalx, goaly = 3.0,7.0
L1 = 5.0    
L2 = 5.0
circle_x1, circle_y1 = (x0 + 2.3), (y0 + 4.8)
r1 = 1.0
とするとわかりやすい結果が出る
@@@@@@@@@@@@@@@@@@@@@@@@現時点では，一度グラフの隅に行くと上手く行かないので改善が必要@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
"""
①「衝突判定+Cspaceの構築」
elapsed_time:0.7537147998809814[sec]
loop回数 1203 
---------------------------------------
②「Cspaceの構築」(+rrtなどの経路計画の手法によって経路を求める※現時点ではこの部分はできてない)
elapsed_time:8.555245161056519[sec]
loop回数　65341
"""
##############
import numpy as np
import math
from sympy.geometry import *
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as animation
#ここから逆運動学(inverse_kinematic.pyと同じ)---------------------------------------------------------
def now_hand_position(L,theta):#順運動学
    x = L * math.cos(theta)
    y = L * math.sin(theta)
    return x,y

    
def  Jacobian(L1,theta1, L2, theta2):#ヤコビアン2×2
    """
    -----------------------説明-------------------------
    J = [[∂x/∂θ1, ∂x/∂θ2],
         [∂y/∂θ1, ∂y/∂θ2]]
    J = [[xをθ1で偏微分, xをθ2で偏微分],     
         [yをθ1で偏微分, yをθ2で偏微分]]
    -----------------------------------------------------
    """
    J = np.array([[-L1*math.sin(theta1) - L2*math.sin(theta1+theta2), -L2*math.sin(theta1+theta2)],
                  [ L1*math.cos(theta1) + L2*math.cos(theta1+theta2),  L2*math.cos(theta1+theta2)]])
    return J

def trans_matrix(mat):#転置行列2×2
    """
    -----------------------説明-------------------------
    行列はnumpyの記述方法で記述する
    行列A       = [[a, b],
                   [c, d]]　のとき
    転置行列A.T = [[a, c],
                   [b, d]]
    -----------------------------------------------------
    """
    mat_minus =  np.array([[mat[0,0], mat[1,0]],
                           [mat[0,1], mat[1,1]]])
    return mat_minus

def inverse_matrix(mat):#逆行列2×2（以下の公式が使えるのは2×2のときのみ，3×3以上は掃き出し法により算出）
    """
    -----------------------説明-------------------------
    A=  [[a, b],
         [c, d]] の逆行列は，
    A^−1   = 1/ad−bc  * [[d, −b]
                         [-c, a]] 
    -----------------------------------------------------
    """
    inverse_mat = 1/(mat[0,0]*mat[1,1] - mat[0,1]*mat[1,0]) * np.array([[mat[1,1], -mat[0,1]],
                                                                        [-mat[1,0], mat[0,0]]])
    return inverse_mat                                                                      

def pseudo_inverse_matrix(mat, trans_mat):#擬似逆行列（正方行列以外の逆行列を用いるときに使用．今回は，2×2の正方行列であるため，この関数は使用しなくても可）
    """
    -----------------------説明-------------------------
    擬似逆行列A+ =(A.T * A)^-1 * A.T
    記号の意味-----------
    A.T ：転置行列
    A^-1：逆行列
    *   ：内積
    ---------------------
    -----------------------------------------------------
    """
    mat_p_inverse = np.dot(inverse_matrix(np.dot(trans_mat,mat)),trans_mat)
    return mat_p_inverse

def Inverse_kinematics(x, y, goal_x, goal_y, L1, L2, theta1, theta2):
    j = 0
    for i in range(100):
        J = Jacobian(L1, theta1, L2, theta2)#ヤコビアン
        J_trans = trans_matrix(J)#ヤコビアンの転置
        J_inverse = inverse_matrix(J)#ヤコビアンの逆行列
        J_p_inverse = pseudo_inverse_matrix(J, J_trans)#ヤコビアンの擬似逆行列（今回は正方行列なのでJ_inverseと同じ値になる）
        """
        if  (J_p_inverse == -np.inf).any() or (J_p_inverse == np.inf).any() :
            theta1 += 0.1
            theta2 += 0.1
            continue
        """
        Q = np.array([[theta1],
                      [theta2]])#各角度をリストQに格納
        dx = goal_x - x #x方向の目標距離と現在距離との差分（差が大きいと動かす角度も増加）
        dy = goal_y - y #y方向の目標距離と現在距離との差分（差が大きいと動かす角度も増加）
        delta_Q = np.dot(J_inverse, np.array([[dx*0.1],
                                              [dy*0.1]]))#下記のニュートン法の式のJ^-1*f_nに当たる（はず）
        Q_new = Q + delta_Q#新しい角度を求める（ニュートン法に基づく）ニュートン法：x_n+1 = x_n - J^-1*f_n
        theta1 = Q_new[0,0]; theta2 = Q_new[1,0]
        x1, y1 = now_hand_position(L1, theta1)#リンク〜関節1（リンク2の根本）までのx,y
        x2, y2 = now_hand_position(L2, theta1+theta2)#関節1（リンク2の根本）から手先までのx,y
        x = x1+x2 #原点（リンク1の根本）〜手先までのx
        y = y1+y2 #原点（リンク1の根本）〜手先までのy
        j += 1 #for文を繰り返したかをカウント（デバック用）
        print("number",j,"x",x,"y", y)
        #各関節が2π(360度)を超えたら0.0radに戻る
        if 2*math.pi < theta1:
            theta1 = 0.0
        if 2*math.pi < theta2:
            theta2 = 0.0
        if theta1 < 0.0:
            theta1 = 2*math.pi
        if theta2 < 0.0:
            theta2 = 2*math.pi

        #手先が目標座標あたりにきたらfor文から抜ける
        if (goal_x-0.01< x <goal_x+0.01) and (goal_y-0.01 < y <goal_y+0.01):
            break
    #繰り返し距離が一定以上なら，計算ができていないとしてprint文で出力
    if 90 < j:
        print("位置を計算できませんでした（特異点，もしくは実現不可能な座標の可能性があります）")
    print("final_x", x ,"final_y", y)#最終的な手先の座標を出力
    print("number of loop",j)#最終的なループの回数を出力
    return theta1, theta2
#ここまで逆運動学---------------------------------------------------------------------------------
#ここから衝突判定の関数（circle_collision.pyの衝突判定を関数化しただけ）-------------------------------
def collision(x0,y0,x1,y1,x2,y2):
    #線分の始点から円の中心点の方向に向かうベクトル
    first_line1_r1_x1, first_line1_r1_y1 = circle_x1-x0, circle_y1-y0 
    first_line2_r1_x2, first_line2_r1_y2 = circle_x1-x1, circle_y1-y1
    #線分の終点から円の中心点の方向に向かうベクトル
    last_line1_r1_x1, last_line1_r1_y1  = circle_x1-x1, circle_y1-y1
    last_line2_r1_x2, last_line2_r1_y2 = circle_x1-x2, circle_y1-y2
    #線分の始点から円の中心点の方向に向かうベクトルの大きさ
    first1_size = math.sqrt((first_line1_r1_x1)**2 + (first_line1_r1_y1)**2)
    first2_size = math.sqrt((first_line2_r1_x2)**2 + (first_line2_r1_y2)**2)
    #線分の終点から円の中心点の方向に向かうベクトルの大きさ
    last1_size = math.sqrt((last_line1_r1_x1)**2 + (last_line1_r1_y1)**2)
    last2_size = math.sqrt((last_line2_r1_x2)**2 + (last_line2_r1_y2)**2)
    #print("first1",first1_size)
    #print("first2",first2_size)
    #print("last1", last1_size)
    #print("last2", last2_size)

    #線分のベクトルの大きさ
    line1_size = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    line2_size = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    #単位ベクトル化
    unit_x1, unit_y1 = (x1-x0)/line1_size, (y1-y0)/line1_size
    unit_x2, unit_y2 = (x2-x1)/line2_size, (y2-y1)/line2_size
    #【線分】と【線分の始点から円の中心点の方向に向かうベクトル】の内積
    first1_dot = (x1-x0) * first_line1_r1_x1 + (y1-y0) * first_line1_r1_y1
    first2_dot = (x2-x1) * first_line2_r1_x2 + (y2-y1) * first_line2_r1_y2
    #【線分】と【線分の終点から円の中心点の方向に向かうベクトル】の内積
    last1_dot = (x1-x0) * last_line1_r1_x1 + (y1-y0) * last_line1_r1_y1
    last2_dot = (x2-x1) * last_line2_r1_x2 + (y2-y1) * last_line2_r1_y2

    #線分と円の中心点との距離　A×B = Ax*By - Bx*Ay = |A||B|sinθ
    r1_line1_min = abs(unit_x1 * (circle_y1-y0) - (circle_x1-x0) * unit_y1)
    r1_line2_min = abs(unit_x2 * (circle_y1-y1) - (circle_x1-x1) * unit_y2)
    #print("aaaaaaaa",r1_line1_min)
    #print("aaaaaaaa",r1_line2_min)

    #リンク１の接触判定----------------
    hit1 = False
    if(r1_line1_min < r1):#直線内に円がある（線分内にあるかはこの時点で不明）
        #print("接触している可能性がある")
        #結果が同じ(A、Bともに鋭角 or 鈍角)場合は範囲外(触れていない可能性がある)
        if( (first1_dot < 0 and last1_dot < 0) or (0 < first1_dot and 0 < last1_dot) or (first1_dot == 0 and last1_dot == 0)):
            hit1 = False
        else:#二つの結果(Aが鋭角、Bが鈍角等)が異なれば線分の範囲内(確定で接触)
            hit1 = True
            #print("接触している(Aが鋭角、Bが鈍角等2つの結果が異なる)")
    if(hit1 == False):
        #線分の末端が円の範囲内にあるかどうか
        if((first1_size < r1) or (last1_size < r1)):
            hit1 = True#確定で接触
            #print("B接触している（線分の末端が円の範囲内にある）")
        else:
            hit1 = False#確定で非接触
    #---------------------------------
    #リンク2の接触判定---------------
    hit2 = False
    if(r1_line2_min < r1):#直線内に円がある（線分内にあるかはこの時点で不明）
        #print("接触している可能性がある")
        #結果が同じ(A、Bともに鋭角 or 鈍角)場合は範囲外
        if( (first2_dot < 0 and last2_dot < 0) or (0 < first2_dot and 0 < last2_dot) or (first2_dot == 0 and last2_dot == 0)):
            hit2 = False
        else:#二つの結果(Aが鋭角、Bが鈍角等)が異なれば線分の範囲内
            hit2 = True
            #print("接触している(Aが鋭角、Bが鈍角等2つの結果が異なる)")
    if(hit2 == False):
        #線分の末端が円の範囲内にあるかどうか
        if((first2_size < r1) or (last2_size < r1)):
            hit2 = True
            #print("B接触している（線分の末端が円の範囲内にある）")
        else:
            hit2 = False
    #--------------------------
    return hit1, hit2
#ここまで衝突判定の関数--------------------------------------------------------------------------
if __name__ == '__main__':

    #ここから逆運動学(inverse_kinematics.pyと同じ）--------------------------------------------------------------------------------------------
    #必要な変数（リンクの長さ，関節角度，目標座標など）を用意##
    L1 = 5.0    
    deg1 = 20
    rad1 = deg1*math.pi/180
    L2 = 5.0
    deg2 = 20
    rad2 = deg2*math.pi/180    
    x0, y0 = L1+L2, L1+L2
    x1, y1 = now_hand_position(L1, rad1)
    x1, y1 = x1+x0, y1+y0
    x2, y2 = now_hand_position(L2,rad1+rad2)
    x2, y2 = x2+x1, y2+y1
    x = x1+x2
    y = x1+x2
    goal_x = float(input("目標座標xを入力（3がおすすめ）"))
    goal_y = float(input("目標座標yを入力（7がおすすめ）"))

    circle_x1, circle_y1 = (x0 + 2.3), (y0 + 4.8)
    r1 = 1.0
    ###########################################################
    #描画-------------------------------------------------------------------
    fig = plt.figure(figsize=(10,5),dpi=96)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    # 軸の範囲
    plt.axes(ax1)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.xlim([0, 8])
    #plt.ylim([0, 8])
    ims = []
    plt.axes(ax2)
    plt.xlabel('theta1(rad)')
    plt.ylabel('theta2(rad)')
    plt.xlim([0, 180])
    plt.ylim([0, 360])
    circle1 = pat.Circle(xy = (circle_x1, circle_y1), radius= r1, color="blue")
    ax1.add_patch(circle1)
    ax1.set_xlim(0, (L1+L2)*2)#x方向（プラスマイナス)に最大限リンクを伸ばしても見える範囲
    ax1.set_ylim(0, (L1+L2)*2)#グラフを表示した時の比率を1:1にするためxと同じ範囲に設定
    #---------------------------------------------------------------------------------------------

    #逆運動学（最終的な各関節の角度が返り値）
    final_theta1, final_theta2 = Inverse_kinematics(x, y, goal_x, goal_y, L1, L2, rad1, rad2)
    #rad→degに変換（確認)
    deg1, deg2 = 180*final_theta1/math.pi, 180*final_theta2/math.pi
    print("final_theta1,final_theta2",deg1, deg2)
    #逆運動学ここまで-------------------------------------------------------------------------------------------------------

    #ここからCspace構築と衝突判定同時-----------------------------------------------------------------------------------------------------------------
    collisionSx = []#常に通った場所は障害物として判定(x方向=theta1)
    collisionSy = []#常に通った場所は障害物として判定(y方向=theta2)
    collisionS = []
    start_deg1 = float(input("スタート時の角度deg1を入力(80がおすすめ）"))
    start_deg2 = float(input("スタート時の角度deg2を入力(10がおすすめ）"))
    now_deg1, now_deg2 = start_deg1, start_deg2
    goal_deg1, goal_deg2 = deg1, deg2
    ax2.plot(start_deg1, start_deg2, color='red',marker='o')
    ax2.plot(goal_deg1, goal_deg2, color='red',marker='o')

    thetaS = []
    for i in range(10000):
        if (abs(goal_deg1- now_deg1) < 1.0) and (abs(goal_deg2 - now_deg2) < 1.0):
            print("end")
            break
        S_to_G_vecX, S_to_G_vecY = goal_deg1 - now_deg1, goal_deg2 - now_deg2 #Cspaceの最初の角度→最後の角度を結ぶ直線ベクトル
        now_rad1 = now_deg1*math.pi/180
        now_rad2 = now_deg2*math.pi/180
        S_to_G_vec_size = math.sqrt((goal_deg1 - now_deg1)**2 + (goal_deg2 - now_deg2)**2)
        unit_deg1, unit_deg2 = (goal_deg1-now_deg1)/S_to_G_vec_size, (goal_deg2-now_deg2)/S_to_G_vec_size#単位ベクトル化
        unit_rad1, unit_rad2 = unit_deg1*math.pi/180, unit_deg2*math.pi/180


        x1,y1 = now_hand_position(L1, now_rad1+unit_rad1)
        x1, y1 = x1+x0, y1+y0
        x2,y2 = now_hand_position(L2, now_rad1+unit_rad1+now_rad2+unit_rad2)
        x2, y2 = x2+x1, y2+y1
        hit1, hit2 = collision(x0,y0,x1,y1,x2,y2)
        if (hit1 == False and hit2 == False) and ((now_deg1+unit_deg1,now_deg2+unit_deg2)not in collisionS):#衝突していない　かつ　通ったところのない場所（角度）
            now_deg1, now_deg2 = now_deg1+unit_deg1, now_deg2+unit_deg2
            if now_deg1 < 0:  
                now_deg1 = 0
            if 180 < now_deg1 :
                now_deg1 = 180
            if now_deg2 < 0:  
                now_deg2 = 0
            if 360 < now_deg2 :
                now_deg2 = 360
            ax2.plot(now_deg1, now_deg2, color='green',marker='+')#描画
            collisionS.append((now_deg1,now_deg2))
            #print("stright")
            continue
            

        #リンク1またはリンク2のどちらかが接触または「現在の位置を(1つ以上前のループで）常に通っている」場合描画してy方向に1進む
        if (hit1 == True or hit2 == True) or ((now_deg1,now_deg2) in collisionS):
            #ax2.plot(unit_deg1+now_deg1, unit_deg2+now_deg2, color='0.0',marker='+')
            unit_deg1, unit_deg2 = 0, 1.0
            unit_rad1, unit_rad2 = unit_deg1*math.pi/180, unit_deg2*math.pi/180
            x1,y1 = now_hand_position(L1, now_rad1+unit_rad1)
            x1, y1 = x1+x0, y1+y0
            x2,y2 = now_hand_position(L2, now_rad1+unit_rad1+now_rad2+unit_rad2)
            x2, y2 = x2+x1, y2+y1
            hit1, hit2 = collision(x0,y0,x1,y1,x2,y2)
            if (hit1 == False and hit2 == False) and ((now_deg1+unit_deg1,now_deg2+unit_deg2)not in collisionS):
                now_deg1, now_deg2 = now_deg1+unit_deg1, now_deg2+unit_deg2
                if now_deg1 < 0:  
                    now_deg1 = 0
                if 180 < now_deg1 :
                    now_deg1 = 180
                if now_deg2 < 0:  
                    now_deg2 = 0
                if 360 < now_deg2 :
                    now_deg2 = 360
                collisionS.append((now_deg1,now_deg2))
                ax2.plot(now_deg1, now_deg2, color='green',marker='+')
                #print("up")
                continue
                
                
                


        #リンク1またはリンク2のどちらかが接触または「現在の位置を(1つ以上前のループで）常に通っている」場合描画してx方向に1進む
        if (hit1 == True or hit2 == True) or ((now_deg1,now_deg2) in collisionS):
            #ax2.plot(unit_deg1+now_deg1, unit_deg2+now_deg2, color='0.0',marker='+')
            unit_deg1, unit_deg2 = 1.0 , 0.0
            unit_rad1, unit_rad2 = unit_deg1*math.pi/180, unit_deg2*math.pi/180
            x1,y1 = now_hand_position(L1, now_rad1+unit_rad1)
            x1, y1 = x1+x0, y1+y0
            x2,y2 = now_hand_position(L2, now_rad1+unit_rad1+now_rad2+unit_rad2)
            x2, y2 = x2+x1, y2+y1
            hit1, hit2 = collision(x0,y0,x1,y1,x2,y2)
            if (hit1 == False and hit2 == False) and ((now_deg1+unit_deg1,now_deg2+unit_deg2)not in collisionS):
                now_deg1, now_deg2 = now_deg1+unit_deg1, now_deg2+unit_deg2
                if now_deg1 < 0:  
                    now_deg1 = 0
                if 180 < now_deg1 :
                    now_deg1 = 180
                if now_deg2 < 0:  
                    now_deg2 = 0
                if 360 < now_deg2 :
                    now_deg2 = 360
                collisionS.append((now_deg1,now_deg2))
                ax2.plot(now_deg1, now_deg2, color='green',marker='+')
                #print("right")
                continue
                    
                    
        #リンク1またはリンク2のどちらかが接触または「現在の位置を(1つ以上前のループで）常に通っている」場合描画してx方向に-1進む
        if (hit1 == True or hit2 == True) or ((now_deg1,now_deg2) in collisionS):
            #ax2.plot(unit_deg1+now_deg1, unit_deg2+now_deg2, color='0.0',marker='+')
            unit_deg1, unit_deg2 = -1.0, 0.0
            unit_rad1, unit_rad2 = unit_deg1*math.pi/180, unit_deg2*math.pi/180
            x1,y1 = now_hand_position(L1, now_rad1+unit_rad1)
            x1, y1 = x1+x0, y1+y0
            x2,y2 = now_hand_position(L2, now_rad1+unit_rad1+now_rad2+unit_rad2)
            x2, y2 = x2+x1, y2+y1
            hit1, hit2 = collision(x0,y0,x1,y1,x2,y2)
            if (hit1 == False and hit2 == False) and ((now_deg1,now_deg2)not in collisionS):
                now_deg1, now_deg2 = now_deg1+unit_deg1, now_deg2+unit_deg2
                if now_deg1 < 0:  
                    now_deg1 = 0
                if 180 < now_deg1 :
                    now_deg1 = 180
                if now_deg2 < 0:  
                    now_deg2 = 0
                if 360 < now_deg2 :
                    now_deg2 = 360
                collisionS.append((now_deg1,now_deg2))
                ax2.plot(now_deg1, now_deg2, color='green',marker='+')
                #print("left")
                continue
                        
                        
        #リンク1またはリンク2のどちらかが接触または「現在の位置を(1つ以上前のループで）常に通っている」場合描画してy方向に-1進む
        if (hit1 == True or hit2 == True) or ((now_deg1,now_deg2) in collisionS):
            #ax2.plot(unit_deg1+now_deg1, unit_deg2+now_deg2, color='0.0',marker='+')
            unit_deg1, unit_deg2 =  0.0,  -1.0
            unit_rad1, unit_rad2 = unit_deg1*math.pi/180, unit_deg2*math.pi/180
            x1,y1 = now_hand_position(L1, now_rad1+unit_rad1)
            x1, y1 = x1+x0, y1+y0
            x2,y2 = now_hand_position(L2, now_rad1+unit_rad1+now_rad2+unit_rad2)
            x2, y2 = x2+x1, y2+y1
            hit1, hit2 = collision(x0,y0,x1,y1,x2,y2)
            if (hit1 == False and hit2 == False) and ((now_deg1+unit_deg1,now_deg2+unit_deg2)not in collisionS):
                now_deg1, now_deg2 = now_deg1+unit_deg1, now_deg2+unit_deg2
                if now_deg1 < 0:  
                    now_deg1 = 0
                if 180 < now_deg1 :
                    now_deg1 = 180
                if now_deg2 < 0:  
                    now_deg2 = 0
                if 360 < now_deg2 :
                    now_deg2 = 360
                collisionS.append((now_deg1,now_deg2))
                ax2.plot(now_deg1, now_deg2, color='green',marker='+')
                #print("down")
                continue
    #ここまでCspace構築と衝突判定同時-----------------------------------------------------------------------------------------------------------------

    #####以下グラフ表示に関する記述########
    x0, y0 = L1+L2, L1+L2
    x1, y1 = now_hand_position(L1, final_theta1)
    x0_x1, y0_y1 = x0    + x1, y0    + y1
    x2, y2 = now_hand_position(L2, final_theta1+final_theta2)
    x0_x2, x0_y2 = x0_x1 + x2, y0_y1 + y2

    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1, 1, 1)
    #Segment(Point(x0,y0), Point(x1,y1)):2つの点を結ぶ線分 
    s0 = Segment(Point(x0   , y0)   ,Point(x0_x1, y0_y1))
    s1 = Segment(Point(x0_x1, y0_y1),Point(x0_x2, x0_y2))
    segments = [s0, s1]
    #アームの駆動範囲の円（真円）
    circle1 = pat.Circle(xy = (x0, y0), radius= L1+L2, color="blue")
    ax1.add_patch(circle1)
    # グラフ描画
    for i,s in enumerate(segments) :
        plt.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y])
    plt.xlim(0,(L1+L2)*2)
    plt.ylim(0,(L1+L2)*2)
    plt.show()
    print(len(collisionSx))
    print(len(collisionSy))

    print(collisionSx)
    print(collisionSy)