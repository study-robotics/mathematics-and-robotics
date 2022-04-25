from sympy.geometry import *
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import matplotlib.animation as animation
import math
import numpy as np
#障害物は真円の物体ひとつのみとする※アームが宙に浮いているイメージ
L1 = 5
L2 = 5
x0, y0 = L1+L2, L1+L2 #[リンクを最大限伸ばした数値]を原点とする※(以降で導出するx1,x2,y1,y2でマイナスの数値が出ないようにするため)
circle_x1, circle_y1 = (x0 + 2.3), (y0 + 4.8)
r1 = 1.0
fig = plt.figure(figsize=(10,5),dpi=96)
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_aspect("equal")#画像の比率を同じにする
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_aspect("equal")#画像の比率を同じにする
# 軸の範囲
plt.axes(ax1)
plt.xlabel('x')
plt.ylabel('y')
#plt.xlim([0, 8])
#plt.ylim([0, 8])
plt.axes(ax2)
plt.xlabel('theta1(rad)')
plt.ylabel('theta2(rad)')

circle1 = pat.Circle(xy = (circle_x1, circle_y1), radius= r1, color="blue")
ax1.add_patch(circle1)
ax1.set_xlim(0, (L1+L2)*2)#x方向（プラスマイナス)に最大限リンクを伸ばしても見える範囲
ax1.set_ylim(0, (L1+L2)*2)#グラフを表示した時の比率を1:1にするためxと同じ範囲に設定
ax2.set_xlim(0, 180)
ax2.set_ylim(0, 360)
class Theta2:
    val = 0
class Theta1:
    laps = 0
def updata(i):
  
  if (i % 360 == 0) and (i != 0):
    Theta1.laps += 1 #そのままだと360度を超えるので何周しているかを数える
    Theta2.val += 2 #そのままだと毎回theta2が0度になってしまうためクラス経由で
  theta1 = i - Theta1.laps*360
  theta2 = Theta2.val

  
  
  rad1 = theta1 * math.pi/180
  rad2 = theta2 * math.pi/180
  ax1.cla()
  ax1.set_xlim(0, (L1+L2)*2)
  ax1.set_ylim(0, (L1+L2)*2)
  ax1.add_patch(circle1)

  
  x1, y1 =  x0 + L1*math.cos(rad1), y0 + L1*math.sin(rad1)
  x2, y2 =  x1 + L2*math.cos(rad1+rad2), y1 + L2*math.sin(rad1+rad2) 
  
  ####アニメーション#####################################
  
  # セグメント（線分）の生成
  s0 = Segment(Point(x0, y0),Point(x1, y1))
  s3 = Segment(Point(x1, y1),Point(x2, y2))
  segments = [s0, s3]
  
  # グラフ描画
  for i,s in enumerate(segments) :
    ax1.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y])
  
  ############################################################3

  
  
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

  #リンク1またはリンク2のどちらかが接触したら描画
  
  if(hit1 == True or hit2 == True):
      ax2.plot(theta1, theta2, color='black',marker='.')
      
  else:
      pass

ani = animation.FuncAnimation(fig, updata, interval=20,  frames =64801)
#ani.save('anim.gif', writer="imagemagick")
ani.save('anim2.mp4', writer="ffmpeg")
plt.show()

#####################################
