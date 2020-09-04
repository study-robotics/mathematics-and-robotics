from sympy.geometry import *
import matplotlib.pyplot as plt
import matplotlib.patches as pat
#import matplotlib.animation as animation
import math
import numpy as np
import time
start = time.time()
#障害物は真円の物体ひとつのみとする※アームが宙に浮いているイメージ
L1 = 5
L2 = 5
x0, y0 = L1+L2, L1+L2 #[リンクを最大限伸ばした数値]を原点とする※(以降で導出するx1,x2,y1,y2でマイナスの数値が出ないようにするため)
circle_x1, circle_y1 = (x0 + 2.3), (y0 + 4.8)
r1 = 1.0
#############
circle_x1_sikaku_min = circle_x1 - r1
circle_x1_sikaku_max = circle_x1 + r1
circle_y1_sikaku_min = circle_y1 - r1
circle_y1_sikaku_max = circle_y1 + r1

##############
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
plt.xlabel('theta1(deg)')
plt.ylabel('theta2(deg)')
plt.xlim([0, 180])
plt.ylim([0, 360])
circle1 = pat.Circle(xy = (circle_x1, circle_y1), radius= r1, color="blue")
ax1.add_patch(circle1)
ax1.set_xlim(0, (L1+L2)*2)#x方向（プラスマイナス)に最大限リンクを伸ばしても見える範囲
ax1.set_ylim(0, (L1+L2)*2)#グラフを表示した時の比率を1:1にするためxと同じ範囲に設定
to_r1 = 1/L1
to_r2 = 1/L2
pointsN1 = L1*L1#線分の分割数
pointsN2 = L2*L2
circle_w = r1*2
circle_h = r1*2
loop_counter = 0 #接触判定の回数（矩形同士の接触判定で接触していると判定されたものを更に細かく接触判定）
hit_counter = 0 #接触した回数
start = time.time()

for theta1 in range(181):
    #ax1.cla()
    rad1 = theta1 * math.pi/180
    delta_x1, delta_y1 =  to_r1*math.cos(rad1), to_r1*math.sin(rad1)
    #ax1.set_xlim(0, (L1+L2)*2)
    #ax1.set_ylim(0, (L1+L2)*2)
    #ax1.add_patch(circle1)
    for theta2 in range(361):
        #ax1.cla()
        points1 = []#線分を分割した各点の座標
        points2 = []
        rad2 = theta2 * math.pi/180


        delta_x2, delta_y2 =  to_r2*math.cos(rad1 + rad2), to_r1*math.sin(rad1 + rad2)
        #ax1.set_xlim(0, (L1+L2)*2)
        #ax1.set_ylim(0, (L1+L2)*2)
        #ax1.add_patch(circle1)

        

        
        x1, y1 =  x0 + L1*math.cos(rad1), y0 + L1*math.sin(rad1)
        x2, y2 =  x1 + L2*math.cos(rad1+rad2), y1 + L2*math.sin(rad1+rad2) 
        
        ##########################四角形同士で判定（未検証）
        Xs = [x0, x1, x2]
        Ys = [y0, y1, y2]
        xmin, xmax, ymin, ymax = 100, 0, 100, 0
        for i in Xs:
            if i < xmin:
                xmin = i
            if xmax < i:
                xmax = i

        for i in Ys:
            if i < ymin:
                ymin = i
            if ymax < i:
                ymax = i
        sikaku_w = abs(xmax - xmin)
        sikaku_h = abs(ymax - ymin)

        colission_possibility = False
        if circle_x1_sikaku_min < xmax+0.5 and xmin-0.5 <= circle_x1_sikaku_max:
            #colission_possibility = True
            if circle_y1_sikaku_min < ymax+0.5 and ymin-0.5 < circle_y1_sikaku_max:
                colission_possibility = True
            
        if colission_possibility == True:
            loop_counter += 1
            ##############################

            ####アニメーション#####################################
            """
            # セグメント（線分）の生成
            s0 = Segment(Point(x0, y0),Point(x1,y1))
            #s1 = Segment(Point(x0, y0),Point(circle_x1, circle_y1))
            #s2 = Segment(Point(x1, y1),Point(circle_x1, circle_y1))

            s3 = Segment(Point(x1, y1),Point(x2, y2))
            #s4 = Segment(Point(x1, y1),Point(circle_x1, circle_y1))
            #s5 = Segment(Point(x2, y2),Point(circle_x1, circle_y1))


            #s1 = Segment((x1, y1),(x2, y2)) # Point()は省略可能
            #s2 = Segment((6, 7),(7,1)) # Point()は省略可能
            #segments = [s0,s1,s2]
            #segments = [s0, s1, s2, s3, s4, s5]
            segments = [s0, s3]

            #plt.plot(middle_x2, middle_y2, color='0.0',marker='.' )
            middle_x1, middle_y1 = (x1-x0)/2, (y1-y0)/2
            middle_x2, middle_y2 = x1 + (x2 - x1)/2, y1 + (y2 - y1)/2
            #x2, y2 = 6, 4
            #middle_x2, middle_y2 = x1 + ((x2-x1)/2) , y1 + ((y2-y1)/2)
            
            # グラフ描画
            for i,s in enumerate(segments) :
                line, = ax1.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y])
                ims.append([line])
                #pn = 'p' + str(i)
                #for p, m in zip( s.points,('',"'") ) :
                    #ax1.plot(p.x, p.y, color='0.0',marker='.')
                    #ax1.text(p.x, p.y + 0.2, pn+m , size=10, horizontalalignment='center', verticalalignment='bottom')
            #ax1.plot(middle_x1, middle_y1, color='0.0',marker='.' )
            #x1.plot(middle_x2, middle_y2, color='0.0',marker='.' )

            #ax1.plot(circle_x1, circle_y1, color='0.0',marker='.' )
            plt.pause(0.00000000000001)
            """
            ############################################################3
            """
            for i in range(pointsN1):
                #points1.append((x0 + i*(pointsN1/x1-x0), y0 + i*(pointsN1/y1-y0)))
                points1.append((x0 + i*delta_x1, y0 + i*delta_y1))
            """
            points1 = [(x0 + i*delta_x1, y0 + i*delta_y1) for i in range(pointsN1)]
            
            """
            for i in range(pointsN2):
                #points2.append((x1 + i*(pointsN2/x2-x1), y1 + i*(pointsN2/y2-y1)))
                points2.append((x1 + i*delta_x2, y1 + i*delta_y2))
            """
            points2 = [(x1 + i*delta_x2, y1 + i*delta_y2) for i in range(pointsN2)]
            #リンク１の接触判定----------------
            hit1 = False
            
            for i in range(pointsN1):
                if hit1 == False:
                    x, y = points1[i]
                    #C = math.sqrt((circle_x1-x)**2+(circle_y1-y)**2)
                    C = (circle_x1-x)**2+(circle_y1-y)**2
                    if   C <= (to_r1 + r1)**2:
                        hit1 = True
                else:
                    break



            #---------------------------------

            #リンク2の接触判定---------------
            if hit1 == False:
                hit2 = False
                for i in range(pointsN2):
                    if hit2 == False:
                        x, y = points2[i]
                        #C = math.sqrt((circle_x1-x)**2+(circle_y1-y)**2) 
                        C = (circle_x1-x)**2+(circle_y1-y)**2 
                        if C <= (to_r2 + r1)**2:
                            hit2 = True
                    else:
                        break

            #--------------------------

            #リンク1またはリンク2のどちらかが接触したら描画
            if(hit1 == True or hit2 == True):
                #print("hit!!!!!!")
                #円の描画#######
                #circle1 = pat.Circle(xy = (circle_x1, circle_y1), radius= r1, color="red")
                #ax1.add_patch(circle1)
                ###################
                hit_counter += 1
                ax2.plot(theta1, theta2, color='0.0',marker='+')


#ani = animation.ArtistAnimation(fig, ims)
#ani.save('anim.gif', writer="imagemagick")
#ani.save('anim.mp4', writer="ffmpeg")
#print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
#plt.show()

#####################################
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
#google colabにて時間検証（2020/04/29)
#図の表示は時間に含まない
#1回目elapsed_time:10.556167602539062[sec]
#2回目elapsed_time:10.513708591461182[sec]
#3回目elapsed_time:9.963657855987549[sec]
#4回目elapsed_time:10.092783212661743[sec]
#5回目elapsed_time:9.989397764205933[sec]
#5回の平均10.2231430054


print("loop_counter",loop_counter)
print("hit_count",hit_counter)
#loop_counter 27093　(矩形の衝突判断で衝突していると判定されたもの)
#hit_count 14737

theta1 = 110 #角度　0度～180度の間で指定
rad1 = theta1 * math.pi/180
theta2 = 270 #0度～360度で設定
rad2 = theta2 * math.pi/180



ax2.plot(theta1,theta2, color="red", marker="+")
x1, y1 =  x0 + L1*math.cos(rad1), y0 + L1*math.sin(rad1)
x2, y2 =   x1 + L2*math.cos(rad1+rad2), y1+  L2*math.sin(rad1+rad2) 
# セグメント（線分）の生成
s0 = Segment(Point(x0, y0),Point(x1,y1))
#s1 = Segment(Point(x0, y0),Point(circle_x1, circle_y1))
#s2 = Segment(Point(x1, y1),Point(circle_x1, circle_y1))

s1 = Segment(Point(x1, y1),Point(x2, y2))
#s4 = Segment(Point(x1, y1),Point(circle_x1, circle_y1))
#s5 = Segment(Point(x2, y2),Point(circle_x1, circle_y1))

#########
Xs = [x0, x1, x2]
Ys = [y0, y1, y2]
xmin, xmax, ymin, ymax = 100, 0, 100, 0
for i in Xs:
    if i < xmin:
        xmin = i
    if xmax < i:
        xmax = i

for i in Ys:
    if i < ymin:
        ymin = i
    if ymax < i:
        ymax = i
sikaku_w = abs(xmax - xmin)
sikaku_h = abs(ymax - ymin)
ax1.plot(xmin-0.5,ymin-0.5,color="blue", marker=".")
ax1.plot(xmax+0.5,ymax+0.5,color="red", marker=".")
ax1.plot(circle_x1_sikaku_min,circle_y1_sikaku_min,color="red", marker=".")
ax1.plot(circle_x1_sikaku_max,circle_y1_sikaku_max,color="red", marker=".")



#########

#s1 = Segment((x1, y1),(x2, y2)) # Point()は省略可能
#s2 = Segment((6, 7),(7,1)) # Point()は省略可能
#segments = [s0,s1,s2]
#segments = [s0, s1, s2, s3, s4, s5]
segments = [s0, s1]
for i,s in enumerate(segments) :
    ax1.plot([s.p1.x, s.p2.x], [s.p1.y, s.p2.y])
for i in range(pointsN1):
    delta_x1, delta_y1 =  to_r1*math.cos(rad1), to_r1*math.sin(rad1)
    circle1 = pat.Circle(xy = (x0 + i*delta_x1, y0 + i*delta_y1), radius= to_r1, color="blue")
    ax1.add_patch(circle1)
for i in range(pointsN2):
    delta_x2, delta_y2 =  to_r2*math.cos(rad1+rad2), to_r2*math.sin(rad1+rad2)
    circle1 = pat.Circle(xy = (x1 + i*delta_x2, y1 + i*delta_y2), radius= to_r2, color="red")
    ax1.add_patch(circle1)
#delta_x2, delta_y2 =  to_r2*math.cos(rad2), to_r1*math.sin(rad2)

plt.show()