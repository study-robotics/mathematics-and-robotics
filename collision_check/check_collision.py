from sympy.geometry import *
import matplotlib.pyplot as plt
import matplotlib.patches as pat
#import matplotlib.animation as animation
import math
import numpy as np
import time
from matplotlib.animation import ArtistAnimation
start = time.time()

class Circle():
    def __init__(self, r, x, y):
        self.type = "circle"
        self.r = r
        self.x = x
        self.y = y

def circle_collision_check(to_r, circle_r, circle_x, circle_y, points, pointsN):
    for i in range(pointsN):
        x, y = points[i]
        #C = math.sqrt((circle_x1-x)**2+(circle_y1-y)**2)
        C = (circle_x-x)**2+(circle_y-y)**2
        if C <= (to_r + circle_r)**2:
            #print("collision!")
            return True
    return False

def exec_collision_check(obstacle, link_length, theta, show_result=False):
    x0, y0 = link_length, link_length #[リンクを最大限伸ばした数値]を原点とする※(以降で導出するx1,x2,y1,y2でマイナスの数値が出ないようにするため)
    rad = theta * math.pi/180
    x1, y1 = x0 + link_length*math.cos(rad), y0 + link_length*math.sin(rad)
    to_r = 1/link_length # collision check num
    delta_x, delta_y =  to_r*math.cos(rad), to_r*math.sin(rad)
    points = []
    pointsN = link_length*link_length #線分の分割数
    
    # create collision check points (on the link)
    for i in range(pointsN):
        points.append((x0+i*delta_x, y0+i*delta_y))
    
    if obstacle.type == "circle":
        result = circle_collision_check(to_r, obstacle.r, obstacle.x, obstacle.y, points, pointsN)
    else:
        print("obstacle type is don't found")

    if show_result==True:
        fig = plt.figure(figsize=(5,5),dpi=96)
        ax = fig.add_subplot(1, 1, 1)
        plt.axes(ax)
        plt.xlabel('x'); plt.ylabel('y')
        ax.set_xlim(0, link_length*2); ax.set_ylim(0, link_length*2)
        
        # draw link range circle
        circle = pat.Circle(xy = (x0, y0), radius=link_length, ec='#000000', fill=False)
        ax.add_patch(circle)
        
        # if collision, link color is red
        if result == True:
            ax.plot([x0, x1], [y0, y1],color="red")
        else:
            ax.plot([x0, x1], [y0, y1],color="green")
        
        # draw obstacle
        if obstacle.type == "circle":
            circle = pat.Circle(xy = (obstacle.x, obstacle.y), radius=obstacle.r, color="blue")
            ax.add_patch(circle)

        plt.show()

    return x0, y0, x1, y1, result

def save_collision_check_animation(obstacle, link_length, check_resolution):
    fig = plt.figure(figsize=(5,5),dpi=96)
    ax = fig.add_subplot(1, 1, 1)
    plt.axes(ax)
    plt.xlabel('x'); plt.ylabel('y')
    ax.set_xlim(0, link_length*2); ax.set_ylim(0, link_length*2) # x, y方向（プラスマイナス)に最大限リンクを伸ばしても見える範囲
    
    artists = [] # animation save array
    print("start save the animation")
    for theta in range(0, 360, check_resolution):
        # ----- obstacle is circle -------------
        if obstacle.type == "circle":
            x0, y0, x1, y1, result = exec_collision_check(obstacle, link_length, theta)
            
            # draw circle object
            circle = pat.Circle(xy = (obstacle.x, obstacle.y), radius=obstacle.r, color="blue")
            ax.add_patch(circle)
        # --------------------------------------
            
        # draw link range of motion circle
        circle = pat.Circle(xy = (x0, y0), radius=link_length, ec='#000000', fill=False)
        ax.add_patch(circle)
            
        # if link and object is collision, link color is red
        if result == True:
            artist =  ax.plot([x0, x1], [y0, y1],color="red")
        else:
            artist =  ax.plot([x0, x1], [y0, y1],color="green")
        artists.append(artist)
    
    anim = ArtistAnimation(fig, artists)
    s = anim.to_jshtml()
    with open( 'anim.html', 'w') as f:
        f.write(s)
    print("finished save the animation")

def main():
    L = 5
    theta =230
    check_resolution = 1

    obstacle_circle_r = 1
    obstacle_circle_x = 2
    obstacle_circle_y = 2
    
    obstacle_circle = Circle(obstacle_circle_r , obstacle_circle_x, obstacle_circle_y)
    #exec_collision_check(obstacle_circle, L, theta, show_result=True)

    save_collision_check_animation(obstacle_circle, L, check_resolution)


if __name__ == '__main__':
    main()
