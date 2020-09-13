#!/usr/bin/python
# -*- coding: utf-8 -*-
u"""
Copyright (c) 2016 Atsushi Sakai
Released under the MIT license
https://github.com/AtsushiSakai/PythonRobotics/blob/master/LICENSE
"""

import random
import math
import copy
import numpy as np


class RRT():
    u"""
    Class for RRT Planning
    """

    def __init__(self, start, goal, obstacleList, randArea,
                 expandDis=0.5, goalSampleRate=20, maxIter=1000):
        u"""
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]
        """
        self.start = Node(start[0], start[1]) #ノードのスタート地点 Node(x[0],y[0]) 
        self.end = Node(goal[0], goal[1])#ノードのゴール地点 Node(x[0],y[0]) 
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter #繰り返し回数の最大

    def Planning(self, animation=True):
        u"""
        Pathplanning
        animation: flag for animation on or off
        """
        animation = False

        iter = 1
        self.nodeList = [self.start]
        for i in range(self.maxIter):
            rnd = self.get_random_point()
            nind = self.GetNearestListIndex(self.nodeList, rnd)#点と最も距離が近いノード（全ノードと比較して）のインデックスを格納

            newNode = self.steer(rnd, nind)#新ノード（点と一番近い位置にある親ノードを結びその親ノードからexpandDis=0.5の所に生成したノード）保存※rrtではこれをそのままnewNodeとする
            print(newNode.cost)

            if self.__CollisionCheck(newNode, obstacleList):#新ノードが障害物上になかったら
                nearinds = self.find_near_nodes(newNode)#新ノードの近傍（半径r）にあるノードのインデックスを保存
                newNode = self.choose_parent(newNode, nearinds)#上記で格納した近傍ノードから，最もコストが少なくなる親ノードを探してそれを新たな親ノードとする
                self.nodeList.append(newNode)
                self.rewire(newNode, nearinds)

            if animation:
                self.DrawGraph(rnd)

            """
            #最新のノードがゴールの近くにあったら終了
            if self.isNearGoal(newNode):
                print("total Iter",iter)
                break
            """
            iter += 1

        # generate coruse
        lastIndex = self.get_best_last_index()
        path = self.gen_final_course(lastIndex)
        return path

    def choose_parent(self, newNode, nearinds):
        if len(nearinds) == 0:#r近傍にノードが存在しない場合，何も操作せずに新ノードを返す(親ノードがそのまま)=rrtと同じ結果
            return newNode

        dlist = []
        for i in nearinds:
            dx = newNode.x - self.nodeList[i].x 
            dy = newNode.y - self.nodeList[i].y 
            d = math.sqrt(dx ** 2 + dy ** 2)#r近傍のノードと新しいノードとの間の距離
            theta = math.atan2(dy, dx) #r近傍のノードと新しいノードとの間の角度
            if self.check_collision_extend(self.nodeList[i], theta, d):#r近傍のノードと新しいノードとの間に障害物がなかったら距離をlistに追加
                dlist.append(self.nodeList[i].cost + d)
            else:
                dlist.append(float("inf"))#float(inf)無限大の浮動小数点(障害物があると認識=無限大にすることで候補から除外)

        mincost = min(dlist)#新ノードとそのr近傍にあるノードの中で一番近いノードとの距離(=コスト)を格納
        minind = nearinds[dlist.index(mincost)]#上記のmincostのインデックスを格納

        if mincost == float("inf"):#mincost="inf"つまり，新ノードとそのr近傍にあるノード全ての間において障害物があった場合micost="inf"となり，newNodeがそのまま返される
            print("mincost is inf")
            return newNode

        newNode.cost = mincost#上記で求めた親ノードと新ノードとの距離（コスト）をnewNode.costに格納
        newNode.parent = minind#親ノードのインデックスを格納

        return newNode

    def steer(self, rnd, nind):

        # expand tree
        nearestNode = self.nodeList[nind] #ランダムに生成した点に最も近い場所にあるノードを保存
        theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)#点とnearestNodeの間の角度を計算
        newNode = copy.deepcopy(nearestNode)#完全なコピー
        newNode.x += self.expandDis * math.cos(theta)#新ノードのx=伸ばしたい距離（定数）*cos(theta)
        newNode.y += self.expandDis * math.sin(theta)#新ノードのy=伸ばしたい距離（定数）*sin(theta)

        newNode.cost += self.expandDis #新ノードのコスト=総距離
        newNode.parent = nind #newNodeの親ノードのインデックスを格納
        return newNode

    def get_random_point(self):

        if random.randint(0, 100) > self.goalSampleRate:#random.int(0,100):0〜100の間で乱数を生成
            #ランダムな場所に点を生成（この点を伸ばしたどこかに新ノードが生成される）
            rnd = [random.uniform(self.minrand, self.maxrand), #random.uniform()（任意の範囲の浮動小数点数）
                   random.uniform(self.minrand, self.maxrand)]
        else:  # goal point sampling
            rnd = [self.end.x, self.end.y]

        return rnd

    def get_best_last_index(self):

        disglist = [self.calc_dist_to_goal(
            node.x, node.y) for node in self.nodeList]#全てのノードとゴール地点との間の2次元ノルム(ユーグリッド距離)
        goalinds = [disglist.index(i) for i in disglist if i <= self.expandDis]#ノードとゴール地点との距離がexpandDis=0.5より小さいノードのインデックス
        print(goalinds)

        mincost = min([self.nodeList[i].cost for i in goalinds])#ゴールとの距離がexpandDis=0.5以下の時のノードの中で一番コストが小さいノードのコストを保存
        for i in goalinds:
            if self.nodeList[i].cost == mincost:#上記をfloat→intに変更?
                return i

        return None

    def gen_final_course(self, goalind):
        path = [[self.end.x, self.end.y]]#ゴールの座標
        while self.nodeList[goalind].parent is not None:#ゴールに最近傍のノード→親ノード→親ノード→親ノード...→最初のノードと戻っていき最適な通路を結ぶ
            node = self.nodeList[goalind] #1週目：ゴールに最近傍のノードを選択，2週目：1週目のノードの親ノードを選択...
            path.append([node.x, node.y]) #上記のノードのpathを追加
            goalind = node.parent #ノード＝現在ノードの親ノード
        path.append([self.start.x, self.start.y])
        return path

    def calc_dist_to_goal(self, x, y):
        return np.linalg.norm([x - self.end.x, y - self.end.y])#全てのノードとゴール地点との間の2次元ノルム(ユーグリッド距離)

    def find_near_nodes(self, newNode):
        nnode = len(self.nodeList)
        #r = R*(log*nnode/nnode)**(1/d)
        #R:parameter, N:number of node, d:dimension
        #In this case → R=50.0, N:incrase over time, d=2
        #Nが増えれば半径rは小さくなる(初めはノードの数が少ないのでrが大きくないと効率が悪いが，ノードが多くなるとrが小さくてもノードが見つかるようになるため)
        #50.0 * (math.log(nnode)/nnode)**(1/2)
        #(A)**(1/2) → math.sqrt(A) なので
        r = 50.0 * math.sqrt((math.log(nnode) / nnode))
        #  r = self.expandDis * 5.0
        dlist = [(node.x - newNode.x) ** 2 +
                 (node.y - newNode.y) ** 2 for node in self.nodeList]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds

    def rewire(self, newNode, nearinds):
        nnode = len(self.nodeList)
        for i in nearinds:
            nearNode = self.nodeList[i]

            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y

            d = math.sqrt(dx ** 2 + dy ** 2)#「newNode」と「残りのr近傍ノード」との間の距離（コスト）
            
            scost = newNode.cost + d#「newNodeのコスト」＋「newNodeと残りのr近傍ノードとの間の距離（コスト）」

            if nearNode.cost > scost:#「r近傍ノードの元々のコスト」が「newNodeと残りのr近傍ノードを繋げた距離（コスト）」より上回った時，「r近傍ノードの親ノード」を「newNode」に変更する（繋ぎ直す）
                theta = math.atan2(dy, dx)
                if self.check_collision_extend(nearNode, theta, d):#「newNode」と「残りのr近傍ノード」の間を結ぶ直線上に障害物がなかったら
                    nearNode.parent = nnode - 1#「newNode」を親ノードとする
                    nearNode.cost = scost#nearNodeのコストを「newNodeを親にした時のコスト」に変更する

    def check_collision_extend(self, nearNode, theta, d):

        tmpNode = copy.deepcopy(nearNode)

        for i in range(int(d / self.expandDis)):#新しいノードとそのr近傍のノードとの間の距離をexpandDis間隔で検証した時に，間に障害物があったらFalseを返す
            tmpNode.x += self.expandDis * math.cos(theta)
            tmpNode.y += self.expandDis * math.sin(theta)
            if not self.__CollisionCheck(tmpNode, obstacleList):
                return False

        return True

    def DrawGraph(self, rnd=None):
        u"""
        Draw Graph
        """
        import matplotlib.pyplot as plt
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))#ランダムな場所に生成した点と，全てのノードを比較して一番距離が近いノードのインデックスを保存する

        return minind

    def __CollisionCheck(self, node, obstacleList):
        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = dx * dx + dy * dy #障害物とノードとの間の距離
            if d <= size ** 2: #ノードが障害物に重なっている場所にあったら，Falseを返す（そこにはノードは生成しない）
                return False  # collision

        return True  # safe
    
    #最新のノードがゴールの近くにあるか
    def isNearGoal(self, node):
        d = self.calc_dist_to_goal(node.x, node.y)
        if d < self.expandDis:
            return True
        return False


class Node():
    u"""
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


if __name__ == '__main__':
    print("Start rrt start planning")
    import matplotlib.pyplot as plt
    # ====Search Path with RRT====
    obstacleList = [#障害物の一覧
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt = RRT(start=[0, 0], goal=[5, 10],
              randArea=[-2, 15], obstacleList=obstacleList)
    path = rrt.Planning(animation=True)

    # Draw final path
    rrt.DrawGraph()
    plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    plt.grid(True)
    plt.pause(0.01)  # Need for Mac
    plt.show()
