[](ここから経路計画-------------------------------------------------------------------------------------)

# 経路計画
ロボットの形状や移動モデルを考慮し，スタート地点からゴール地点までの動作の手順を探索すること 
# 目次
* [経路計画の主な手順](#path_plan_method)
* [ランダムサンプリング](#random_sampling)
    * [RRT](#rrt)
    * [RRT&#42;](#rrt_star)
    * [ランダムサンプリングのプログラム](#random_sampling_pro)

<a id="path_plan_method"></a>
## 経路計画の主な手法  
経路計画の主な手順を説明する．　　

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/image/1.JPG" width="500px">　　

最初にロボット（移動ロボット等）のスタート地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">(=<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{1}">)とゴール地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">を設定する．なお，今回の説明では<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">の座標と<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">の座標が既知であると仮定する．  

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/image/2.JPG" width="500px">　

次に，ノード（スタートからゴールまでの中継地点）を設置して，各ノードを繋げて経路（候補）を生成する．※ノードの設置方法・つなぎ方は使用するアルゴリズムによって異なる．  

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/image/3.JPG" width="500px">　

最後に，生成された経路候補から最も最適な経路を選択する．この時，「経路の距離=コスト」にした場合は単純に「最短距離=最適経路」になる．例えば，「遠回りをしてもいいからできるだけ障害物が少ない経路を生成したい」と考えた場合「障害物が少なければ少ないほどコストが小さくなる」というような重み付けをすればよい．
<a id="random_sampling"></a> 
## ランダムサンプリング
ノードをランダムに配置しながらゴールを目指す方法

<a id="rrt"></a>
## RRT

<a id="rrt_star"></a> 
## RRT&#42; 
RRTを改善した手法．RRTと違い，最適な経路が生成される．

<a id="random_sampling_pro"></a> 
### ランダムサンプリングのプログラム
|　プログラミング名　|　説明　|
| ---- | ---- |
| rrt_star.py | RRT&#42;を実行する　|
