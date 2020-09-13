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
## 経路計画の主な手順 
経路計画の主な手順を説明する．　　

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/image/1.JPG" width="500px">　　

最初にロボット（移動ロボット等）のスタート地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">(=<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{1}">)とゴール地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">を設定する．なお，今回の説明では<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">の座標と<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">の座標が既知であると仮定する．  
　　　　
<br>

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/image/2.JPG" width="500px">　

次に，ノード（スタートからゴールまでの中継地点）を設置して，各ノードを繋げて経路候補を生成する．※ノードの設置方法・つなぎ方は使用するアルゴリズムによって異なる．  

<br>

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/image/3.JPG" width="500px">　

最後に，生成された経路候補から最も最適な経路を選択する．この時，「コスト＝経路の距離」にした場合は単純に「最短距離=最適経路」になる．例えば，「遠回りをしてもいいからできるだけ障害物が少ない経路を生成したい」と考えた場合「障害物が少ない経路を選んだらコストが小さくなる」というような重み付けをすればよい．
<a id="random_sampling"></a> 
## ランダムサンプリング
ノードをランダムに配置しながらゴールを目指す方法

[](ここからRRT---------------------------------------------------------------------------------------------------------------------)
<a id="rrt"></a>
## RRT
RRTの手順を説明する．

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/1.JPG" width="500px">  

(1) ロボット（移動ロボット等）のスタート地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">(=<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{1}">)とゴール地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">を設定する．なお，今回の説明では<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">の座標と<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">の座標が既知であると仮定する．  

<br>

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/2.JPG" width="500px">
(2) 経路計画を行う範囲内のランダムな場所に点を打つ（以降これをrandom pointと呼ぶ）．※ここでは，説明のため上図では既にノードがいくつか設置されています．  

<br>

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/3.JPG" width="500px">

(3) random pointと「その時点で存在する全てのノード」との間の直線距離を計算する．計算の結果，random pointと一番近い位置にあるノードをnearest nodeとする．上図の場合，<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{3}">がnearset nodeになる．

<br>

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/4.JPG" width="500px">

(4) nearest nodeからrandom point方向に定数<img src="https://latex.codecogs.com/gif.latex?\epsilon">だけ進んだ場所に新しいノードnew nodeを設置する．※定数<img src = "https://latex.codecogs.com/gif.latex?\epsilon">は任意の数値．

<br>

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/5.JPG" width="500px">

ここで，上図のように「nearest nodeとnew nodeの間の直線上」に障害物があった場合，ノードは設置せずに(1)に戻る．

<br>

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/6.JPG" width="500px">

(5) nearset nodeとnew nodeを繋ぐ（経路に追加する）．

<br>

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/7.JPG" width="500px">

(6) (1)から(4)をゴールまでの経路が見つかるまで繰り返す．

<br>

<img src = "https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt/image/7.JPG" width="500px">

(7) スタートからゴールを繋ぐ経路が見つかったら経路探索終了．

<br>

上記の手順から分かるように，RRTでは単純にランダムにノードを設置しているシンプルな手法なため最適な経路が生成される保証がない（ジグザグな経路が生成されることも多い）．
[](ここまでRRT---------------------------------------------------------------------------------------------------------------------)

[](ここからRRT*---------------------------------------------------------------------------------------------------------------------)
<a id="rrt_star"></a> 
## RRT&#42; 
RRTを改善した手法．RRTと違い，最適な経路が生成される．
[](ここまでRRT*---------------------------------------------------------------------------------------------------------------------)

<a id="random_sampling_pro"></a> 
### ランダムサンプリングのプログラム
|　プログラミング名　|　説明　|
| ---- | ---- |
| rrt_star.py | RRT&#42;を実行する　|
