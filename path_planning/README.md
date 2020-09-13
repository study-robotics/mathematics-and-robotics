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
### RRTの手順

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
## RRT&#42; の手順

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/1.JPG" width="500px">

(1) ロボット（移動ロボット等）のスタート地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">(=<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{1}">)とゴール地点<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">を設定する．なお，今回の説明では<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{start}">の座標と<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{goal}">の座標が既知であると仮定する．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/2.JPG" width="500px">

(2) 経路計画を行う範囲内のランダムな場所に点を打つ（以降これをrandom pointと呼ぶ）．※ここでは，説明のため上図では既にノードがいくつか設置されています．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/3.JPG" width="500px">

(3) random pointと「その時点で存在する全てのノード」との間の直線距離を計算する．計算の結果，random pointと一番近い位置にあるノードをnearest nodeとする．上図の場合，<img src ="https://latex.codecogs.com/gif.latex?\mathbf{p}_{3}">がnearset nodeになる．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/4.JPG" width="500px">

(4) nearest nodeからrandom point方向に定数<img src = "https://latex.codecogs.com/gif.latex?\epsilon">だけ進んだ場所に新しいノードnew nodeを設置する．※定数<img src = "https://latex.codecogs.com/gif.latex?\epsilon">は任意の数値．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/5.JPG" width="500px">

(5) new nodeを中心点とした半径<img src="https://latex.codecogs.com/gif.latex?r">の円を作成する．new nodeと「半径<img src="https://latex.codecogs.com/gif.latex?r">の円内にあるノード」の距離を測定する．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/6.JPG" width="500px">

(6) (5)で測定した距離の中で**最も距離が短い**「半径<img src = "https://latex.codecogs.com/gif.latex?r">の円内にあるノード（上図では<img src = "https://latex.codecogs.com/gif.latex?\mathbf{p}_{8}">になる）」とnew nodeを接続する．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/7.JPG" width="500px">

(7) **残りの**「半径<img src="https://latex.codecogs.com/gif.latex?r">の円内にあるノード（上図では<img src = "https://latex.codecogs.com/gif.latex?\mathbf{p}_{7}">になる）」の親ノードを「new nodeに変更したときの距離（上図の紫色の経路）」と「変更しなかったときの経路（上図の朱色の経路）」を比較する．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/8.JPG" width="500px">

(8) (6)で親ノードを「new nodeに変更したときの距離」の方が短かった場合**残りの**「半径<img src="https://latex.codecogs.com/gif.latex?r">の円内にあるノード」の親ノードをnew nodeに繋ぎなおす．（上図の場合，<img src = "https://latex.codecogs.com/gif.latex?\mathbf{p}_{7}">の親ノードを<img src = "https://latex.codecogs.com/gif.latex?\mathbf{p}_{6}">から<img src = "https://latex.codecogs.com/gif.latex?\mathbf{p}_{8}">に繋ぎ直している．）  
<br>

(9) (2)～(8)を任意の回数繰り返す．  
<br>

<img src ="https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/image/9.JPG" width="500px">

(10) 作成された経路の中から最も最適な経路を選択する．  
<br>

上記の説明のようにRRT&#42;では，繋ぎなおす作業(rewire)を行っているため設置するノードの総数（=反復回数）が多ければ多いほど最適な経路が生成される．
ただし，rewireを行っているためRRTに比べて処理時間が2倍程度掛かる．
[](ここまでRRT*---------------------------------------------------------------------------------------------------------------------)


<a id="random_sampling_pro"></a> 
### ランダムサンプリングのプログラム
|　プログラミング名　|　説明　|
| ---- | ---- |
| [rrt_star.py](https://github.com/study-robotics/mathematics-and-robotics/blob/master/path_planning/random_sampling/rrt_star/rrt_star.py) | RRT&#42;を実行する．（AtsushiSakai氏のプログラムにコメントを追記したもの）　|
