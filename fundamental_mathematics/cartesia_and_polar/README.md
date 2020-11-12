# 数学の基礎
ロボット工学の色々な分野で用いられる基礎的な数学について記述する．
# 目次
<details>
   <summary>目次</summary>
   
* [直交座標系と極座標系](#cartesia_and_polar)
    * [概要](#cartesia_and_polar_method)
    * [2次元座標系における座標変換の手順](#cartesia_and_polar_2d)
    * [2次元座標系における変換の具体例](#cartesia_and_polar_2d_ex)
        * [直交座標系→極座標系](#cartesia_and_polar_2d_c_to_p)
        * [極座標系→直交座標系](#cartesia_and_polar_2d_p_to_c)
        * [応用例](#cartesia_and_polar_2d_app)
</details>

[](ここから直交座標系と極座標系----------------------------------------------------------------------------------------)
<a id="cartesia_and_polar"></a> 
# 直交座標系と極座標系

<a id="cartesia_and_polar_method"></a> 
## 概要
座標の表現方法には様々な種類があるが，ここでは直交座標系と極座標系について記述する．  

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/fundamental_mathematics/cartesia_and_polar/image/1.JPG" width=300px>

最初に上図のような最も一般的（小学校などで習う）な座標系を「直交座標系」という．この座標系は，2次元平面上で横軸を<img src="https://latex.codecogs.com/gif.latex?x">，
縦軸を<img src="https://latex.codecogs.com/gif.latex?y">とした時に(<img src="https://latex.codecogs.com/gif.latex?x">, <img src="https://latex.codecogs.com/gif.latex?y">)
で座標上の点の位置を示すことができる．  
<br>
<br>

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/fundamental_mathematics/cartesia_and_polar/image/2.JPG" width=300px>

次に上図のように「座標上の点の原点からの直線距離<img src="https://latex.codecogs.com/gif.latex?r">」と「<img src="https://latex.codecogs.com/gif.latex?x">軸から反時計周り方向に測った
角度<img src="https://latex.codecogs.com/gif.latex?\theta">」によって
(<img src="https://latex.codecogs.com/gif.latex?r">, <img src="https://latex.codecogs.com/gif.latex?\theta">)の形で座標上の点の位置を示す座標系を
「極座標系」という．  
<a id="cartesia_and_polar_2d"></a> 
## 2次元座標系における座標変換の手順
直交座標系と極座標系は互いの座標を変換することが可能である．2次元座標系の場合，座標上の点が直交座標系で(<img src="https://latex.codecogs.com/gif.latex?x">, <img src="https://latex.codecogs.com/gif.latex?y">)，極座標系で(<img src="https://latex.codecogs.com/gif.latex?r">, <img src="https://latex.codecogs.com/gif.latex?\theta">)と表された時，極座標系→直交座標系に変換する場合  
<img src = "https://latex.codecogs.com/gif.latex?x=r&space;\cos&space;\theta">      
<img src = "https://latex.codecogs.com/gif.latex?y=r&space;\sin&space;\theta">  
で変換することが可能である．  
直交座標系→極座標系に変換するには，<img src = "https://latex.codecogs.com/gif.latex?x=r&space;\cos&space;\theta">，<img src = "https://latex.codecogs.com/gif.latex?y=r&space;\sin&space;\theta">の関係性を式変形して，  
<img src ="https://latex.codecogs.com/gif.latex?r=\sqrt{x^{2}&plus;y^{2}}">  
<img src = "https://latex.codecogs.com/gif.latex?\theta=\tan^{-1}&space;\left(\frac{y}{x}\right)">   
とすることで変換可能である．  

<a id="cartesia_and_polar_2d_ex"></a> 
## 2次元座標系における変換の具体例
<a id="cartesia_and_polar_2d_c_to_p"></a> 
### 直交座標系→極座標系
<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/fundamental_mathematics/cartesia_and_polar/image/3.JPG" width=300px>

具体例として上図の直交座標系で(3,4)の位置に点がある場合を考える．  
<br>  
<br>

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/fundamental_mathematics/cartesia_and_polar/image/4.JPG" width=300px>

<img src="https://latex.codecogs.com/gif.latex?r">，<img src="https://latex.codecogs.com/gif.latex?\theta">はそれぞれ下記の様に求められる．  
<img src="https://latex.codecogs.com/gif.latex?r=\sqrt{3^{2}&plus;4^{2}}=5">  
<img src="https://latex.codecogs.com/gif.latex?\theta=\tan^{-1}\left(\frac{4}{3}\right)&space;\risingdotseq&space;53.1">  

<a id="cartesia_and_polar_2d_p_to_c"></a> 
### 極座標系→直交座標系
<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/fundamental_mathematics/cartesia_and_polar/image/5.JPG" width=300px>

次に，上図の極座標系で(8, 45)の位置に点がある場合を考える．  
<br>  
<br>
<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/fundamental_mathematics/cartesia_and_polar/image/6.JPG" width=300px>  

<img src="https://latex.codecogs.com/gif.latex?x">，<img src="https://latex.codecogs.com/gif.latex?y">はそれぞれ下記の様に求められる．  
<img src="https://latex.codecogs.com/gif.latex?x&space;=&space;8\cos\left({45}\right)\risingdotseq&space;5.65">  
<img src="https://latex.codecogs.com/gif.latex?y&space;=&space;8\sin\left({45}\right)\risingdotseq&space;5.65">  
このような極座標系→直交座標系への変換は[運動学](https://github.com/study-robotics/mathematics-and-robotics/tree/master/kinematics)で用いられる．  
<a id="cartesia_and_polar_2d_app"></a> 
### 応用例（後で修正予定）
次に，応用例を説明する．  

<img src="https://github.com/study-robotics/mathematics-and-robotics/blob/master/fundamental_mathematics/cartesia_and_polar/image/7.JPG" width=300px>  

上図のように直線上の点を求めたいときも極座標系→直交座標系の変換は利用することが可能である．例として， [極座標系→直交座標系の例](#cartesia_and_polar_2d_p_to_c)のように変換した後に，
直線上にある点の座標を求める方法を考える．ここで，直交座標系で「原点から(5.65, 5.65)方向に4だけ進んだ位置の座標」を求めたい場合

<img src="https://latex.codecogs.com/gif.latex?x">，<img src="https://latex.codecogs.com/gif.latex?y">はそれぞれ下記のように求められる．  
<img src="https://latex.codecogs.com/gif.latex?x&space;=&space;4\cos\left({45}\right)\risingdotseq&space;2.83">  
<img src="https://latex.codecogs.com/gif.latex?y&space;=&space;4\sin\left({45}\right)\risingdotseq&space;2.83">  
このような手法は，直線上の衝突判定を行う時に用いられる．  
[](ここまで直交座標系と極座標系----------------------------------------------------------------------------------------)



