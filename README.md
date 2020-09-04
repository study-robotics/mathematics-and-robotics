# robotics_program

[](ここから運動学・逆運動学-------------------------------------------------------------------------------------)
## 運動学・逆運動学
<details>
  <summary> エネルギー・運動量・角運動量の保存量などを用いて物体の運動を議論する方法 </summary>
  
  ## 運動学
  ロボットアームの「各関節の角度」から「手先の座標」を求める問題
  ## 逆運動学
  ロボットアームの「手先の座標」から「各関節の角度」を求める問題 
  ### プログラム
  |  プログラム名 |  説明  |
  | ---- | ---- |
  | inverse_kinematics.py | 2軸のアームの逆運動学．「現在の各関節の角度」と「手先の目標位置」を入力すると，「各関節の角度」を計算して，図として出力する．|
  | 3link_inverse_kinematics.py | 3軸のアームの逆運動学．「現在の各関節の角度」と「手先の目標位置」を入力すると，「各関節の角度」を計算して，図として出力する．  |
  | anime_inverse_kinematics.py | 2軸のアームの逆運動学．「現在の各関節の角度」と「手先の目標位置」を入力すると，「各関節の角度」を計算して，その過程をmp4で保存．|
  
</details>


[](ここまで運動学・逆運動学-------------------------------------------------------------------------------------)

[](ここから経路計画-------------------------------------------------------------------------------------)

## 経路計画
<details>
  <summary> ロボットの形状や移動モデルを考慮し，スタート地点からゴール地点までの動作の手順を探索すること </summary>
  
  ## ランダムサンプリング
  ### プログラミング
  |　プログラミング名　|　説明　|
  | ---- | ---- |
  | rrt_star.py | RRT&#42;を実行する　|
  
</details>
