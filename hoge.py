import numpy as np
import matplotlib.pyplot as plt
import ipywidgets
from mpl_toolkits.mplot3d import Axes3D

def make_homogeneous_transformation_matrix(link_length, theta):
    """
    2次元平面における同次変換行列を求める
    
    Parameters
    ----------
    link_length : float
        リンクの長さ
    theta : float
        回転角度(rad)

    Returns
    -------
    T : numpy.ndarray
        同時変換行列
    """
    return np.array([[np.cos(theta), -np.sin(theta), link_length*np.cos(theta)],
                     [np.sin(theta),  np.cos(theta), link_length*np.sin(theta)],
                     [            0,              0,                        1]])

def draw_link_coordinate(ax, matrix, axes_length):   
    """
    2次元の変換行列より単位ベクトルを描画
    
    Parameters
    ----------
    ax : matplotlib.axes._axes.Axes
        描画用
    matrix: numpy.array
        2次元の変換行列
    axes_length : float
        各軸方向の単位ベクトルの長さ

    Returns
    -------
    なし
    """
    # x方向の単位ベクトル
    unit_x = matrix@np.array([[axes_length],
                              [0],
                              [1]])

    # y方向の単位ベクトル
    unit_y = matrix@np.array([[0],
                              [axes_length],
                              [1]])
                   
    x = matrix[0][2]; y = matrix[1][2]
    
    # x方向の単位ベクトルを描画
    ax.plot([x, unit_x[0][0]], [y, unit_x[1][0]], "o-", color="red", ms=2) 
    # y方向の単位ベクトル
    ax.plot([x, unit_y[0][0]], [y, unit_y[1][0]], "o-", color="green", ms=2) 

def generate_vbox_text_widget(link_num):
    """
    text widgetsをlink_num個作成 -> Vboxに格納して縦に並べる（範囲は-180〜180）
    
    Parameters
    ----------
    link_num : int
        ロボットのリンクの数

    Returns
    -------
    vox_text_widgets : ipywidgets.widgets.widget_box.VBox
        text widgetsをnum個，縦に並べたVBox
    """
    text_widgets = []
    for i in range(link_num):
      text_widgets.append(ipywidgets.FloatText(min=-180.0, max=180.0))
    vox_text_widgets = ipywidgets.VBox(text_widgets)
    return vox_text_widgets

def generate_vbox_slider_widget(link_num):
    """
    slider widgetsをlink_num個作成 -> Vboxに格納して縦に並べる．（範囲は-180〜180）
    
    Parameters
    ----------
    link_num : int
        ロボットのリンクの数

    Returns
    -------
    vox_slider_widgets : ipywidgets.widgets.widget_box.VBox
        slider widgetsをnum個，縦に並べたVBox
    """
    slider_widgets = []
    for i in range(link_num):
      slider_widgets.append(ipywidgets.FloatSlider(value=0.0, min=-180.0, max=180.0, description = "param"+str(i+1), disabled=False))
    vox_slider_widgets = ipywidgets.VBox(slider_widgets)
    return vox_slider_widgets


def link_slider_and_text(box1, box2, link_num):
    """
    Box内の複数のwidetを連携させる（二つのbox内のwidgetの数が同じである必要あり）
    
    Parameters
    ----------
    box1 : ipywidgets.widgets.widget_box.VBox
        boxの名前
    box2 : ipywidgets.widgets.widget_box.VBox
        boxの名前
    link_num : int
        linkの数
    """
    for i in range(link_num):
      ipywidgets.link((box1.children[i], 'value'), (box2.children[i], 'value'))

def draw_interactive(link_num):
    """
    結果をアニメーションで表示
    Parameters
    ----------
    link_num : int
        linkの数
    """
    # slider widgetを作成
    posture_sliders = generate_vbox_slider_widget(link_num)
    # text widgetを作成
    posture_texts = generate_vbox_text_widget(link_num)

    # slider widget と　posture widget を横に並べる
    slider_and_text = ipywidgets.Box([posture_sliders, posture_texts])

    # slider wiget と text widget を連携
    link_slider_and_text(posture_sliders, posture_texts, link_num)

    # リセットボタン
    reset_button = ipywidgets.Button(description = "Reset")
    # 姿勢のリセットボタン
    def reset_values(button):
        for i in range(link_num):
            posture_sliders.children[i].value = 0.0
    reset_button.on_click(reset_values)

    # main文にslider widgetsの値を渡す
    params = {}
    for i in range(link_num):
        params[str(i)] = posture_sliders.children[i]
    final_widgets = ipywidgets.interactive_output(main, params)
    
    display(slider_and_text, reset_button, final_widgets)

def main(*args, **kwargs):

    params = kwargs
    
    ################ ここから同次変換行列による順運動学の処理（メイン部分） #############################
    # 各linkの長さ（不変）
    l1 = 4.0
    l2 = 4.0

    # 回転角度（可変）
    theta1 = params["0"]
    theta2 = params["1"]

    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)

    # 同次変換行列
    T12 = make_homogeneous_transformation_matrix(l1, theta1) # T12：link「1」座標系 -> link「2」座標系
    T2e = make_homogeneous_transformation_matrix(l2, theta2) # T2e：link「2」座標系 -> 「e」nd effector座標系

    # 「link1座標系」から「n番目のlink座標系」までの同次変換行列を定義
    T1e = T12@T2e # T1e：link「1」座標系 -> 「e」nd effector座標系
    
    # link1座標系の原点を基準とした時の，各linkの原点座標
    x1, y1 = 0, 0

    # link1座標系 -> link2座標系への変換
    o2 = T12@np.array([[x1],
                       [y1],
                       [1]])
    x2, y2 = o2[0][0], o2[1][0]

    # link1座標系 -> end effector座標系への変換
    oe = T1e@np.array([[x1],
                       [y1],
                       [1]])
    xe, ye = oe[0][0], oe[1][0]
    ###################### ここまで同次変換行列による順運動学の処理 ######################3


    ######### 以下，描画関連 #####################################################
    fig = plt.figure(figsize=(5,10))
    ax1 = fig.add_subplot(2,1,1)
    # 各linkの描画
    ax1.plot([x1, x2], [y1, y2], "-", color="tomato", ms=6) # link1
    ax1.plot([x2, xe], [y2, ye], "-", color="lightgreen", ms=6) # link2

    # 各linkの座標軸を描画
    axes_length = (l1+l2)*0.1 # 各座標系の軸の長さは「リンクの長さ×0.1」に設定
    draw_link_coordinate(ax1, np.array([[1, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 1]]), axes_length)
    draw_link_coordinate(ax1, T12, axes_length)
    draw_link_coordinate(ax1, T1e, axes_length)
    
    # 範囲設定
    ax1.set_xlim(-(l1+l2), l1+l2)
    ax1.set_ylim(-(l1+l2), l1+l2)

    # 軸ラベル
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    plt.show()

draw_interactive(2)