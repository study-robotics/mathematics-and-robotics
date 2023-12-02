import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ロボットアームのパラメータ
link1_length = 1.0
link2_length = 1.0

# 初期位置
theta1 = np.pi / 4
theta2 = np.pi / 4

# シミュレーションパラメータ
k_p = 10.0  # 位置制御ゲイン
k_d = 1.0   # 速度制御ゲイン
desired_position = np.array([1.5, 1.5])  # 目標位置

# シミュレーションの時間ステップ
dt = 0.02
total_time = 10.0

# Matplotlibの初期化
fig, ax = plt.subplots()
ax.set_xlim(-2, 3)
ax.set_ylim(-1, 4)

# ロボットアームの初期化
link1, = ax.plot([], [], lw=4)
link2, = ax.plot([], [], lw=4)
end_effector, = ax.plot([], [], 'ro', markersize=10)

# 逆運動学関数
def inverse_kinematics(x, y):
    # 逆運動学を解く
    L = np.sqrt(x**2 + y**2)
    phi2 = np.arccos((x**2 + y**2 - link1_length**2 - link2_length**2) / (2 * link1_length * link2_length))
    theta2 = np.pi - phi2
    alpha = np.arctan2(y, x)
    beta = np.arccos((x**2 + y**2 + link1_length**2 - link2_length**2) / (2 * link1_length * L))
    theta1 = alpha - beta

    return theta1, theta2

# インピーダンス制御
def impedance_control(current_position, desired_position, current_velocity):
    error = desired_position - current_position
    control_force = k_p * error - k_d * current_velocity
    return control_force

# シミュレーションステップ
def update(frame):
    global theta1, theta2

    # 逆運動学を使用して現在の位置を計算
    x = link1_length * np.cos(theta1) + link2_length * np.cos(theta1 + theta2)
    y = link1_length * np.sin(theta1) + link2_length * np.sin(theta1 + theta2)
    
    # インピーダンス制御を使用して関節角速度を計算
    control_force = impedance_control(np.array([x, y]), desired_position, np.array([0.0, 0.0]))
    theta1_dot, theta2_dot = np.linalg.solve(np.array([[link1_length, link2_length * np.cos(theta2)],
                                                      [0, link2_length * np.sin(theta2)]]),
                                           control_force)
    
    # 関節角を更新
    theta1 += theta1_dot * dt
    theta2 += theta2_dot * dt
    
    # ロボットアームを描画
    x1 = 0
    y1 = 0
    x2 = link1_length * np.cos(theta1)
    y2 = link1_length * np.sin(theta1)
    x3 = x
    y3 = y

    link1.set_data([x1, x2], [y1, y2])
    link2.set_data([x2, x3], [y2, y3])
    end_effector.set_data(x, y)
    
    return link1, link2, end_effector

# アニメーションを作成
ani = FuncAnimation(fig, update, frames=int(total_time / dt), blit=True, interval=dt * 1000)

plt.show()