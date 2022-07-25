import numpy as np

def H_2d(x, y, theta):     
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta),  np.cos(theta), y],
                     [            0,              0, 1]])

link1_length = 2.0
theta1 = 30
theta1 = np.radians(theta1)
link2_length = 3.0
theta2 = 10
theta2 = np.radians(theta2)

# 同時変換行列による順運動学（軸数が増えたり三次元空間になった時でも，同時変換行列を増やすだけで良い）
x0, y0 = 0.0, 0.0
H012 = H_2d(x0, y0, theta1)@H_2d(link1_length, 0.0, theta2)
oe = H012@to_e
print(oe)

# 三角関数による順運動学（直感的に分かりやすが軸数が増えたり三次元空間になったときに，計算が複雑になる）
to_e = np.array([[link2_length],
                 [0.0],
                 [1]])
x2, y2 = link1_length*np.cos(theta1),  link1_length*np.sin(theta1)
xe, ye = link1_length*np.cos(theta1) + link2_length*np.cos(theta1+theta2), link1_length*np.sin(theta1) + link2_length*np.sin(theta1+theta2)

print(xe, ye)

