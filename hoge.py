import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, 0.1) #-5から5まで0.1区切りで配列を作る
y = np.sin(x) #配列xの値に関してそれぞれsin(x)を求めてy軸の配列を生成

plt.plot(x,y) # この場合のplot関数の第一引数xは、x軸に対応し、第二引数のyがy軸にあたります。
plt.show()