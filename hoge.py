import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots()
x = np.arange(21)
y = np.random.rand(21)
n = len(x)
ax.set_xlim([0, 20])
ax.set_ylim([0, 1])
ind = 3    # カーソル位置のインデックス
cur_v = ax.axvline(x[ind], color='k', linestyle='--', linewidth=0.5)
cur_h = ax.axhline(y[ind], color='k', linestyle='--', linewidth=0.5)
ax.plot(x, y, "o-",picker=15)
cur_point, = ax.plot(x[ind], y[ind], color='k', markersize=10, marker='o')
ax.set_title('index: {}, x = {}, y = {}'.format(
                                ind, round(x[ind], 4), round(y[ind], 4)))


def on_key(event):
    global ind
    if event.key == 'right':
        move = 1
    elif event.key == 'left':
        move = -1
    else:
        return

    # インデックスの更新
    ind += move
    ind %= n  # グラフの端に行くと戻るようにする

    # カーソルとタイトルの更新
    cur_v.set_xdata(x[ind])
    cur_h.set_ydata(y[ind])
    cur_point.set_data(x[ind], y[ind])
    ax.set_title('index: {}, x = {}, y = {}'.format(
                                    ind, round(x[ind], 4), round(y[ind], 4)))
    plt.draw()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()