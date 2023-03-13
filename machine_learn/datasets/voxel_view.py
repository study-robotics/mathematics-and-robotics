import binvox_rw
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt

with open('./hoge/32/0.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f, fix_coords=False)

"""
with open('./hoge/128/after_resolution/32_0.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f, fix_coords=False)
"""
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.voxels(model.data, edgecolor='k')

plt.show()