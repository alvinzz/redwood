import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_aspect('equal')

# ax.plot(x, y, z, label='parametric curve')
x1=[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
y1=[0]*6
z1=[0]*6
ax.scatter(x1, y1, z1)

x2=[-1.5, 1.5, -1.5, 1.5]
y2=[0, 0, 1, 1]
z2=[1]*4
ax.scatter(x2, y2, z2)

x3=[0, 0, 0, 0]
y3=[0, 1, 2, 3]
z3=[2]*4
ax.scatter(x3, y3, z3)

for i in range(6):
    for j in range(4):
        ax.plot([x1[i], x2[j]], [y1[i], y2[j]], [z1[i], z2[j]], c='black', alpha=0.2)

for i in range(4):
    for j in range(4):
        ax.plot([x3[i], x2[j]], [y3[i], y2[j]], [z3[i], z2[j]], c='black', alpha=0.2)

plt.show()
