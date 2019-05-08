import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

file_name = "forchheimer_L_Lp_dependency_Iter.txt"
data = np.loadtxt(file_name, dtype=np.int, delimiter=",")

# num_steps = 1
L = 0.25*np.arange(11)
Lp = np.arange(2.2, 4.3, 0.2)
L, Lp = np.meshgrid(L, Lp)

# make figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("L")
ax.set_ylabel("Lp")
ax.set_zlabel("$\sharp$")

# import pdb; pdb.set_trace()

ax.plot_surface(L, Lp, data)

file_name = "forchheimer_L_Lp.pdf"
fig.savefig(file_name, bbox_inches='tight')
plt.gcf().clear()