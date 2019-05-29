import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

num_steps = 5
L = np.linspace(0, 2)
Lp = np.arange(2.2, 4.3, 0.1)
Lp, L = np.meshgrid(Lp, L)

for i in range(num_steps):
    # load data
    file_name = "cross_L_Lp_Iter_" + str(i + 1) + ".txt"
    data = np.loadtxt(file_name, dtype=np.int, delimiter=",")

    indices = np.where(data < 72)

    # make figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel("$L_{\gamma, u}$")
    ax.set_ylabel("$log(L_{\gamma, p})$")
    ax.set_zlabel("$\sharp$")

    # import pdb; pdb.set_trace()

    ax.plot_trisurf(L[indices], Lp[indices], data[indices], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(azim=200)

    file_name = "cross_L_Lp_" + str(i + 1) + ".pdf"
    fig.savefig(file_name, bbox_inches='tight')
    plt.gcf().clear()