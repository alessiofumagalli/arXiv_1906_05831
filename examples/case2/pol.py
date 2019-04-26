import matplotlib.pyplot as plt
import numpy as np

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

file_name = "cross_L_dependency.txt"
data = np.loadtxt(file_name, dtype=np.int, delimiter=",")

num_steps = 5
L = 0.025*np.arange(101)

for step in np.arange(num_steps):
    plt.plot(L, data[:, step])
    plt.ylabel("$\sharp$")
    plt.xlabel("$L$")
    plt.grid(True)
    file_name = "cross_" + str(step) + ".pdf"
    plt.savefig(file_name, bbox_inches='tight')
    plt.gcf().clear()
