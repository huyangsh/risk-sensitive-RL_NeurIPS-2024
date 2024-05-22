import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""
paths = [
    f"./log/selected/PG_Toy-10/Toy-10_0.01_0.1_0.1",
    f"./log/selected/PG_Toy-10/Toy-10_0.01_1.0_0.1",
    f"./log/selected/PG_Toy-10/Toy-10_0.01_2.0_0.1",
    f"./log/selected/PG_Toy-10/Toy-10_0.01_3.0_0.1"
]
labels = [r"$\beta = 0.1$", r"$\beta = 1.0$", r"$\beta = 2.0$", r"$\beta = 3.0$"]
"""

paths = [
    f"./log/selected/PG_Toy-10/Toy-10_0.15_0.01_0.1",
    f"./log/selected/PG_Toy-10/Toy-10_0.15_0.1_0.1",
    f"./log/selected/PG_Toy-10/Toy-10_0.15_1.0_0.1"
]
labels = [r"$\beta = 0.01$", r"$\beta = 0.10$", r"$\beta = 1.00$"]

loss_list = []
for path in paths:
    loss_list.append(np.load(f"{path}_loss.npy"))

T = 25
loss_list = [x[0:T+1] for x in loss_list]
loss_list = np.array(loss_list).T

plt.figure()
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.plot(np.arange(0, T+1), loss_list, label=labels)
plt.xticks(np.arange(0, T+1, 5))
plt.xlabel("#iterations")
plt.ylabel(r"optimality gap")
plt.grid()
plt.legend()
plt.savefig(f"./log/selected/PG_Toy-10/merge_0.15.png", dpi=200)