import numpy as np
import matplotlib
import matplotlib.pyplot as plt

path = f"./log/selected/PG_Toy-10/Toy-10_0.01_0.1_0.1"
loss_list = np.load(f"{path}_loss.npy")

T = 50

plt.figure()
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
plt.plot(np.arange(0, T+1), loss_list[:T+1])
plt.xticks(np.arange(0, T+1, 10))
plt.xlabel("#iterations")
plt.ylabel(r"optimality gap")
plt.grid()
plt.savefig(f"{path}_loss.png", dpi=200)