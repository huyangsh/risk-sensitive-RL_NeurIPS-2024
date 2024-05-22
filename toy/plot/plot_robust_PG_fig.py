import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt

path = "./plot/PG_0.15_Toy-10_robust"
with open(f"{path}.pkl" ,"rb") as f:
    lst = pkl.load(f)
    reward_pis, reward_std = lst[0], lst[1]

delta_list = np.arange(0, 0.31, 0.01)
labels = [r"policy gradient ($\beta = 0.1$)", r"policy gradient ($\beta = 1.0$)"]
tag = "0.15"

"""
delta_list = np.arange(0, 0.61, 0.02)
labels = [r"policy gradient ($\beta = 1.0$)", r"policy gradient ($\beta = 2.0$)", r"policy gradient ($\beta = 3.0$)"]
tag = "0.01"
"""

reward_pis = np.array(reward_pis)
reward_std = np.array(reward_std)

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})

plt.plot(delta_list, reward_std, label="risk-neutral")
plt.plot(delta_list, reward_pis, label=labels)
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\hat{V}_{\pi}(\delta)$")
plt.grid()
plt.legend()
plt.savefig(f"{path}.png", dpi=200, bbox_inches="tight")