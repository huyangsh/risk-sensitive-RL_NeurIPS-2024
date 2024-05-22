import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt

env_name = "Toy-100_design_0.1"
with open(f"./plot/{env_name}_robust.pkl" ,"rb") as f:
    lst = pkl.load(f)
    reward_agent, reward_std, reward_opt = lst[0], lst[1], lst[2]

delta_list = np.arange(0, 0.31, 0.02)

reward_agent = np.array(reward_agent)
reward_agent_avg = reward_agent.mean(axis=0)
reward_agent_std = reward_agent.std(axis=0)

reward_std = np.array(reward_std)
reward_opt = np.array(reward_opt)

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'lines.linewidth': 2})

plt.plot(delta_list, reward_std, label="risk-neutral")
# plt.plot(delta_list, reward_opt, label="optimal")
plt.plot(delta_list, reward_agent_avg, label="RFZI")
plt.fill_between(delta_list, reward_agent_avg-reward_agent_std, reward_agent_avg+reward_agent_std, color="C1", alpha=0.1)
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\hat{V}_{\pi}(\delta)$")
plt.grid()
plt.legend()
plt.savefig(f"./plot/{env_name}_robust.png", dpi=200, bbox_inches="tight")