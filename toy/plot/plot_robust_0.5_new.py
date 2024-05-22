import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

env_name = "Toy-100_zone_0.5"
with open(f"./plot/{env_name}_robust_opt.pkl" ,"rb") as f:
    reward_opt = pkl.load(f)

with open(f"./plot/{env_name}_robust.pkl" ,"rb") as f:
    lst = pkl.load(f)
    reward_agent, reward_std = lst[0], lst[1]

delta_list = np.arange(0, 0.31, 0.01)

reward_agent = np.array(reward_agent)
reward_agent_avg = reward_agent.mean(axis=0)
reward_agent_std = reward_agent.std(axis=0)

reward_std = np.array(reward_std)

plt.plot(delta_list, reward_std, label="classical")
plt.plot(delta_list, reward_opt, label="optimal")
plt.plot(delta_list, reward_agent_avg, label="RFZI")
plt.fill_between(delta_list, reward_agent_avg-reward_agent_std, reward_agent_avg+reward_agent_std, color="C1", alpha=0.1)
plt.xlabel(r"$\delta$")
plt.ylabel(r"$\hat{V}_{\pi}(\delta)$")
plt.legend()
plt.savefig(f"./plot/{env_name}_robust.png", dpi=200)