import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt


path = "./plot/Toy-100_design_0.1_alpha_discounted"
with open(f"{path}.pkl" ,"rb") as f:
    lst = pkl.load(f)
    reward_std, reward_opt, reward_agent = tuple(lst)

reward_agent = np.array(reward_agent).T
reward_agent_avg = reward_agent.mean(axis=0)
reward_agent_std = reward_agent.std(axis=0)

reward_std = np.array(reward_std)
reward_opt = np.array(reward_opt)

print(reward_agent.T)

alpha_list = np.arange(0, 0.5, 0.01)
plt.plot(alpha_list, reward_std, label="classical")
plt.plot(alpha_list, reward_opt, label="optimal")
plt.plot(alpha_list, reward_agent_avg, label="RFZI")
plt.fill_between(alpha_list, reward_agent_avg-reward_agent_std, reward_agent_avg+reward_agent_std, color="C1", alpha=0.1)
plt.xlabel(r"$\alpha$")
plt.ylabel(r"value at $\alpha$")
plt.legend()
plt.savefig(f"{path}.png", dpi=200)