import numpy as np
import matplotlib
import matplotlib.pyplot as plt

eval_freq = 10
env_name = "Toy-100_zone_0.5"
prefix = f"./log/selected/{env_name}/{env_name}"
seeds = [0, 10, 20, 30, 40]

V_opt = 0
V_pi_list, reward_list = [], []
for seed in seeds:
    with open(f"{prefix}_{seed}.log", "r") as f:
        log = f.read()

    pos = log.find("E[V_opt] = ")
    log = log[(pos+len("E[V_opt] = ")):]
    V_opt = float(log[:9])
    log = log[9:]

    pos = log.find("average reward = ")
    V_pi, reward = [], []
    while pos >= 0:
        log = log[(pos+len("average reward = ")):]
        stop = log.find(",")
        reward.append(float(log[:stop]))

        log = log[stop:]
        pos = log.find("E[V_pi] = ")

        log = log[(pos+len("E[V_pi] = ")):]
        stop = log.find(".\n")
        V_pi.append(float(log[:stop]))

        log = log[stop:]
        pos = log.find("average reward = ")
    
    V_pi_list.append(V_pi)
    reward_list.append(reward)

T = 200
V_pi_list = [[10] + x[1:T+1] for x in V_pi_list]
reward_list = [[5] + x[1:T+1] for x in reward_list]

loss_list = -np.array(V_pi_list) + V_opt
reward_list = np.array(reward_list)

loss_avg = loss_list.mean(axis=0)
loss_std = loss_list.std(axis=0)
reward_avg = reward_list.mean(axis=0)
reward_std = reward_list.std(axis=0)

t = np.arange(start=1, stop=len(loss_avg)+1) * eval_freq
t_max = t[-1]

matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams.update({'lines.linewidth': 2})

plt.figure()
plt.plot(t, reward_avg)
plt.fill_between(t, reward_avg-reward_std, reward_avg+reward_std, alpha=0.1)
# plt.axhline(V_opt, 0, 1, color="r", linestyle="--", label=r"$V_{\mathrm{opt}}$")
plt.xlabel("#iterations")
plt.ylabel("reward")
plt.grid()
# plt.legend(loc="lower right")
plt.savefig(f"./plot/{env_name}_reward.png", dpi=200, bbox_inches="tight")

plt.figure()
plt.plot(t, loss_avg)
plt.fill_between(t, loss_avg-loss_std, loss_avg+loss_std, alpha=0.1)
plt.xlabel("#iterations")
plt.ylabel("loss")
plt.grid()
plt.savefig(f"./plot/{env_name}_loss.png", dpi=200, bbox_inches="tight")