import time
# import gym
import numpy as np 
# import torch

from torch import nn
from torch.nn import functional as F


def unpack_batchv2(batch):
  return batch.state, batch.action, batch.reward, batch.next_state, batch.next_action,batch.next_reward,batch.next_next_state,batch.done


def unpack_batch(batch):
  return batch.state, batch.action, batch.next_state, batch.reward, batch.done


class Timer:

	def __init__(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def reset(self):
		self._start_time = time.time()
		self._step_time = time.time()
		self._step = 0

	def set_step(self, step):
		self._step = step
		self._step_time = time.time()

	def time_cost(self):
		return time.time() - self._start_time

	def steps_per_sec(self, step):
		sps = (step - self._step) / (time.time() - self._step_time)
		self._step = step
		self._step_time = time.time()
		return sps


def eval_policy(policy, eval_env, eval_episodes=100):
	"""
	Eval a policy
	"""
	ep_rets = []
	avg_len = 0.
	for i in range(eval_episodes):
		ep_ret = 0.
		# eval_env.seed(i)
		state, done = eval_env.reset(seed=i+25), False
		# print("eval_policy state", state)
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			ep_ret += reward
			avg_len += 1
		ep_rets.append(ep_ret)

	avg_ret = np.mean(ep_rets)
	std_ret = np.std(ep_rets)
	avg_len /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: avg eplen {avg_len}, avg return {avg_ret:.3f} $\pm$ {std_ret:.3f}")
	print("---------------------------------------")
	return avg_len, avg_ret, std_ret, ep_rets



def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers."""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)


class MLP(nn.Module):
	def __init__(self,
								input_dim,
								hidden_dim,
								output_dim,
								hidden_depth,
								output_mod=None):
		super().__init__()
		self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth,
											output_mod)
		self.apply(weight_init)

	def forward(self, x):
		return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
	if hidden_depth == 0:
		mods = [nn.Linear(input_dim, output_dim)]
	else:
		mods = [nn.Linear(input_dim, hidden_dim), nn.ELU(inplace=True)]
		for i in range(hidden_depth - 1):
			mods += [nn.Linear(hidden_dim, hidden_dim), nn.ELU(inplace=True)]
		mods.append(nn.Linear(hidden_dim, output_dim))
	if output_mod is not None:
		mods.append(output_mod)
	trunk = nn.Sequential(*mods)
	return trunk

def to_np(t):
	if t is None:
		return None
	elif t.nelement() == 0:
		return np.array([])
	else:
		return t.cpu().detach().numpy()

def clear_data():
	import pickle as pkl
	with open('/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/temp_good_results/rfsac_nystrom_True_rf_num_2048_sample_dim_8192/seed_0_2023-09-02-12-24-08/train_params.pkl',
			  'rb') as f:
		a = pkl.load(f)
		a['replay_buffer'] = None
		print(a)
	with open('/home/mht/PycharmProjects/lvrep-rl-cloned/log/Quadrotor2D-v2_sigma_0.0_rew_scale_10.0/temp_good_results/rfsac_nystrom_True_rf_num_2048_sample_dim_8192/seed_0_2023-09-02-12-24-08/train_params.pkl',
			  'wb') as f:
		pkl.dump(a, f)


if __name__ == '__main__':
    clear_data()