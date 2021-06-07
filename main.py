
import gym

import time
import psutil
import mujoco_py
# from torch.utils.tensorboard import SummaryWriter
import os
import datetime



#ENV_NAME = "Hopper-v2"
ENV_NAME = 'HalfCheetah-v2'
test_env = gym.make(ENV_NAME)
TRAIN = True


if not os.path.exists(ENV_NAME):
	os.mkdir(ENV_NAME)


# test_env.render()
n_states = test_env.observation_space.shape[0]
n_actions = test_env.action_space.shape[0]
action_bounds = [test_env.action_space.low[0],test_env.action_space.high[0]]
test_env.close()



env = gym.make(ENV_NAME)





## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.nn.parameter import Parameter
from torch.optim.adam import Adam


from torch.nn.utils import spectral_norm
from torch.distributions.multivariate_normal import MultivariateNormal


device = 'cuda'



class RSSM(nn.Module):
	def __init__(self,):
		super(RSSM,self).__init__()

class MLP(nn.Module):
	def __init__(self,in_dim,out_dim,hiddens=[]):
		"""
		model = MLP(in_dim=2,out_dim=2,hiddens=[4,5,6,7])
		"""
		super(MLP, self).__init__()

		self.linears = nn.ModuleList([])
		self.bns = nn.ModuleList([])

		for i in range(len(hiddens)):
			if i ==0:
				exec('self.linears.append(nn.Linear({},{}))'.format(in_dim,hiddens[i]))
				exec('self.bns.append(nn.BatchNorm1d({}))'.format(hiddens[i]))
			else:
				exec('self.linears.append(nn.Linear({},{}))'.format(hiddens[i-1],hiddens[i]))
				exec('self.bns.append(nn.BatchNorm1d({}))'.format(hiddens[i]))

		if hiddens==None or len(hiddens) ==0:
			self.head = nn.Linear(in_dim,out_dim)
		else:
			self.head = nn.Linear(hiddens[-1],out_dim)

	def forward(self,x):

		for i in range(len(self.linears)):
			x = F.relu(self.bns[i](self.linears[i](x)))
		return self.head(x)

class MLP_no_bn(nn.Module):
	def __init__(self,in_dim,out_dim,hiddens=[]):
		"""
		model = MLP(in_dim=2,out_dim=2,hiddens=[4,5,6,7])
		"""
		super(MLP_no_bn, self).__init__()

		self.linears = nn.ModuleList([])

		for i in range(len(hiddens)):
			if i ==0:
				exec('self.linears.append(nn.Linear({},{}))'.format(in_dim,hiddens[i]))
			else:
				exec('self.linears.append(nn.Linear({},{}))'.format(hiddens[i-1],hiddens[i]))

		if hiddens==None or len(hiddens) ==0:
			self.head = nn.Linear(in_dim,out_dim)
		else:
			self.head = nn.Linear(hiddens[-1],out_dim)

	def forward(self,x):

		for i in range(len(self.linears)):
			x = F.relu(self.linears[i](x))
		return self.head(x)



############## model fitting part
import random 
class ReplayBuffer(object):
	def __init__(self,BufferSize):
		self.BufferSize = BufferSize
		self.Buffer = []

	def collect(self,traj):
		if len(self.Buffer) > self.BufferSize-1:
			self.Buffer.pop(0)
			self.Buffer.append(traj)
		else:
			self.Buffer.append(traj)

	def sample_batch(self,batchsize):

		return random.sample(self.Buffer,batchsize)

	def sample_batch_trajs(self,batchsize,traj_len):

		return None




### initialize buffer period
import collections
import numpy as np


n_actions
s_dim = 4
h_dim = 4
o_dim = n_states


h_in_dim = h_dim + s_dim + n_actions
h_ou_dim = h_dim
H_in_dim = h_dim
trans_in_dim = h_dim
trans_ou_dim = s_dim
enc_in_dim = o_dim + h_dim
enc_ou_dim = s_dim
dec_in_dim = h_dim + s_dim
dec_ou_dim = o_dim
rew_in_dim = h_dim + s_dim
rew_ou_dim = 1

rnn_kernel = MLP_no_bn(in_dim=h_in_dim,out_dim=h_ou_dim,hiddens=[8]).cuda()
trans_mu_net = MLP_no_bn(in_dim=trans_in_dim,out_dim=trans_ou_dim,hiddens=[8]).cuda()
trans_logsigma_net = MLP_no_bn(in_dim=trans_in_dim,out_dim=trans_ou_dim,hiddens=[8]).cuda()
trans_epsilon_dist = MultivariateNormal(loc=torch.zeros(trans_ou_dim),covariance_matrix=torch.eye(trans_ou_dim))


enc_mu_net = MLP_no_bn(in_dim=enc_in_dim,out_dim=enc_ou_dim,hiddens=[8]).cuda()
enc_logsigma_net = MLP_no_bn(in_dim=enc_in_dim,out_dim=enc_ou_dim,hiddens=[8]).cuda()
enc_epsilon_dist = MultivariateNormal(loc=torch.zeros(enc_ou_dim),covariance_matrix=torch.eye(enc_ou_dim))

dec_mu_net = MLP_no_bn(in_dim=dec_in_dim,out_dim=dec_ou_dim,hiddens=[8]).cuda()
dec_logsigma_net = MLP_no_bn(in_dim=dec_in_dim,out_dim=dec_ou_dim,hiddens=[8]).cuda()
dec_epsilon_dist = MultivariateNormal(loc=torch.zeros(dec_ou_dim),covariance_matrix=torch.eye(dec_ou_dim))

rew_mu_net = MLP_no_bn(in_dim=rew_in_dim,out_dim=rew_ou_dim,hiddens=[8]).cuda()
rew_logsigma_net = MLP_no_bn(in_dim=rew_in_dim,out_dim=rew_ou_dim,hiddens=[8]).cuda()
rew_epsilon_dist = MultivariateNormal(loc=torch.zeros(rew_ou_dim),covariance_matrix=torch.eye(rew_ou_dim))

rssm_optimizer = Adam([
	{'params':rnn_kernel.parameters()},
	{'params':trans_mu_net.parameters()},
	{'params':trans_logsigma_net.parameters()},
	{'params':enc_mu_net.parameters()},
	{'params':enc_logsigma_net.parameters()},
	{'params':dec_mu_net.parameters()},
	{'params':dec_logsigma_net.parameters()},
	{'params':rew_mu_net.parameters()},
	{'params':rew_logsigma_net.parameters()}],lr=0.01)







### Model fitting period
# BATCHSIZE = 64
# UPDATESTEPS = 2


def compute_gaussian_loglike(x,p,mu,logsigma2):
	return -p/2*torch.log(2*torch.tensor(np.pi)) - 1/2*logsigma2.sum(dim=1) - (1/2/torch.exp(logsigma2)*torch.square(x-mu)).sum(dim=1)


def compute_gaussian_kl_div(mu1,mu2,Sigma2_1,Sigma2_2):
	'''
	Sigma1 means Sigma Square Matrix
	'''

	pass

	return None

def compute_gaussian_kl_div_with_diag_cov(mu1,mu2,logsigma2_1,logsigma2_2):
	if len(mu1.shape) >1:
		return mu1.shape[1]
	else:
		return mu1.shape[0]
	kl_div = 1/2*((torch.exp(logsigma2_1-logsigma2_2)).sum() + (torch.square(mu2-mu1)/torch.exp(logsigma2_2)).sum() - k + (logsigma2_2-logsigma2_1).sum())

	return kl_div




def compute_traj_loss(traj,kl_beta=1.0):
	'''
	traj should be a Trans tensor 
	'''

	observs = torch.tensor(traj.observation,dtype=torch.float32).cuda()
	actions = torch.tensor(traj.action,dtype=torch.float32).cuda()
	rewards = torch.tensor(traj.reward,dtype=torch.float32).cuda()
	next_observs = torch.tensor(traj.next_observation,dtype=torch.float32).cuda()
	dones = torch.tensor(traj.done).cuda()

	traj_len = observs.shape[0]


	traj_loss = torch.tensor([0.]).cuda()

	ht = torch.zeros(1,h_ou_dim).cuda()
	ot = observs[0].unsqueeze(0)
	st = enc_mu_net(torch.cat([ht,ot],dim=1)) + torch.exp(enc_logsigma_net(torch.cat([ht,ot],dim=1)))*enc_epsilon_dist.sample([1]).cuda()
	at = actions[0].unsqueeze(0)
	rt = rewards[0].unsqueeze(0).unsqueeze(0)

	posterior_mu = enc_mu_net(torch.cat([ht,ot],dim=1))
	posterior_logsigm2 = 2*enc_logsigma_net(torch.cat([ht,ot],dim=1))
	posterior_p = st.shape[1]
	prior_mu = trans_mu_net(ht)
	prior_logsigm2 = 2*trans_logsigma_net(ht)
	prior_p = posterior_p

	# kl_loss = compute_gaussian_loglike(st,posterior_p,posterior_mu,posterior_logsigm2) - compute_gaussian_loglike(st,prior_p,prior_mu,prior_logsigm2)

	kl_loss = compute_gaussian_kl_div_with_diag_cov(posterior_mu,prior_mu,posterior_logsigm2,prior_logsigm2)

	for i in range(1,traj_len):

		ht= rnn_kernel(torch.cat([ht,st,at],dim=1))
		ot = observs[i].unsqueeze(0)
		st = enc_mu_net(torch.cat([ht,ot],dim=1)) + torch.exp(enc_logsigma_net(torch.cat([ht,ot],dim=1)))*enc_epsilon_dist.sample([1]).cuda()
		at = actions[i].unsqueeze(0)


		rec_mean = dec_mu_net(torch.cat([ht,st],dim=1))
		rec_logsigm2 = 2*dec_logsigma_net(torch.cat([ht,st],dim=1))
		rec_p = ot.shape[1]
		rec_loss = -compute_gaussian_loglike(ot,rec_p,rec_mean,rec_logsigm2)

		rew_mean = rew_mu_net(torch.cat([ht,st],dim=1))
		rew_logsigm2 = 2*rew_logsigma_net(torch.cat([ht,st],dim=1))
		rew_p = 1
		rew_loss = -compute_gaussian_loglike(rt.unsqueeze(0),rew_p,rew_mean,rew_logsigm2)

		traj_loss = traj_loss + rec_loss + rew_loss + kl_beta*kl_loss

		posterior_mu = enc_mu_net(torch.cat([ht,ot],dim=1))
		posterior_logsigm2 = 2*enc_logsigma_net(torch.cat([ht,ot],dim=1))
		posterior_p = st.shape[1]
		prior_mu = trans_mu_net(ht)
		prior_logsigm2 = 2*trans_logsigma_net(ht)
		prior_p = posterior_p

		kl_loss = compute_gaussian_loglike(st,posterior_p,posterior_mu,posterior_logsigm2) - compute_gaussian_loglike(st,prior_p,prior_mu,prior_logsigm2)

	return traj_loss

# for _ in range(UPDATESTEPS):
# 	trajs = ExpReplayBuffer.sample_batch(BATCHSIZE)

# 	traj_loss = torch.tensor([0.])
# 	for traj in trajs:
# 		traj_loss = traj_loss + compute_traj_loss(Transition(*zip(*traj)))
# 	traj_loss = traj_loss/BATCHSIZE

# 	rssm_optimizer.zero_grad()
# 	traj_loss.backward()
# 	rssm_optimizer.step()


### data collection period


# Plan_Hor = 10
# Opt_Iters = 2
# Candidates = 20
# Top_K = 10


def cem_plan_in_latent_space(Plan_Hor,Opt_Iters,Candidates,Top_K,ht,st):

	ht = ht.repeat(Candidates,1).cuda()
	st = st.repeat(Candidates,1).cuda()

	action_seq_dist = MultivariateNormal(loc=torch.zeros(n_actions*Plan_Hor),covariance_matrix=torch.eye(n_actions*Plan_Hor))

	mu,sigm = torch.zeros(n_actions*Plan_Hor),torch.ones(n_actions*Plan_Hor)

	for opt_iter in range(Opt_Iters):
		a_seqs = torch.clamp(action_seq_dist.sample(torch.Size([Candidates])),action_bounds[0],action_bounds[1])
		a_seqs = sigm*a_seqs + mu

		rts = torch.zeros(Candidates,rew_ou_dim)
		for i in range(Plan_Hor):
			at = a_seqs[:,i*n_actions:(i+1)*n_actions].cuda()
			rt = rew_mu_net(torch.cat([ht,st],dim=1)) + torch.exp(rew_logsigma_net(torch.cat([ht,st],dim=1)))*rew_epsilon_dist.sample([Candidates]).cuda()
			rts = rts + rt.cpu()

			ht = rnn_kernel(torch.cat([ht,st,at],dim=1))
			st = trans_mu_net(ht) + torch.exp(trans_logsigma_net(ht))*trans_epsilon_dist.sample([Candidates]).cuda()

		top_k_ind = torch.argsort(rts.squeeze(),descending=True)[0:Top_K]
		top_k_as = a_seqs[top_k_ind]

		mu = top_k_as.mean(dim=0)
		sigm = Top_K/(Top_K - 1)*torch.mean(torch.abs(top_k_as-mu),dim=0)

	return mu,sigm



# with torch.no_grad():

# 	env.reset()
# 	for warm_epsode in range(200):
# 		ot = torch.tensor(env.reset(),dtype=torch.float32).unsqueeze(0)
# 		ht = torch.zeros(1,h_ou_dim)
# 		done = False

# 		episode_traj = []
# 		while not done:
# 			st = enc_mu_net(torch.cat([ht,ot],dim=1)) + torch.exp(enc_logsigma_net(torch.cat([ht,ot],dim=1)))*enc_epsilon_dist.sample([1])
# 			mu,sigm = cem_plan_in_latent_space(Plan_Hor,Opt_Iters,Candidates,Top_K,ht,st)

# 			at = mu[0:n_actions].unsqueeze(0)
# 			oot, rt, done, _ = env.step(at)
# 			episode_traj.append(Transition(observation=ot,action=at,reward=rt,next_observation=oot,done=done))
# 			ot = torch.tensor(oot,dtype=torch.float32).unsqueeze(0)
# 		ExpReplayBuffer.collect(episode_traj)



### evaluation period

# EvaluationBuffer = ReplayBuffer(1000)


def evaluate(EvaluationBuffer,render=False):

	with torch.no_grad():

		ot = torch.tensor(env.reset(),dtype=torch.float32).unsqueeze(0).cuda()
		ht = torch.zeros(1,h_ou_dim).cuda()
		done = False
		episode_r = torch.tensor([0.])
		episode_traj = []
		while not done:
			st = enc_mu_net(torch.cat([ht,ot],dim=1)) + torch.exp(enc_logsigma_net(torch.cat([ht,ot],dim=1)))*enc_epsilon_dist.sample([1]).cuda()
			mu,sigm = cem_plan_in_latent_space(Plan_Hor,Opt_Iters,Candidates,Top_K,ht,st)

			at = mu[0:n_actions].unsqueeze(0)
			oot, rt, done, _ = env.step(at)
			env.render()
			episode_r = episode_r + rt
			episode_traj.append(Transition(observation=ot.cpu().squeeze().numpy(),action=at.cpu().squeeze().numpy(),reward=rt,next_observation=oot,done=done))
			ot = torch.tensor(oot,dtype=torch.float32).unsqueeze(0).cuda()
		EvaluationBuffer.collect(episode_traj)

		return episode_r






### full pipeline

# buffer hyperparam
Transition = collections.namedtuple('Trans', ['observation','action','reward','next_observation','done'])

BufferSize = 100
ExpReplayBuffer = ReplayBuffer(BufferSize)
EvaluationBuffer = ReplayBuffer(1000)


# train hyper param
TOTAL_EPS = 1000
BATCHSIZE = 50
UPDATESTEPS = 1

# plan hyperparam
Plan_Hor = 12
Opt_Iters = 10
Candidates = 1000
Top_K = 100
KL_BETA = 0.1

# prefill some episodes
env.reset()
warm_rewards = []
for warm_epsode in range(100):
	obs = env.reset()
	done = False

	episode_traj = []
	episode_r = 0
	while not done:
		action = env.action_space.sample()
		next_obs, reward, done, _ = env.step(action)
		env.render()
		episode_traj.append(Transition(observation=obs,action=action,reward=reward,next_observation=next_obs,done=done))
		obs = next_obs
		episode_r = episode_r + reward
	print("prefill episode {}, reward {}".format(warm_epsode,episode_r))
	ExpReplayBuffer.collect(episode_traj)
	warm_rewards.append(episode_r)

eval_rewards = []
eval_episodes = []
train_rewards = []
train_episodes = []
total_ep = 0
for total_ep in range(total_ep,TOTAL_EPS):
	# update parameter
	for _ in range(UPDATESTEPS):
		trajs = ExpReplayBuffer.sample_batch(BATCHSIZE)

		traj_loss = torch.tensor([0.]).cuda()
		for traj in trajs:
			traj = Transition(*zip(*traj))
			traj_loss = traj_loss + compute_traj_loss(traj,kl_beta=KL_BETA)
		traj_loss = traj_loss/BATCHSIZE

		rssm_optimizer.zero_grad()
		traj_loss.backward()
		rssm_optimizer.step()


	# collect data
	with torch.no_grad():

		env.reset()
		ot = torch.tensor(env.reset(),dtype=torch.float32).unsqueeze(0).cuda()
		ht = torch.zeros(1,h_ou_dim).cuda()
		done = False

		episode_traj = []
		episode_reward = 0
		while not done:
			st = enc_mu_net(torch.cat([ht,ot],dim=1)) + torch.exp(enc_logsigma_net(torch.cat([ht,ot],dim=1)))*enc_epsilon_dist.sample([1]).cuda()
			mu,sigm = cem_plan_in_latent_space(Plan_Hor,Opt_Iters,Candidates,Top_K,ht,st)

			mu = torch.clamp(mu + np.sqrt(0.3)*np.random.randn(len(mu)),action_bounds[0],action_bounds[1])

			at = mu[0:n_actions].unsqueeze(0)
			oot, rt, done, _ = env.step(at)
			episode_reward = episode_reward + rt
			env.render()
			episode_traj.append(Transition(observation=ot.squeeze().cpu().numpy(),action=at.squeeze().cpu().numpy(),reward=rt,next_observation=oot,done=done))
			ot = torch.tensor(oot,dtype=torch.float32).unsqueeze(0).cuda()
		ExpReplayBuffer.collect(episode_traj)
		train_rewards.append(episode_reward)
		train_episodes.append(total_ep)



	# evaluation


	epr = evaluate(EvaluationBuffer)
	eval_rewards.append(epr)
	eval_episodes.append(total_ep)
	print("train episode {}, eval reward {}".format(total_ep,epr))
	np.save('./eval_episodes.npy',np.array(eval_episodes))
	np.save('./eval_rewards.npy',np.array(eval_rewards))

	print("loop {}".format(total_ep))




env.reset()
random_rewards = []
random_episodes = []
for random_episode in range(TOTAL_EPS):
	obs = env.reset()
	done = False

	episode_traj = []
	episode_r = 0
	while not done:
		action = env.action_space.sample()
		next_obs, reward, done, _ = env.step(action)
		env.render()
		episode_traj.append(Transition(observation=obs,action=action,reward=reward,next_observation=next_obs,done=done))
		obs = next_obs
		episode_r = episode_r + reward
	print("prefill episode {}, reward {}".format(random_episode,episode_r))
	ExpReplayBuffer.collect(episode_traj)
	random_rewards.append(episode_r)
	random_episodes.append(random_episode)


def cum_mean(a):
	return np.cumsum(a)/range(1,len(a)+1)




############################# visualization part


import matplotlib.pyplot as plt

eval_episodes = np.load('./{}_eval_episodes.npy'.format(ENV_NAME))
eval_rewards = np.load('./{}_eval_rewards.npy'.format(ENV_NAME),allow_pickle=True)




eval_rewards = np.array([eval_reward.squeeze().numpy() for eval_reward in eval_rewards])
eval_rewards = cum_mean(eval_rewards)
random_rewards = cum_mean(random_rewards)


plt.figure()
plt.title('comparison of planet and random policy')

l1 = plt.plot(eval_episodes,eval_rewards,color='blue',label='planet')
l2 = plt.plot(random_episodes,random_rewards,color='red',label='random')


plt.legend()

plt.xlabel('train episodes')
plt.ylabel('average reward')

plt.show()
# plt.plot(train_episodes,train_rewards,color='orange');plt.show()























