import gym
import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np
import time

from buffer import ReplayMemory
from dqn import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from wrapper import *

env = gym.make('PongNoFrameskip-v4')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

if __name__ == '__main__':
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # hyperparameters
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1
    EPS_END = 0.02
    EPS_DECAY = 1000000
    TARGET_UPDATE = 1000
    RENDER = False
    lr = 1e-4
    INITIAL_MEMORY = 10000
    MEMORY_SIZE = 10 * INITIAL_MEMORY

    # create networks
    policy_net = DQN(n_actions=4).to(device)
    target_net = DQN(n_actions=4).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # setup optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    steps_done = 0

    # create environment
    env = gym.make("PongNoFrameskip-v4")
    env = make_env(env)

    # initialize replay memory
    memory = ReplayMemory(MEMORY_SIZE)

    # train model
    train(env, 400)
    torch.save(policy_net, "dqn_pong_model")
    policy_net = torch.load("dqn_pong_model")
    test(env, 1, policy_net, render=False)
