import gym
import numpy as np

env = gym.make('Pong-v0')
env.reset()
print(env.observation_space.high)
print(env.observation_space)
print(env.action_space.sample())
