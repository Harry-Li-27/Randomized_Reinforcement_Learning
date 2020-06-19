import math

import gym
import numpy as np
import torch

import utils
from policies import QPolicy


class TabQPolicy(QPolicy):
    def __init__(self, env, buckets, actionsize, lr, gamma, model=None):
        """
        Inititalize the tabular q policy

        @param env: the gym environment
        @param buckets: specifies the discretization of the continuous state space for each dimension
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate for the model update
        @param gamma: discount factor
        @param model (optional): Stores the Q-value for each state-action
            model = np.zeros(self.buckets + (actionsize,))

        """
        super().__init__(len(buckets), actionsize, lr, gamma)
        self.env = env
        self.buckets = buckets
        # self.Q_vals = [[[[[0.0 for i in range(actionsize)] for j in range(buckets[3])] for k in range(buckets[2])] for l in range(buckets[1])] for m in range(buckets[0])]
        if model is None:
            self.model = np.zeros(self.buckets + (actionsize,))
        else:
            self.model = model
    def discretize(self, obs):
        """
        Discretizes the continuous input observation

        @param obs: continuous observation
        @return: discretized observation
        """
        upper_bounds = [self.env.observation_space.high[0], 5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    def qvals(self, states):
        """
        Returns the q values for the states.

        @param state: the state

        @return qvals: the q values for the state for each action.
        """
        # print(type(states))
        qval = np.zeros((len(states), self.actionsize))
        for i in range(len(states)):
            index = self.discretize(states[i])
            qval[i] = self.model[index[0]][index[1]][index[2]][index[3]]
        # print(qval.shape)
        return qval

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        current_index = self.discretize(state)
        next_index = self.discretize(next_state)
        target = 0
        if done:
            target = reward
        else:
            max = 0
            for value in self.model[next_index[0]][next_index[1]][next_index[2]][next_index[3]]:
                if value > max:
                    max = value
            target = reward + self.gamma * max
        # print(target, reward)
        temp = self.model[current_index[0]][current_index[1]][current_index[2]][current_index[3]][action]
        # print(temp)
        self.model[current_index[0]][current_index[1]][current_index[2]][current_index[3]][action] = temp + self.lr * (target - temp)
        # print(self.model[current_index[0]][current_index[1]][current_index[2]][current_index[3]][action])
        return (target - temp) ** 2


    def save(self, outpath):
        """
        saves the model at the specified outpath
        """
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')

    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n
    policy = TabQPolicy(env, buckets=(12, 10, 12, 10), actionsize=actionsize, lr=args.lr, gamma=args.gamma)
    # print(policy.model)

    utils.qlearn(env, policy, args)
    # print(policy.model)

    torch.save(policy.model, 'models/tabular.npy')
