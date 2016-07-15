import numpy as np
import random
import sys

N = 10 # N - Armed Bandit Problem
EPOCHS = 10000
p_matrix = np.random.rand(N)

class NonStationaryBandit(object):

    def __init__(self, initial_reward, stationary_time_limit=10):
        self.reward = initial_reward
        self._counter = 0
        self._stationary_time_limit = stationary_time_limit

    def act(self):
        if self._counter < self._stationary_time_limit:
            return self.reward
        # Otherwise begin random walk
        random_value = random.random()
        if random_value < 0.5:
            return 1
        else:
            return -1

class IncrementalSampleAverageUpdate(object):

    def __init__(self, action_space, e=None, temperature=None, q_matrix=None):
        self.e = 0.1 if e is None else e
        self.temperature = 1000 if temperature is None else temperature
        self.q_matrix = np.zeros(action_space) if q_matrix is None else q_matrix
        self.action_matrix = np.zeros(action_space)
        self.total_reward = 0.0

    def act(self, t):
        selection_probabilities = np.exp(self.q_matrix/self.temperature)/np.sum(np.exp(self.q_matrix/self.temperature))
        random_value = random.random()
        for index, prob in enumerate(selection_probabilities):
            cum_prob = 0 if index is 0 else np.sum(selection_probabilities[0: index])
            if random_value >= cum_prob and random_value < prob + cum_prob:
                return index

    def alpha(self, t):
        return 1.0/t

    def update(self, t, last_obs, last_act, last_reward):
        self.q_matrix[last_act] = self.q_matrix[last_act] + self.alpha(t)*(last_reward - self.q_matrix[last_act])
        if(last_reward < 0.0):
            print ("Negative!")
        self.total_reward += last_reward
def train(no_of_bandits, no_of_train_episodes):
    t = 1

    bandits = [ NonStationaryBandit(1.0, 3) for i in range(0, no_of_bandits) ]
    agent = IncrementalSampleAverageUpdate(no_of_bandits)

    while t <= no_of_train_episodes:
        print("This is t = {}".format(t))
        print("This is the q_matrix = {}".format(agent.q_matrix))
        action_taken = agent.act(t)
        reward = bandits[action_taken].act()
        print("Thi is action selected = {} with reward received = {}".format(action_taken, reward))
        agent.update(t, None, action_taken, reward)
        t += 1
    print("Total Reward received = ", agent.total_reward)

if __name__=='__main__':
    train(N, EPOCHS)
