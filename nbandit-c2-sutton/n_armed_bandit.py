import numpy as np
import random

N = 10 # N - Armed Bandit Problem
p_matrix = np.random.rand(N)

class EGreedy(object):
    """
    :param e is set to 0.01
    """
    def __init__(self, no_of_train_episodes, e, temperature = 100, q_matrix = np.zeros(N)):
        self.no_of_train_episodes = no_of_train_episodes
        self.e = e
        self.temperature = temperature
        self.q_matrix = np.zeros(N) if q_matrix is None else q_matrix
        self.action_matrix = np.zeros(N)
        self.total_reward = 0.0

    def action(self, t):
        """
        :param t the count for the training step
        """
        selection_probabilities = np.exp(self.q_matrix/self.temperature)/np.sum(np.exp(self.q_matrix/self.temperature))
        random_value = random.random()
        for index, elem in enumerate(selection_probabilities):
            cum_prob = 0 if index is 0 else np.sum(selection_probabilities[0: index])
            state_prob = selection_probabilities[index]
            if random_value > cum_prob and random_value < state_prob + cum_prob:
                return index

        return action_selection

    def reward(self, last_action, reward_received):
        self.q_matrix[last_action] = (self.q_matrix[last_action]*self.action_matrix[last_action] + reward_received)/(self.action_matrix[last_action]+1)
        self.action_matrix[last_action] += 1
        self.total_reward += reward_received

    def q_matrix(self):
        return self.q_matrix

    def train(self):
        t = 0
        while t < self.no_of_train_episodes:
            print("This is t = {}".format(t))
            print("This is the q_matrix = {}".format(self.q_matrix))
            action_taken = self.action(t)
            print("Thi is action selected = {} with reward received = {}".format(action_taken, p_matrix[action_taken]))
            self.reward(action_taken, p_matrix[action_taken])
            t += 1

if __name__=='__main__':
    model = EGreedy(100, 0.01)
    model.train()
    print(p_matrix)
    print(model.q_matrix)
    print("Total Rewards = {}".format(model.total_reward))
