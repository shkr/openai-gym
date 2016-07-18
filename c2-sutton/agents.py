import abc
import numpy as np
import random

class Agent(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def select_action(self):
        """Method should return an action to take"""

    @abc.abstractmethod
    def send_observation(self, action, reward, obs_params):
        """
        :param action: action taken
        :param reward: reward recevied
        :param obs_params: other observation parameters
        """

class EGreedySoftmax(Agent):
    """
    :param e is set to 0.01
    :param tau is the temperature at which the softmax probabilities are calculated
    for each arm. P(arm) = e(q/tau)/ sum(e(q/tau))
    """
    def __init__(self, action_space, e = 0.01, tau = 1000):
        self._action_space = action_space
        self.e = e
        self.q_matrix = np.zeros(action_space)
        self.q_play = np.zeros(action_space)

    def select_action(self):
        """
        :param t the count for the training step
        """
        print(e)
        print(type(e))
        if np.random.randn() < self.e:
            return np.random.randint(self._action_space)
        else:
            scores = np.exp(self.q_matrix/self.tau)
            sel_prob = scores/sum(scores)
            rand_value = np.random.randn()
            for i in range(0, self._action_space):
                cum_prob = 0 if i==0 else sum(sel_prob[0: i])
                if (rand_value > cum_prob) and (rand_value <= (cum_prob + sel_prob[i])):
                    return i

    def send_observation(self, action, reward, obs_params):
        """
        """
        # Update equation
        self.q_play[action] += 1
        self.q_matrix[action] = self.q_matrix[action] + (1.0/self.q_play[action])*(reward - self.q_matrix[action])

class EGreedy(Agent):
    """
    :param e is set to 0.01
    """
    def __init__(self, action_space, e = 0.01):
        self._action_space = action_space
        self.e = e
        self.q_matrix = np.zeros(action_space)
        self.q_play = np.zeros(action_space)

    def select_action(self):
        """
        :param t the count for the training step
        """
        if np.random.randn() < self.e:
            return np.random.randint(self._action_space)
        else:
            return np.argmax(self.q_matrix)

    def send_observation(self, action, reward, obs_params):
        """
        """
        # Update equation
        self.q_play[action] += 1
        self.q_matrix[action] = self.q_matrix[action] + (1.0/self.q_play[action])*(reward - self.q_matrix[action])


