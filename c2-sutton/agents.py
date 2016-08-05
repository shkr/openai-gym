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

class SupervisedBinaryAgent(Agent):
    """
    A Supervisory agent
    which keeps a tally of instructions where a true signal
    has been received between two choices. Then it consistently
    chooses the action with the highest rate of true signals
    """
    def __init__(self):
        self.q_matrix = np.zeros(2)

    def select_action(self):
        """
        """
        scores = self.q_matrix
        return np.argmax(scores)

    def send_observation(self, action, reward, obs_param):
        """
        """
        if obs_param['is_correct']:
            self.q_matrix[action] += 1


class LRI(Agent):
    """
    Linear Reward Inaction
    This is a classical method from the field of learning automata
    It imitates a supervised learning algorithm however it is stochastic.
    Instead of committing totally to the correct action it updates
    a probability to select it after each action and uses that.
    """

    def __init__(self, action_space=2, alpha=0.1):
        self._action_space = action_space
        self._alpha = alpha
        self.q_matrix = np.full(action_space, 1.0/action_space)

    def select_action(self):
        """
        """
        sel_prob = self.q_matrix
        rand_value = np.random.uniform()
        for i in range(0, self._action_space):
            cum_prob = 0 if i==0 else sum(sel_prob[0: i])
            if (rand_value > cum_prob) and (rand_value <= (cum_prob + sel_prob[i])):
                return i

    def send_observation(self, action, reward, obs_param):
        """
        """
        if obs_param['is_correct']:
            prob_modifier  = self._alpha*(1-self.q_matrix[action])

            for this_action in range(0, self._action_space):
                if this_action is action:
                    self.q_matrix[action] += prob_modifier
                else:
                    self.q_matrix[action] -= prob_modifier/(self._action_space-1)


class LRP(Agent):
    """
    Linear Reward Penalty
    This is a classical method from the field of learning automata
    It imitates a supervised learning algorithm however it is stochastic.
    Instead of committing totally to the correct action it updates
    a probability to select it after each action and uses that.
    """

    def __init__(self, action_space=2, alpha=0.1):
        self._action_space = action_space
        self._alpha = alpha
        self.q_matrix = np.full(action_space, 1.0/action_space)

    def select_action(self):
        """
        """
        sel_prob = self.q_matrix
        rand_value = np.random.uniform()
        for i in range(0, self._action_space):
            cum_prob = 0 if i==0 else sum(sel_prob[0: i])
            if (rand_value > cum_prob) and (rand_value <= (cum_prob + sel_prob[i])):
                return i

    def send_observation(self, action, reward, obs_param):
        """
        """

        prob_modifier  = self._alpha*(1-self.q_matrix[action])
        modifier_sign = 2*obs_param['is_correct']  - 1

        for this_action in range(0, self._action_space):
            if this_action==action:
                self.q_matrix[action] += (prob_modifier*modifier_sign)
            else:
                self.q_matrix[action] += (-1*prob_modifier*modifier_sign)/(self._action_space-1)


class SoftmaxFixedStep(Agent):
    """
    :param e is set to 0.01
    :param tau is the temperature at which the softmax probabilities are calculated
    for each arm. P(arm) = e(q/tau)/ sum(e(q/tau))
    """
    def __init__(self, action_space, tau, step_size):
        self._action_space = action_space
        self.q_matrix = np.zeros(action_space)
        self._step_size = step_size

    def select_action(self):
        """
        :param t the count for the training step
        """
        scores = np.exp(self.q_matrix/self.tau)
        sel_prob = scores/np.sum(scores)
        rand_value = np.random.uniform()
        for i in range(0, self._action_space):
            cum_prob = 0 if i==0 else sum(sel_prob[0: i])
            if (rand_value > cum_prob) and (rand_value <= (cum_prob + sel_prob[i])):
                return i

    def send_observation(self, action, reward, obs_params):
        """
        """
        # Update equation
        self.q_matrix[action] = self.q_matrix[action] + (self._step_size)*(reward - self.q_matrix[action])


class Softmax(Agent):
    """
    :param tau is the temperature at which the softmax probabilities are calculated
    for each arm. P(arm) = e(q/tau)/ sum(e(q/tau))
    """
    def __init__(self, action_space, tau):
        self._action_space = action_space
        self.tau = tau
        self.q_matrix = np.zeros(action_space)
        self.q_play = np.zeros(action_space)

    def select_action(self):
        """
        :param t the count for the training step
        """
        scores = np.exp(self.q_matrix/self.tau)
        sel_prob = scores/np.sum(scores)
        rand_value = np.random.uniform()
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
    def __init__(self, action_space, e):
        self._action_space = action_space
        self.e = e
        self.q_matrix = np.zeros(action_space)
        self.q_play = np.zeros(action_space)

    def select_action(self):
        """
        :param t the count for the training step
        """
        if np.random.uniform() <= self.e:
            return np.random.randint(self._action_space)
        else:
            return np.argmax(self.q_matrix)

    def send_observation(self, action, reward, obs_params):
        """
        """
        # Update equation
        self.q_play[action] += 1
        self.q_matrix[action] = self.q_matrix[action] + (1.0/self.q_play[action])*(reward - self.q_matrix[action])


