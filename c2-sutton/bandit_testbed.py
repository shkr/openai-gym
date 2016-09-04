import numpy as np
import logging
import abc
from multiprocessing import Pool
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Environment(object):
    """
    An environment has no memory. It returns a reward for each action
    It sends a sample from a guassian distribution centered at a mean for each arm
    and variance 1.
    It stores the means for each arm in the matrix Q_star of shape (n_arms)
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, a):
        """
        Returns a reward for an action which is part of the action_space otherwise None
        """

class Bandit(Environment):
    """
    A Bandit with n_arms
    It sends a sample from a guassian distribution centered at a mean for each arm
    and variance 1.
    It stores the means for each arm in the matrix Q_star of shape (n_arms)
    """
    def __init__(self, n_arms):
        self._n_arms = n_arms
        self.Q_star = np.random.randn(self._n_arms)

    def act(self, a):
        """
        :param a: a is the index of the arm that is used or the action applied on the bandit
        """
        random_normal_sample = float(np.random.randn())
        return float(self.Q_star[a]) + random_normal_sample


class InstructiveBinaryBandit(Environment):
    """
    """
    def __init__(self, _type):
        if _type is 'A':
            self.Q_star = np.array([0.1, 0.2])
        else:
            self.Q_star = np.array([0.8, 0.9])

    def act(self, a):
        """
        :param a: a is the index of the arm that is used or the action applied on the bandit
        """
        random_uniform_sample = np.random.random()
        if random_uniform_sample < self.Q_star[a]:
            return 1, {'success': True}
        else:
            return 0, {'success': False}


class InstructiveBinaryTestBed(object):
    """
    The N-Armed Bandit Testbed is played with a Bandit who has N arms
    A play is called, when an action is taken on the bandit and a reward is returned.
    P is the number of plays played with the Bandit.
    A set of P plays is called a game.
    We will play in total G games
    agent is the agent which is playing in this testbed.
    the agent_cls is expected to extend Agent class
    """
    def __init__(self, n_plays, n_games, agent_cls, cls_args):
        self._n_plays = n_plays
        self._n_games = n_games
        self._agent_cls = agent_cls
        self._cls_args = cls_args

    @classmethod
    def run_game(cls, bandit, agent, n_plays):
        """
        """
        game = {"optimal_action": [],
                "prob_optimal_action": []}
        optimal_action = np.argmax(bandit.Q_star)
        for i in range(0, n_plays):
            action = agent.select_action()
            reward, obs_params = bandit.act(action)
            agent.send_observation(action, reward, obs_params)
            game["optimal_action"].append(action==optimal_action)
            game["prob_optimal_action"].append(sum(game["optimal_action"])/(i+1.0))

        return game

    def run_all_games(self, _type):
        pool = Pool(3)
        games = pool.starmap(self.run_game, [(InstructiveBinaryBandit(_type), \
            self._agent_cls(**self._cls_args), self._n_plays) for i in range(0, self._n_games)])
        return games


class TestBed(object):
    """
    The N-Armed Bandit Testbed is played with a Bandit who has N arms
    A play is called, when an action is taken on the bandit and a reward is returned.
    P is the number of plays played with the Bandit.
    A set of P plays is called a game.
    We will play in total G games
    agent is the agent which is playing in this testbed.
    the agent_cls is expected to extend Agent class
    """
    def __init__(self, n_arms, n_plays, n_games, agent_cls, cls_args):
        self._n_arms = n_arms
        self._n_plays = n_plays
        self._n_games = n_games
        self._agent_cls = agent_cls
        self._cls_args = cls_args
        self._games = []

    @classmethod
    def run_game(cls, bandit, agent, n_plays):
        """
        """
        game = {
                "reward": [],
                "average_reward": [],
                "optimal_action": [],
                "prob_optimal_action": [],
                "total_reward" : 0.0
        }
        optimal_action = np.argmax(bandit.Q_star)
        optimal_reward = np.max(bandit.Q_star)
        for i in range(0, n_plays):

            action = agent.select_action()
            reward = bandit.act(action)
            agent.send_observation(action, reward, {})
            game["reward"].append(reward)
            game["average_reward"].append(sum(game["reward"])/(i+1.0))
            game["optimal_action"].append(action==optimal_action)
            game["prob_optimal_action"].append(sum(game["optimal_action"])/(i+1.0))
            game["total_reward"] += reward

        game["optimal_reward"] = optimal_reward

        return game

    def run_all_games(self, multi_thread=True):
        if multi_thread:
            pool = Pool(3)
            games = pool.starmap(self.run_game, [(Bandit(self._n_arms), \
            self._agent_cls(self._n_arms, **self._cls_args), self._n_plays) for i in range(0, self._n_games)])
            self._games = games
        else:
            games = []
            for i in range(0, self._n_games):
                games.append(self.run_game(Bandit(self._n_arms), self._agent_cls(self._n_arms, **self._cls_args), self._n_plays))
            self._games = games
