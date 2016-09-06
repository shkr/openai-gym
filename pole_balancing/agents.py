import logging
import os, sys
import abc
import gym
import numpy as np
import json
from os import listdir
from os.path import isfile, join

gym.scoreboard.api_key = "sk_hJMbMlYzSpSsucakOMOULg"

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class SequenceAgent(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def store_state(obs):
        """Stores the state of the environment"""


    @abc.abstractmethod
    def score(last_obs, action):
        """Returns a score or expected reward based on the last observation and action """


class GreedyMemoryAgent(SequenceAgent):
    """
    This Agent models the input (state space) as a sequence and 1/0 as the output.
    It is called MemoryAgent because it remembers its history to take better decisions in future
    """
    def __init__(self, env, **kwargs):

        # Environment and Action Space
        self.env = env
        self.action_space = env.action_space

        # Store History
        self.memory_size = kwargs.get('memory_size', 50000)
        self.past_obs = []
        self.past_action = []
        self.value_function =[0]

        # Configuration
        self.e = kwargs.get('e', 0.10)
        self.minimum_exploration = kwargs.get('minimum_exploration', 50)

        # Model parameters
        self.knn = kwargs.get('knn', 10)


    def store_episode(self, obs, reward, done, action):

        self.past_obs.append(obs.tolist())
        self.past_action.append(action)
        self.value_function[-1] += 1
        if done:
            self.value_function.append(0)

    def memory_full(self):
        return len(self.past_obs)==self.memory_size

    def pop_episode(self):
        self.past_obs.pop()
        self.past_action.pop()
        if(self.value_function[0]>1):
            self.value_function[0] -= 1
        else:
            self.value_function.pop(0)


    def policy(self):

        from sklearn.neighbors import KNeighborsRegressor

        # Models used
        classifier = KNeighborsRegressor(n_neighbors=self.knn)

        # classifier
        classifier_input = np.array([[*e[0], e[1]] for e in zip(self.past_obs[:-1], self.past_action)])

        classifier_output_lst = []
        for episode, episode_range in enumerate(self.value_function):
            for i in range(episode_range):
                classifier_output_lst.append(episode_range - i - 1)

        classifier_output_lst.pop()
        classifier_output = np.array(classifier_output_lst)
        classifier.fit(classifier_input, classifier_output)
        return self.past_obs[-1], classifier

    def act(self, curr_observation, last_reward, done, last_action):

        if(last_action!=None):

            # Push last event
            self.store_episode(curr_observation, last_reward, done, last_action)

            # Do minimum exploration
            if(self.minimum_exploration > sum(self.value_function)):
                return self.action_space.sample()

            # If all possible history to be known is known forget earliest event
            if(self.memory_full()):
                self.pop_episode()

            # Compute Policy
            information_state, current_policy = self.policy()

            # Calculate Expected Reward
            expected_reward = {}
            for action in range(self.action_space.__dict__['n']):
                expected_reward[action] = current_policy.predict([[*information_state, action]])[0]

            # Use policy for action selection
            if(np.random.random()<self.e):
                return self.action_space.sample()
            else:
                return (k for k in sorted(expected_reward, key=expected_reward.get, reverse=True)).__next__()
        else:
            return self.action_space.sample()


class RandomAgent(object):

    def __init__(self, env):
        self.action_space = env.action_space

    def act(self, observation, reward, done, last_action):
        return self.action_space.sample()


def run_game(agent_type, params={}, upload=False):

    env = gym.make('CartPole-v0')
    outdir = '/tmp/'+'cart_pole'+'_results'
    env.monitor.start(outdir, force=True, seed=0)
    episode_count = 300
    max_steps = 200

    if(agent_type is 'random'):
        agent = RandomAgent(env, **params)
    else:
        agent = GreedyMemoryAgent(env, **params)

    for i in range(episode_count):
        # Initial Values
        curr_ob = env.reset()
        last_reward = None
        done = False
        last_action = None

        for j in range(max_steps):

            action = agent.act(curr_ob, last_reward, done, last_action)
            curr_ob, last_reward, done, _ = env.step(action)
            last_action = action

            if(done):
                break

    logger.info("Successfully ran {}. Now uploading results to the scoreboard".format(agent))
    env.monitor.close()
    if upload:
        gym.upload(outdir)

    result_stats = json.load(open([join(outdir, f) for f in listdir(outdir) if isfile(join(outdir, f)) and f.startswith('openaigym.episode_batch')][0]))

    return agent, result_stats
