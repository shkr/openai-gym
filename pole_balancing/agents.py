import logging
import os, sys
import abc
import gym
import numpy as np

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
    def __init__(self, env, history=50000):

        # Environment and Action Space
        self.env = env
        self.action_space = env.action_space

        # Store History
        self.history = history
        self.past_obs = []
        self.past_action = []
        self.value_function =[0]

        # Episode number
        self._episode_number = 0
        self.minimum_exploration = 50


    def store_event(self, obs, action, done):

        self.past_obs.append(obs.tolist())
        self.past_action.append(action)
        if not done:
            self.value_function[self.episode_number()] += 1
        else:
            self.start_next_episode()
            self.value_function[self.episode_number()] = 1

    def start_next_episode(self):
        self._episode_number += 1

    def episode_number(self):
        return self._episode_number

    def current_episode_length(self):
        return self.value_function[-1]

    def history_full(self):
        return len(self.past_obs)==self.history

    def history_pop(self):
        self.past_obs.pop()
        self.past_action.pop()
        if(self.value_function[0]>1):
            self.value_function[0] -= 1
        else:
            self.value_function.pop(0)


    def policy(self):

        from sklearn.neighbors import KNeighborsRegressor

        # Models used
        classifier = KNeighborsRegressor(n_neighbors=2)

        # classifier
        classifier_input = np.array([[*e[0], e[1]] for e in zip(self.past_obs, self.past_action)])

        classifier_output_lst = []
        for episode, episode_range in enumerate(self.value_function):
            for i in range(episode_range):
                classifier_output_lst.append(episode_range - i - 1)
        classifier_output = np.array(classifier_output_lst)
        classifier.fit(classifier_input, classifier_output)
        return self.past_obs[-1], classifier

    def act(self, last_observation, last_action, done, policy='e-greedy'):

        if(last_action!=None):

            # Push last event
            self.store_event(last_observation, last_action, done)

            # Do minimum exploration
            if(self.minimum_exploration > sum(self.value_function)):
                return self.action_space.sample()

            # If all possible history to be known is known forget earliest event
            if(self.history_full()):
                self.history_pop()

            # Compute Policy
            information_state, current_policy = self.policy()

            # Calculate Expected Reward
            expected_reward = {}
            for action in range(self.action_space.__dict__['n']):
                expected_reward[action] = current_policy.predict([[*information_state, action]])[0]

            # Use policy for action selection
            if(policy=='greedy'):
                return (k for k in sorted(expected_reward, key=expected_reward.get, reverse=True)).__next__()
            else:
                if(np.random.random()<0.1):
                    return self.action_space.sample()
                else:
                    return (k for k in sorted(expected_reward, key=expected_reward.get, reverse=True)).__next__()
        else:
            return self.action_space.sample()


class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def run_game(agent_type):

    env = gym.make('CartPole-v0')
    agent_name = 'random_agent'
    outdir = '/tmp/'+agent_name+'_results'
    env.monitor.start(outdir, force=True, seed=0)
    episode_count = 300
    max_steps = 200

    if(agent_type is 'random'):
        agent = RandomAgent()
    else:
        agent = GreedyMemoryAgent(env)

    for i in range(episode_count):
        # Initial Values
        ob = env.reset()
        reward = 0.0
        done = False
        last_action = None

        for j in range(max_steps):
            action = agent.act(ob, reward, done, last_action)
            ob, reward, done, _ = env.step(action)
            last_action = action
            if done:
                break

    logger.info("Successfully ran {}. Now uploading results to the scoreboard".format(agent))
    env.monitor.close()
    gym.upload(outdir)
    return agent
