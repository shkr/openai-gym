import logging
import os, sys
import random
import gym
import numpy as np

gym.scoreboard.api_key = "sk_hJMbMlYzSpSsucakOMOULg"

class EGreedyAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.e = 0.01
        self.q_matrix = np.random.uniform(0.0, 1.0, size=action_space.n)
        self.action_matrix = np.zeros(action_space.n)

    def bandit(self, act):
        """
        :param act : an action to ake
        :return: thi returns the expected score for this action
        """
        return self.q_matrix[act]

    def act(self, last_ob, last_reward, done):
        rnd = random.random()
        print("ob = {}, reward = {}, done = {}".format(last_ob, last_reward, done))
        if rnd < self.e:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_matrix)

    def step_size(self, act):
        return self.action_matrix[act]

    def update(self, ob, reward, act, done):
        # Update counter for the act
        self.action_matrix[act]+=1
        # Update expected reward for action
        self.q_matrix[act] = self.bandit(act) + (1.0/self.step_size(act))*(reward - self.bandit(act))

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make('CartPole-v0')
    agent_name = 'random_agent'
    outdir = '/tmp/'+agent_name+'_results'
    env.monitor.start(outdir, force=True, seed=0)

    agent = EGreedyAgent(env.action_space)
    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            action = agent.act(ob, reward, done)
            print("Action taken = {}".format(action))
            ob, reward, done, _ = env.step(action)
            agent.update(ob, reward, action, done)
            if done:
                break

    env.monitor.close()

    logger.info("Successfully ran {}. Now uploading results to the scoreboard")
    gym.upload(outdir)

