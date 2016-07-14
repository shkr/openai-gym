import logging
import os, sys

import gym


gym.scoreboard.api_key = "sk_hJMbMlYzSpSsucakOMOULg"

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    env = gym.make('CartPole-v0')
    agent_name = 'random_agent'
    outdir = '/tmp/'+agent_name+'_results'
    env.monitor.start(outdir, force=True, seed=0)

    agent = RandomAgent(env.action_space)
    episode_count = 100
    max_steps = 200
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()

        for j in range(max_steps):
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

    env.monitor.close()

    logger.info("Successfully ran {}. Now uploading results to the scoreboard")
    gym.upload(outdir)

