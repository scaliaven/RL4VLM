import sys; sys.path.append('/scratch/yx3038/RL4VLM/gym-junqi')
import gym
import gym_junqi

env = gym.make("gym_junqi:junqi-v0")

env.reset()
next_obs, reward, terminated, info = env.step(0)
import pdb; pdb.set_trace()