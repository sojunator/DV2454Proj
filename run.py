import gym
import numpy as np


env = gym.make("CartPole-v1")
observation = env.reset()

# Since observation can be of any dim, we first flatten the array
inputLayer = observation.flatten()
actionSpace = env.action_space

dimensions = {}

dimensions["inputLayer"] = inputLayer
dimensions["actionSpace"] = actionSpace
dimensions["n"] = 2
dimensions["m"] = inputLayer + 2


def sigmoid(x):
    return 1 / (1 + np.exp(x))

def createNetwork(dimensions):
    weights = np.random.rand(dimensions["m"], dimensions["n"])
