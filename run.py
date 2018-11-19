import gym
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(x))

def createNetwork(dimensions):
    network = []
    hiddenLayer = 2 * np.random.random((dimensions["inputLayer"], dimensions["inputLayer"] + 2)) - 1

    network.append(hiddenLayer)

    for i in range(dimensions["nrOfHiddenLayers"]):
        hiddenLayer = 2 * np.random.random((dimensions["inputLayer"] + 2, dimensions["inputLayer"] + 2)) - 1
        network.append(hiddenLayer)

    hiddenLayer = 2 * np.random.random((dimensions["inputLayer"] + 2, dimensions["outputLayer"])) - 1

    network.append(hiddenLayer)

    return network

def forwardPropagation(inputLayer, network):
    Z = inputLayer
    for layer in network:
        Z = np.dot(Z, layer)
    return Z


if __name__ == '__main__':

    env = gym.make("CartPole-v1")
    observation = env.reset()

    # Since observation can be of any dim, we first flatten the array
    inputLayer = observation.flatten()
    actionSpace = env.action_space

    dimensions = {}

    dimensions["inputLayer"] = len(inputLayer)
    dimensions["nrOfHiddenLayers"] = 1
    dimensions["outputLayer"] = actionSpace.n

    network = createNetwork(dimensions)
    action = forwardPropagation(inputLayer, network)
    print(action)
