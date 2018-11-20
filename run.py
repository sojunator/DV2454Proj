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
        Z = sigmoid(np.dot(Z, layer))
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
    outputLayer = forwardPropagation(inputLayer, network)
    action = outputLayer.tolist().index(max(outputLayer.tolist()))

    rewards = []
    networks = []
    for episode in range(10000):
        network = createNetwork(dimensions)

        for i in range(100):
            observation, reward, done, info = env.step(action)

            inputLayer = observation.flatten()

            outputLayer = forwardPropagation(inputLayer, network)

            action = outputLayer.tolist().index(max(outputLayer.tolist()))

            if done:
                if i > 70:
                    rewards.append(i)
                    networks.append(network)
                break

        env.reset()


    highestIndex = rewards.index(max(rewards))

    for i in range(100):
        observation, reward, done, info = env.step(action)
        if done:
            print(i)
            break

        env.render()

        inputLayer = observation.flatten()

        outputLayer = forwardPropagation(inputLayer, networks[highestIndex])

        action = outputLayer.tolist().index(max(outputLayer.tolist()))

    print(rewards[highestIndex])
