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

    # Convert outputlayer to a regular list
    return Z.tolist()


def testNetwork(network, nrOfSteps, env):
    # Create data for our first move
    observation = env.reset()

    # Convert data to useable format
    inputLayer = observation.flatten()

    # Calculate first move
    outputLayer = forwardPropagation(inputLayer, network)
    action = outputLayer.index(max(outputLayer))

    # Score for the network
    score = 0

    for i in range(nrOfSteps):
        observation, reward, done, info = env.step(action)

        score += reward

        if done:
            print("Done after {} episodes".format(i))
            break

        inputLayer = observation.flatten()

        outputLayer = forwardPropagation(inputLayer, network)
        action = outputLayer.index(max(outputLayer))

    return score

if __name__ == '__main__':

    env = gym.make("CartPole-v1")
    observation = env.reset()
    nrOfNetworks = 1000

    # Since observation can be of any dim, we first flatten the array
    inputLayer = observation.flatten()
    actionSpace = env.action_space

    dimensions = {}

    dimensions["inputLayer"] = len(inputLayer)
    dimensions["nrOfHiddenLayers"] = 1
    dimensions["outputLayer"] = actionSpace.n

    networks = []
    for i in range(nrOfNetworks):
        networks.append(createNetwork(dimensions))

    for network in networks:
        testNetwork(network, 201, env)


    env.close()
