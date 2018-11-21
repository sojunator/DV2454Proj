import gym
import numpy as np
import itertools

def sigmoid(x):
    return 1 / (1 + np.exp(x))

def genetics(mother, father, split):
    # Will be a list containing the new networks
    children = []
    child = []

    for mLayer, fLayer in zip(mother, father):
        # Concatenate builds a new matrix from the two sub matrixes
        # Split divides the inputted matrix into two matrices, returned to a list
        child.append(np.concatenate((np.hsplit(mLayer, split)[0],
                                     np.hsplit(fLayer, split)[0]), 1))


    children.append(child)
    # We have to clear child, otherwise the structure return yields a incorrect
    # network
    child.clear()

    # Do the same procedure again, but take the other halfs
    for mLayer, fLayer in zip(mother, father):
        child.append(np.concatenate((np.hsplit(mLayer, split)[1],
                                     np.hsplit(fLayer, split)[1]), 1))

    children.append(child)
    return children


def createNetwork(dimensions):
    network = []
    # Fist layer is a corner case, as we need to have correct dims for the input
    hiddenLayer = 2 * np.random.random((dimensions["inputLayer"], dimensions["inputLayer"] + 2)) - 1

    network.append(hiddenLayer)

    for i in range(dimensions["nrOfHiddenLayers"]):
        # Usage of "InputLayer" key is not accurate, as it's actually the number of
        # Neurons
        hiddenLayer = 2 * np.random.random((dimensions["inputLayer"] + 2, dimensions["inputLayer"] + 2)) - 1
        network.append(hiddenLayer)

    # Output layer is also a corner case, as it needs to result in a 1xNumberOfActions Matrix
    hiddenLayer = 2 * np.random.random((dimensions["inputLayer"] + 2, dimensions["outputLayer"])) - 1

    network.append(hiddenLayer)

    return network

def forwardPropagation(inputLayer, network):
    Z = inputLayer
    for layer in network:
        try:
            Z = sigmoid(np.dot(Z, layer))
        except ValueError:
            print("Z {}".format(Z.shape))
            print("Layer {}\n\n\n\n".format(layer.shape))

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
            break

        inputLayer = observation.flatten()

        outputLayer = forwardPropagation(inputLayer, network)
        action = outputLayer.index(max(outputLayer))

    return score

if __name__ == '__main__':

    env = gym.make("CartPole-v1")
    observation = env.reset()
    nrOfNetworks = 1000
    generations = 1000
    keepPerGen = 100

    # Since observation can be of any dim, we first flatten the array
    inputLayer = observation.flatten()
    actionSpace = env.action_space

    dimensions = {}

    dimensions["inputLayer"] = len(inputLayer)
    dimensions["nrOfHiddenLayers"] = 1
    dimensions["outputLayer"] = actionSpace.n

    # Contains a tuple with score and the network, score starts at zero
    networks = []

    # Create our networks and store them in networks list
    for i in range(nrOfNetworks):
        networks.append([0, createNetwork(dimensions)])

    children = genetics(networks[0][1], networks[1][1], 2)
    networks.append([0, children[0]])
    networks.append([0, children[1]])


    for i in range(generations):
        for network in networks:
            network[0] = testNetwork(network[1], 201, env)


        # Sort networks by performance
        # Keep top 100
        networks.sort(key=lambda x : x[0])
        networks = networks[-keepPerGen:]

        # Apply genetics and breeding
        tempPairs = list(itertools.combinations(networks, 2))

        # iterate over 2 networks pairs, and create two new children
        # Merge these into our network list

        #Generate new networks to refill the pool
        for i in range(nrOfNetworks - len(networks)):
            networks.append([0,createNetwork(dimensions)])


    env.close()
