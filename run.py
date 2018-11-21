import gym
import numpy as np

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
    # Do the same procedure again, but take the other halfs
    for mLayer, fLayer in zip(mother, father):
        child.append(np.concatenate((np.hsplit(mLayer, split)[1],
                                     np.hsplit(fLayer, split)[1]), 1))

    children.append(child)
    return children


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

    # Scores will contain all scores for the respektive network
    networks = []
    scores = []

    # Create our networks and store them in networks list
    for i in range(nrOfNetworks):
        networks.append(createNetwork(dimensions))

    children = genetics(networks[0], networks[1], 2)
    
    for i in range(1000):
        for network in networks:
            scores.append(testNetwork(network, 201, env))

        # Find networks that are good for mutation and breeding
        # If a network does not pass criteria, it get removed
        # criteria is currently being larger than average in score
        averageScore = sum(scores) / len(scores)
        print(averageScore)
        removedNetWorkCounter = 0
        for score in scores:
            if score < averageScore:
                index = scores.index(score)

                networks.pop(index)
                scores.pop(index)

                removedNetWorkCounter += 1

        # Apply genetics and breeding

        #Generate new networks to refill the pool
        for i in range(removedNetWorkCounter):
            networks.append(createNetwork(dimensions))

        # reset scores
        del scores[:]
    env.close()
