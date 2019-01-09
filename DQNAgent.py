import gym
import numpy as np
import itertools
import time
import random

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def takeAction(self, state):
        # Takes action given state
        # Action is either exploration or by network
        action = -1

        if np.random.rand() > self.epsilion:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.model.predict(state))

        return action

    def play(self):
        state = self.env.reset()
        total_reward = 0

        for _ in range(201):
            state = np.reshape(state, [1, self.dimensions["input"]])
            action = np.argmax(self.model.predict(state))

            # take action from either forwardPropagation or exploration
            self.env.render()
            new_state, reward, done, info = self.env.step(action)

            state = new_state
            total_reward += reward
            if done:
                return total_reward

        return total_reward

    def observe(self):
        state = self.env.reset()
        input = np.reshape(state, [1, self.dimensions["input"]])


        for _ in range(1000):
            action = self.takeAction(input)

            # take action from either forwardPropagation or exploration
            new_state, reward, done, info = self.env.step(action)

            # Save the information for training
            self.memory.append((new_state, action, state, reward, done))
            state = new_state
            if done:
                state = self.env.reset()


    def train(self):
        batch = random.sample(self.memory, self.batch_size)                         # Sample some moves

        inputs_shape = (self.batch_size, self.dimensions["input"])   #tate.shape[1:]
        inputs = np.zeros(inputs_shape)
        targets = np.zeros((self.batch_size, self.env.action_space.n))

        for i in range(0, self.batch_size):
            new_state = batch[i][0]
            action = batch[i][1]
            state = batch[i][2]
            reward = batch[i][3]
            done = batch[i][4]

        # Build Bellman equation for the Q function
            new_state = np.reshape(new_state, [1, self.dimensions["input"]])
            state = np.reshape(state, [1, self.dimensions["input"]])
            inputs[i:i+1] = np.expand_dims(state, axis=0)
            targets[i] = self.model.predict(state)
            Q_sa = self.model.predict(new_state)

            if done:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.gamma * np.max(Q_sa)

        # Train network to output the Q function

        self.model.train_on_batch(inputs, targets)

    def build_network(self):
        #Helper function, returns, does not assign
        model = Sequential()
        # setup input layer
        model.add(Dense(self.dimensions["neurons"], input_dim=self.dimensions["input"], activation='relu'))

        # Add hiddenlayers
        for _ in range(self.dimensions["hiddenLayers"]):
            model.add(Dense(self.dimensions["neurons"], activation='relu'))

        # add output layer
        model.add(Dense(self.dimensions["output"], activation='linear'))

        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def __init__(self, fileName = ""):
        self.memory = []
        self.env = gym.make("CartPole-v1")
        self.state = self.env.reset()

        self.inputLayer = self.state.flatten()
        self.actionSpace = self.env.action_space

        self.learning_rate = 0.001
        self.epsilion = 0.2
        self.gamma = 0.9
        self.batch_size = 50

        if fileName != "":
            self.model = load_model(fileName)
        else:
            self.dimensions = {}
            self.dimensions["hiddenLayers"] = 2
            self.dimensions["neurons"] = 24
            self.dimensions["input"] = len(self.inputLayer)
            self.dimensions["output"] = self.actionSpace.n

            self.model = self.build_network()

    def save_network(self, fileName):
        self.model.save(fileName)
        print("Saved {}".format(fileName))
