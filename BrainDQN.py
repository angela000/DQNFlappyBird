#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import pickle
import sys
sys.path.append("game/")
import random
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters:
FRAME_PER_ACTION = 1                    # number of frames per action.
BATCH_SIZE = 32                         # size of mini_batch.
OBSERVE = 1000.                         # 1000 steps to observe before training.
EXPLORE = 1000000.                      # 1000000 frames over which to anneal epsilon.
GAMMA = 0.99                            # decay rate of past observations.
FINAL_EPSILON = 0                       # final value of epsilon: 0.
INITIAL_EPSILON = 0.03                  # starting value of epsilon: 0.03.
REPLAY_MEMORY = 50000                   # number of previous transitions to remember.
SAVER_ITER = 10000                      # number of steps when save checkpoint.
SAVE_PATH = "./saved_parameters/dqn/"   # store network parameters and other parameters for pause.
COUNTERS_SIZE = 10                      # calculate the average score for every 10 episodes.
STOP_STEP = 1500000.                    # the only way to exit training. 1,500,000 time steps.
DIR_NAME = '/dqn/'                      # name of the log directory (be different with other networks).


# Brain重要接口:
# getAction():      根据self.currentState选择action
# setPerception():  得到新observation之后进行记忆学习
class BrainDQN:
    def __init__(self, actionNum, gameName):
        self.actionNum = actionNum
        self.gameName = gameName
        # init replay memory
        self.replayMemory = deque()
        # init other parameters
        self.onlineTimeStep = 0
        # saved parameters every SAVER_ITER
        self.gameTimes = 0
        self.timeStep = 0
        self.epsilon = INITIAL_EPSILON
        self.saved_parameters_file_path = SAVE_PATH + self.gameName + '-saved-parameters.txt'
        # logs, append to file every SAVER_ITER
        self.logs_path = "./logs_" + self.gameName + DIR_NAME   # "logs_bird/dqn/"
        self.lost_hist = []
        self.lost_hist_file_path = self.logs_path + 'lost_hist.txt'
        self.scores = []
        self.scores_file_path = self.logs_path + 'scores.txt'
        # init Q network
        self.createQNetwork()

    def createQNetwork(self):
        # input layer
        self.stateInput = tf.placeholder("float", [None, 80, 80, 4])
        # conv layer 1
        W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
        b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))
        h_conv1 = tf.nn.conv2d(self.stateInput, W_conv1, strides=[1, 4, 4, 1], padding="SAME")
        h_relu1 = tf.nn.relu(h_conv1 + b_conv1)
        # [None, 20, 20, 32]
        h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # [None, 10, 10, 32]
        # conv layer 2
        W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))
        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME")
        h_relu2 = tf.nn.relu(h_conv2 + b_conv2)
        # [None, 5, 5, 64]
        # conv layer 3
        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
        b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))
        h_conv3 = tf.nn.conv2d(h_relu2, W_conv3, strides=[1, 1, 1, 1], padding="SAME")
        h_relu3 = tf.nn.relu(h_conv3 + b_conv3)
        # [None, 5, 5, 64]
        # reshape layer
        h_conv3_flat = tf.reshape(h_relu3, [-1, 1600])
        # [None, 1600]
        # full layer
        W_fc1 = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.01))
        b_fc1 = tf.Variable(tf.constant(0.01, shape=[512]))
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)
        # [None, 512]
        # reader layer 1
        W_fc2 = tf.Variable(tf.truncated_normal([512, self.actionNum], stddev=0.01))
        b_fc2 = tf.Variable(tf.constant(0.01, shape=[self.actionNum]))

        self.QValue = tf.matmul(h_fc1, W_fc2) + b_fc2
        # [None, 2]

        # build train network
        self.actionInput = tf.placeholder("float", [None, self.actionNum])  # [0 ,1] or [1, 0]
        Q_action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput),
                                 reduction_indices=1)                       # Q-value
        self.yInput = tf.placeholder("float", [None])                       # target-Q-value
        self.cost = tf.reduce_sum(tf.square(self.yInput - Q_action))        # cost
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)   # cost

        # load network and other parameters
        self.load_saved_parameters()

        # Evaluation: store the last ten episodes' scores
        self.counters = []

        # tensorboard
        tf.summary.FileWriter(self.logs_path, self.sess.graph)


    # load network and other parameters every SAVER_ITER
    def load_saved_parameters(self):
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(SAVE_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            # restore other params.
            if os.path.exists(self.saved_parameters_file_path) and os.path.getsize(self.saved_parameters_file_path) > 0:
                saved_parameters_file = open(self.saved_parameters_file_path, 'rb')
                self.gameTimes = pickle.load(saved_parameters_file)
                self.timeStep = pickle.load(saved_parameters_file)
                self.epsilon = pickle.load(saved_parameters_file)
                saved_parameters_file.close()
        else:
            # Re-train the network from zero.
            print("Could not find old network weights")

    def trainQNetwork(self):
        # Step1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step2: calculate y
        y_batch = []
        QValue_batch = self.QValue.eval(
            feed_dict={
                self.stateInput: nextState_batch
            }
        )
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        _, self.lost = self.sess.run(
            [self.trainStep, self.cost],
            feed_dict={
                self.yInput : y_batch,
                self.actionInput : action_batch,
                self.stateInput : state_batch
        })
        self.lost_hist.append(self.lost)

        # save network and other data every 100,000 iteration
        if self.timeStep % 100000 == 0:
            self.saver.save(self.sess, SAVE_PATH + self.gameName + '-dqn', global_step=self.timeStep)
            saved_parameters_file = open(self.saved_parameters_file_path, 'wb')
            pickle.dump(self.gameTimes, saved_parameters_file)
            pickle.dump(self.timeStep, saved_parameters_file)
            pickle.dump(self.epsilon, saved_parameters_file)
            saved_parameters_file.close()
            self.save_lost_and_score_to_file()
        if self.timeStep == STOP_STEP:
            self.end_the_game()


    # observ != state. game环境可以给observ，但是state需要自己构造（最近的4个observ）
    def setPerception(self, nextObserv, action, reward, terminal, curScore):
        # 把nextObserv放到最下面，把最上面的抛弃
        newState = np.append(self.currentState[:, :, 1:], nextObserv, axis = 2)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.onlineTimeStep > OBSERVE:
            # Train the network
            self.trainQNetwork()

        # print info
        if self.onlineTimeStep <= OBSERVE:
            state = "observe"
        elif self.onlineTimeStep > OBSERVE and self.onlineTimeStep <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ ACTION", action[1],"/ EPSILON", self.epsilon,
                  "/ REWARD", reward)

        if terminal:
            self.gameTimes += 1
            print("GAME_TIMES:" + str(self.gameTimes))
            self.scores.append(curScore)
        self.currentState = newState
        self.timeStep += 1
        self.onlineTimeStep += 1


    def getAction(self):
        QValue = self.QValue.eval(feed_dict = {self.stateInput: [self.currentState]})[0]
        action = np.zeros(self.actionNum)
        if self.timeStep % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                action_index = random.randrange(self.actionNum)
                action[action_index] = 1
            else:
                action_index = np.argmax(QValue)
                action[action_index] = 1
        else:
            action[0] = 1

        # change episilon
        if self.epsilon > FINAL_EPSILON and self.onlineTimeStep > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action


    def setInitState(self, observ):
        self.currentState = np.stack((observ, observ, observ, observ), axis = 2)

    # Called when the game ends.
    def end_the_game(self):
        self.save_lost_and_score_to_file()
        self.get_lost_and_score_from_file()
        plt.figure()
        plt.plot(self.lost_hist)
        plt.ylabel('lost')
        plt.savefig(self.logs_path + "lost_hist_total.png")

        plt.figure()
        plt.plot(self.scores)
        plt.ylabel('score')
        plt.savefig(self.logs_path + "scores_total.png")

    def save_lost_and_score_to_file(self):
        list_hist_file = open(self.lost_hist_file_path, 'a')
        for l in self.lost_hist:
            list_hist_file.write(str(l) + ' ')
        list_hist_file.close()
        del self.lost_hist[:]
        scores_file = open(self.scores_file_path, 'a')
        for s in self.scores:
            scores_file.write(str(s) + ' ')
        scores_file.close()
        del self.scores[:]

    def get_lost_and_score_from_file(self):
        scores_file = open(self.scores_file_path, 'r')
        scores_str = scores_file.readline().split(" ")
        scores_str = scores_str[0:-1]
        self.scores = list(map(eval, scores_str))
        scores_file.close()

        lost_hist_file = open(self.lost_hist_file_path, 'r')
        lost_hist_list_str = lost_hist_file.readline().split(" ")
        lost_hist_list_str = lost_hist_list_str[0:-1]
        self.lost_hist = list(map(eval, lost_hist_list_str))
        # self.lost_hist = map(lambda x: float(x), lost_hist_list)
        lost_hist_file.close()

