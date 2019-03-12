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
FRAME_PER_ACTION = 1            # number of frames per action.
BATCH_SIZE = 32                 # size of mini_batch.
OBSERVE = 1000.                 # 1000 steps to observe before training.
EXPLORE = 1500000.              # 1500000 frames over which to anneal epsilon.
GAMMA = 0.99                    # decay rate of past observations.
FINAL_EPSILON = 0               # final value of epsilon.
INITIAL_EPSILON = 0.05          # starting value of epsilon.
REPLAY_MEMORY = 50000           # number of previous transitions to remember.
SAVER_ITER = 10000              # number of steps when save checkpoint.

COUNTERS_SIZE = 10              # calculate the average score for every 10 episodes
average_score = []              # store the average score of ten last episodes.
AVERAGE_SIZE = 20              # store the pic for every 500 average scores
DIR_NAME = '/dqn/'              # name of the log directory (be different with other networks)
SAVE_PATH = "./saved_parameters/dqn/"
SAVE_BACK_PATH = "./saved_back/dqn/"


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
        self.timeStep = 0                                   # 总共第timeStep次train
        self.onlineTimeStep = 0                             # 连续第onlineTimeStep次train
        self.gameTimes = 0                                  # 第gameTimes次游戏
        self.epsilon = INITIAL_EPSILON
        self.records = {
            "first_1_score_game_time": 0,
            "first_2_score_game_time": 0,
            "first_3_score_game_time": 0,
            "first_5_score_game_time": 0,
            "first_10_score_game_time": 0,
            "first_20_score_game_time": 0,
            "first_50_score_game_time": 0,
            "max_score": 0,
            "max_score_game_time": 0
        }
        self.lost_hist = []
        self.logs_path = "./logs_" + self.gameName + "/dqn/"
        self.game_times_file_path = SAVE_PATH + self.gameName + '-game-times'
        self.records_file_path = SAVE_PATH + self.gameName + '-records'
        self.lost_hist_file_path = SAVE_PATH + self.gameName + 'lost_hist'
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

        # load network parameters
        self.load_network_parameters()

        # Evaluation: store the last ten episodes' scores
        self.counters = []

        # tensorboard
        tf.summary.FileWriter(self.logs_path, self.sess.graph)


    # 恢复神经网络 / replay memory / game_times / epsilon等数据
    def load_network_parameters(self):
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state(SAVE_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            # 根据model_checkpoint_path得到step
            path_ = checkpoint.model_checkpoint_path
            self.timeStep = int((path_.split('-'))[-1])
            if self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
                self.epsilon = INITIAL_EPSILON - (self.timeStep - OBSERVE) * \
                               (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            else:
                self.epsilon = FINAL_EPSILON
            # restore other params.
            if os.path.exists(self.game_times_file_path) and os.path.getsize(self.game_times_file_path) > 0:
                self.gameTimes = pickle.load(open(self.game_times_file_path, 'rb'))
            if os.path.exists(self.records_file_path + "_timeStep_" + str(self.timeStep)) and \
                    os.path.getsize(self.records_file_path + "_timeStep_" + str(self.timeStep)) > 0:
                self.records = pickle.load(open(self.records_file_path + "_timeStep_" + str(self.timeStep), 'rb'))
            if os.path.exists(self.lost_hist_file_path) and os.path.getsize(self.lost_hist_file_path) > 0:
                self.lost_hist = pickle.load(open(self.lost_hist_file_path, 'rb'))
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
            pickle.dump(self.gameTimes, open(self.game_times_file_path, 'wb'))
            pickle.dump(self.records, open(self.records_file_path + "_timeStep_" + str(self.timeStep), 'wb'))
            pickle.dump(self.lost_hist, open(self.lost_hist_file_path, 'wb'))
        # pic the loss function every 1,000,000 iteration
        if self.timeStep % 1000000 == 0:
            plt.figure()
            plt.plot(self.lost_hist)
            plt.ylabel('lost')
            plt.savefig(self.lost_hist_file_path + "_steps_" + str(self.timeStep)
                        + "_lost_hist.png")
            del self.lost_hist[:]

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
            self.counter_add(self.counters, curScore, self.timeStep, self.gameTimes)
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


    # 当一局游戏结束，会调用此方法进行记录
    def counter_add(self, counters, curScore, steps, gameTimes):
        print("GAME_TIMES:" + str(gameTimes))
        self.update_records(curScore, gameTimes)
        counters.append(curScore)
        # 每10局游戏就计算平均得分并存下
        if len(counters) == COUNTERS_SIZE:  # 10
            average_score.append(np.mean(counters))
            # 每5000局游戏画一张图（500个点），主要是要有max_size个点. 计划训练3000局。
            if len(average_score) >= AVERAGE_SIZE:
                plt.figure()
                plt.plot(average_score)
                plt.ylabel('score')
                plt.savefig(self.logs_path + "steps_" + str(steps) + '_gameTimes_' + str(gameTimes) + "_average_score.png")
                fo = open(self.logs_path + "steps_" + str(steps) + '_gameTimes_' + str(gameTimes)  + "_average_score.txt", "w")
                fo.write(str(average_score))
                fo.close()
                del average_score[:]
            del counters[:]

    def update_records(self, curScore, gameTimes):
        if curScore == 1 and self.records["first_1_score_game_time"] == 0:
            self.records["first_1_score_game_time"] = gameTimes
        elif curScore == 2 and self.records["first_2_score_game_time"] == 0:
            self.records["first_2_score_game_time"] = gameTimes
        elif curScore >= 3 and curScore < 5 and self.records["first_3_score_game_time"] == 0:
            self.records["first_3_score_game_time"] = gameTimes
        elif curScore >= 5 and curScore < 10 and self.records["first_5_score_game_time"] == 0:
            self.records["first_5_score_game_time"] = gameTimes
        elif curScore >= 10 and curScore < 50 and self.records["first_10_score_game_time"] == 0:
            self.records["first_10_score_game_time"] = gameTimes
        elif curScore >= 50 and self.records["first_50_score_game_time"] == 0:
            self.records["first_50_score_game_time"] = gameTimes
        if curScore > self.records["max_score"]:
            self.records["max_score"] = curScore
            self.records["max_score_game_time"] = gameTimes
