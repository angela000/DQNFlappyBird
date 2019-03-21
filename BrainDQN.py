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
FRAME_PER_ACTION = 1                        # number of frames per action.
BATCH_SIZE = 32                             # size of mini_batch.
OBSERVE = 1000.                             # 1000 steps to observe before training.
EXPLORE = 1000000.                          # 1000000 frames over which to anneal epsilon.
GAMMA = 0.99                                # decay rate of past observations.
FINAL_EPSILON = 0                           # final value of epsilon: 0.
INITIAL_EPSILON = 0.03                      # starting value of epsilon: 0.03.
REPLAY_MEMORY = 50000                       # number of previous transitions to remember.
SAVER_ITER = 10000                          # number of steps when save checkpoint.
SAVE_PATH = "./saved_parameters/dqn/"       # store network parameters and other parameters for pause.
RECORD_STEP = (1500000, 2000000, 2500000)   # the time steps to draw pics.
DIR_NAME = '/dqn/'                          # name of the log directory (be different with other networks).


# BrainDQN: DQN（记忆库）
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
        self.total_rewards_this_episode = 0
        self.rewards = []
        self.rewards_file_path = self.logs_path + 'reward.txt'
        self.q_targets = []
        self.q_targets_file_path = self.logs_path + 'q_target.txt'
        self.q_evals = []
        self.q_evals_file_path = self.logs_path + 'q_eval.txt'
        # init Q network
        self._createQNetwork()

    # observ != state. game环境可以给observ，但是state需要自己构造（最近的4个observ）
    def setPerception(self, nextObserv, action, reward, terminal, curScore):
        self.total_rewards_this_episode += reward
        # 把nextObserv放到最下面，把最上面的抛弃
        newState = np.append(self.currentState[:, :, 1:], nextObserv, axis = 2)
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        if len(self.replayMemory) > REPLAY_MEMORY:
            self.replayMemory.popleft()
        if self.onlineTimeStep > OBSERVE:
            # Train the network
            self._trainQNetwork()

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
            self.rewards.append(self.total_rewards_this_episode)
            self.total_rewards_this_episode = 0
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
        if self.epsilon > FINAL_EPSILON and self.onlineTimeStep > OBSERVE and self.epsilon > FINAL_EPSILON:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        return action


    def _createQNetwork(self):
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
        self.q_eval = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput),
                                 reduction_indices=1)                       # Q-value
        self.q_target = tf.placeholder("float", [None])                       # target-Q-value
        self.cost = tf.reduce_sum(tf.square(self.q_target - self.q_eval))     # cost
        self.trainStep = tf.train.AdamOptimizer(1e-6).minimize(self.cost)   # cost

        # load network and other parameters
        self._load_saved_parameters()

        # Evaluation: store the last ten episodes' scores
        self.counters = []

        # tensorboard
        tf.summary.FileWriter(self.logs_path, self.sess.graph)


    # load network and other parameters every SAVER_ITER
    def _load_saved_parameters(self):
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


    def _trainQNetwork(self):
        # Step1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step2: calculate q_target
        q_target = []
        QValue_batch = self.QValue.eval(
            feed_dict={
                self.stateInput: nextState_batch
            }
        )
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                q_target.append(reward_batch[i])
            else:
                q_target.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        _, q_eval, self.lost = self.sess.run(
            [self.trainStep, self.q_eval, self.cost],
            feed_dict={
                self.q_target : q_target,
                self.actionInput : action_batch,
                self.stateInput : state_batch
        })
        self.lost_hist.append(self.lost)
        self.q_targets.append(q_target)
        self.q_evals.append(q_eval)
        # save network and other data every 100,000 iteration
        if self.timeStep % 100000 == 0:
            self.saver.save(self.sess, SAVE_PATH + self.gameName, global_step=self.timeStep)
            saved_parameters_file = open(self.saved_parameters_file_path, 'wb')
            pickle.dump(self.gameTimes, saved_parameters_file)
            pickle.dump(self.timeStep, saved_parameters_file)
            pickle.dump(self.epsilon, saved_parameters_file)
            saved_parameters_file.close()
            self._save_lsrq_to_file()
        if self.timeStep in RECORD_STEP:
            self._record_by_pic()


    def setInitState(self, observ):
        self.currentState = np.stack((observ, observ, observ, observ), axis = 2)

    # Called at the record times.
    def _record_by_pic(self):
        self._save_lsrq_to_file()
        loss, scores, rewards, q_targets, q_evals = self._get_lsrq_from_file()
        plt.figure()
        plt.plot(loss, '-')
        plt.ylabel('loss')
        plt.xlabel('迭代次数')
        plt.savefig(self.logs_path + str(self.timeStep) + "_lost_hist_total.png")

        plt.figure()
        plt.plot(scores)
        plt.ylabel('score', '-')
        plt.xlabel('episode')
        plt.savefig(self.logs_path + str(self.timeStep) + "_scores_total.png")

        plt.figure()
        plt.plot(rewards, '-')
        plt.ylabel('rewards')
        plt.xlabel('episode')
        plt.savefig(self.logs_path + str(self.timeStep) + "_rewards_total.png")

        plt.figure()
        plt.plot(q_targets, '-')
        plt.ylabel('q_targets')
        plt.xlabel('迭代次数')
        plt.savefig(self.logs_path + str(self.timeStep) + "_q_targets_total.png")

        plt.figure()
        plt.plot(q_evals, '-')
        plt.ylabel('q_reals')
        plt.xlabel('迭代次数')
        plt.savefig(self.logs_path + str(self.timeStep) + "_q_reals_total.png")


    # save lost/score/reward/q_target/q_real to file
    def _save_lsrq_to_file(self):
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

        rewards_file = open(self.rewards_file_path, 'a')
        for r in self.rewards:
            rewards_file.write(str(r) + ' ')
        rewards_file.close()
        del self.rewards[:]

        q_target_file = open(self.q_targets_file_path, 'a')
        for qt in self.q_targets:
            q_target_file.write(str(qt) + ' ')
        q_target_file.close()
        del self.q_targets[:]

        q_eval_file = open(self.q_evals_file_path, 'a')
        for qe in self.q_evals:
            q_eval_file.write(str(qe) + ' ')
        q_eval_file.close()
        del self.q_evals[:]


    def _get_lsrq_from_file(self):
        scores_file = open(self.scores_file_path, 'r')
        scores_str = scores_file.readline().split(" ")
        scores_str = scores_str[0:-1]
        scores = list(map(eval, scores_str))
        scores_file.close()

        lost_hist_file = open(self.lost_hist_file_path, 'r')
        lost_hist_list_str = lost_hist_file.readline().split(" ")
        lost_hist_list_str = lost_hist_list_str[0:-1]
        loss = list(map(eval, lost_hist_list_str))
        lost_hist_file.close()

        rewards_file = open(self.rewards_file_path, 'r')
        rewards_str = rewards_file.readline().split(" ")
        rewards_str = rewards_str[0:-1]
        rewards = list(map(eval, rewards_str))
        rewards_file.close()

        q_target_file = open(self.q_targets_file_path, 'r')
        q_targets_str = q_target_file.readline().split(" ")
        q_targets_str = q_targets_str[0:-1]
        q_targets = list(map(eval, q_targets_str))
        q_target_file.close()

        q_evals_file = open(self.q_evals_file_path, 'r')
        q_evals_str = q_evals_file.readline().split(" ")
        q_evals_str = q_evals_str[0:-1]
        q_evals = list(map(eval, q_evals_str))
        q_evals_file.close()

        return loss, scores, rewards, q_targets, q_evals
