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
mlp.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
from collections import deque
from BrainDQN import BrainDQN
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters:
FRAME_PER_ACTION = 1                            # number of frames per action.
BATCH_SIZE = 32                                 # size of mini_batch.
OBSERVE = 1000.                                 # 1000 steps to observe before training.
EXPLORE = 1000000.                              # 1000000 frames over which to anneal epsilon.
GAMMA = 0.99                                    # decay rate of past observations.
FINAL_EPSILON = 0                               # final value of epsilon: 0.
INITIAL_EPSILON = 0.03                          # starting value of epsilon: 0.03.
REPLAY_MEMORY = 50000                           # number of previous transitions to remember.
SAVER_ITER = 10000                              # number of steps when save checkpoint.
RECORD_STEP = (500000, 1000000,1500000, 2000000, 2500000)    # the time steps to draw pics.
REPLACE_TARGET_ITER = 500                       # number of steps when target net parameters update


# BrainDQNNature: BrainDQN + fixed target
class BrainDQNNature(BrainDQN):
    def _createQNetwork(self):
        # input layer
        with tf.variable_scope("eval_net"):
            self.eval_net_input = tf.placeholder("float", [None, 80, 80, 4])
            # conv layer 1
            W_conv1_e = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
            b_conv1_e = tf.Variable(tf.constant(0.01, shape=[32]))

            h_conv1_e = tf.nn.conv2d(self.eval_net_input, W_conv1_e, strides=[1, 4, 4, 1], padding="SAME")
            h_relu1_e = tf.nn.relu(h_conv1_e + b_conv1_e)  # [None, 20, 20, 32]
            h_pool1_e = tf.nn.max_pool(h_relu1_e, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # [None, 10, 10, 32]
            # conv layer 2
            W_conv2_e = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
            b_conv2_e = tf.Variable(tf.constant(0.01, shape=[64]))

            h_conv2_e = tf.nn.conv2d(h_pool1_e, W_conv2_e, strides=[1, 2, 2, 1], padding="SAME")
            h_relu2_e = tf.nn.relu(h_conv2_e + b_conv2_e)  # [None, 5, 5, 64]
            # conv layer 3
            W_conv3_e = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
            b_conv3_e = tf.Variable(tf.constant(0.01, shape=[64]))

            h_conv3_e = tf.nn.conv2d(h_relu2_e, W_conv3_e, strides=[1, 1, 1, 1], padding="SAME")
            h_relu3_e = tf.nn.relu(h_conv3_e + b_conv3_e)  # [None, 5, 5, 64]
            # full layer
            W_fc1_e = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.01))
            b_fc1_e = tf.Variable(tf.constant(0.01, shape=[512]))

            h_conv3_flat_e = tf.reshape(h_relu3_e, [-1, 1600])  # [None, 1600]
            h_fc1_e = tf.nn.relu(tf.matmul(h_conv3_flat_e, W_fc1_e) + b_fc1_e)  # [None, 512]
            # reader layer 1
            W_fc2_e = tf.Variable(tf.truncated_normal([512, self.actionNum], stddev=0.01))
            b_fc2_e = tf.Variable(tf.constant(0.01, shape=[self.actionNum]))

            self.readout_e = tf.matmul(h_fc1_e, W_fc2_e) + b_fc2_e  # [None, 2]

        '''DQNNature'''
        with tf.variable_scope("target_net"):
            self.target_net_input = tf.placeholder("float", [None, 80, 80, 4])
            # conv layer 1
            W_conv1_t = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01), trainable=False)
            b_conv1_t = tf.Variable(tf.constant(0.01, shape=[32]), trainable=False)

            h_conv1_t = tf.nn.conv2d(self.target_net_input, W_conv1_t, strides=[1, 4, 4, 1], padding="SAME")
            h_relu1_t = tf.nn.relu(h_conv1_t + b_conv1_t)  # [None, 20, 20, 32]
            h_pool1_t = tf.nn.max_pool(h_relu1_t, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # [None, 10, 10, 32]
            # conv layer 2
            W_conv2_t = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01), trainable=False)
            b_conv2_t = tf.Variable(tf.constant(0.01, shape=[64]), trainable=False)

            h_conv2_t = tf.nn.conv2d(h_pool1_t, W_conv2_t, strides=[1, 2, 2, 1], padding="SAME")
            h_relu2_t = tf.nn.relu(h_conv2_t + b_conv2_t)  # [None, 5, 5, 64]
            # conv layer 3
            W_conv3_t = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01), trainable=False)
            b_conv3_t = tf.Variable(tf.constant(0.01, shape=[64]), trainable=False)

            h_conv3_t = tf.nn.conv2d(h_relu2_t, W_conv3_t, strides=[1, 1, 1, 1], padding="SAME")
            h_relu3_t = tf.nn.relu(h_conv3_t + b_conv3_t)  # [None, 5, 5, 64]
            # full layer
            W_fc1_t = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.01), trainable=False)
            b_fc1_t = tf.Variable(tf.constant(0.01, shape=[512]), trainable=False)

            h_conv3_flat_t = tf.reshape(h_relu3_t, [-1, 1600])  # [None, 1600]
            h_fc1_t = tf.nn.relu(tf.matmul(h_conv3_flat_t, W_fc1_t) + b_fc1_t)  # [None, 512]
            # reader layer 1
            W_fc2_t = tf.Variable(tf.truncated_normal([512, self.actionNum], stddev=0.01), trainable=False)
            b_fc2_t = tf.Variable(tf.constant(0.01, shape=[self.actionNum]), trainable=False)

            self.readout_t = tf.matmul(h_fc1_t, W_fc2_t) + b_fc2_t  # [None, 2]

            # parameter transfer
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

            with tf.variable_scope('soft_replacement'):
                self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            # build train network
            self.action_input = tf.placeholder("float", [None, self.actionNum])
            self.q_target = tf.placeholder("float", [None])

            self.q_eval = tf.reduce_sum(tf.multiply(self.readout_e, self.action_input), axis=1)  # [None]
            # readout_action -- reward of selected action by a.
            self.cost = tf.reduce_mean(tf.square(self.q_target - self.q_eval))
            self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

            # load network and other parameters
            self._load_saved_parameters()

    def _setDirName(self):
        self.dir_name = '/dqn_nature/'

    def getAction(self):
        '''DoubleDQN'''
        QValue = self.readout_e.eval(feed_dict={self.eval_net_input: [self.currentState]})[0]
        '''DoubleDQN'''
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

    def _trainQNetwork(self):
        # Train the network
        if self.timeStep % REPLACE_TARGET_ITER == 0:
            self.sess.run(self.target_replace_op)

        # Step1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]

        # Step2: calculate y
        q_targets = []
        '''DoubleDQN'''
        target_q_value = self.readout_t.eval(
            feed_dict={
                self.target_net_input: nextState_batch
            }
        )
        '''DoubleDQN'''
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                q_targets.append(reward_batch[i])
            else:
                q_targets.append(reward_batch[i] + GAMMA * np.max(target_q_value[i]))

        _, self.lost = self.sess.run(
            [self.train_step, self.cost],
            feed_dict={
                self.q_target : q_targets,
                self.action_input : action_batch,
                self.eval_net_input : state_batch
        })
        self.lost_hist.append(self.lost)
        self.q_target_list.append(q_targets)

        # save network and other data every 100,000 iteration
        if self.timeStep % 100000 == 0:
            self.saver.save(self.sess, self.save_path + self.gameName, global_step=self.timeStep)
            saved_parameters_file = open(self.saved_parameters_file_path, 'wb')
            pickle.dump(self.gameTimes, saved_parameters_file)
            pickle.dump(self.timeStep, saved_parameters_file)
            pickle.dump(self.epsilon, saved_parameters_file)
            saved_parameters_file.close()
            self._save_loss_score_timestep_reward_qtarget_to_file()
        if self.timeStep in RECORD_STEP:
            self._record_by_pic()
