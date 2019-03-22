#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import pickle
import sys
sys.path.append("game/")
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
from BrainPrioritizedReplyDQN_CC import BrainPrioritizedReplyDQN
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
SAVE_PATH = "./saved_parameters/double_prioritized_reply_dqn/"   # store network parameters and other parameters for pause.
RECORD_STEP = (1500000, 2000000, 2500000)       # the time steps to draw pics.
DIR_NAME = '/double_prioritized_reply_dqn/'     # name of the log directory (be different with other networks).
N_FEATURES = 80 * 80 * 4                        # number of features



class BrainDoublePrioritizedReplyDQN(BrainPrioritizedReplyDQN):

    def _setDirName(self):
        self.dir_name = '/double_prioritized_reply_dqn/'

    def createQNetwork(self):
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
        '''>>>>>>PrioritizedReplyDQN'''
        self.ISWeights = tf.placeholder(tf.float32, [None, 1])
        '''PrioritizedReplyDQN<<<<<<'''
        self.q_eval = tf.reduce_sum(tf.multiply(self.readout_e, self.action_input), axis=1)  # [None]
        # readout_action -- reward of selected action by a.
        self.abs_errors = tf.abs(self.q_target - self.q_eval)
        '''>>>>>>PrioritizedReplyDQN'''
        self.cost = tf.reduce_mean(self.ISWeights * tf.square(self.q_target - self.q_eval))
        '''PrioritizedReplyDQN<<<<<<'''
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

        # load network and other parameters
        self._load_saved_parameters()


    def _trainQNetwork(self):
        # Step1: obtain priority minibatch from replay memory
        # sample a minibatch(32) to train on
        tree_idx, batch_memory, ISWeights_value = self.replayMemory.sample(BATCH_SIZE)
        # get the batch variables
        # (d[0], d[1], d[2], d[3], d[4])
        # (observation, a_t, r_t, terminal, observation_)
        state_batch = np.reshape(batch_memory[:, :N_FEATURES], (BATCH_SIZE, 80, 80, 4))
        action_batch = batch_memory[:, N_FEATURES:N_FEATURES+2].astype(int)
        reward_batch = batch_memory[:, N_FEATURES + 2]
        terminal_batch = batch_memory[:, N_FEATURES + 3]
        next_state_batch = np.reshape(batch_memory[:, -N_FEATURES:], (BATCH_SIZE, 80, 80, 4))

        # Step2: calculate q_target
        q_target = []
        '''Double DQN'''
        readout_j1_batch = self.readout_t.eval(feed_dict={self.target_net_input: next_state_batch})  # (32, 2)
        readout_j1_batch_for_action = self.readout_e.eval(feed_dict={self.eval_net_input: next_state_batch})  # (32, 2)
        max_act4next = np.argmax(readout_j1_batch_for_action, axis=1)
        selected_q_next = readout_j1_batch[range(len(max_act4next)), max_act4next]  # (batch_size, 1)
        '''Double DQN'''
        for i in range(0, BATCH_SIZE):
            terminal = terminal_batch[i]
            if terminal:
                q_target.append(reward_batch[i])
            else:
                q_target.append(reward_batch[i] + GAMMA * selected_q_next[i])

        _, abs_errors, q_evals, loss = self.sess.run(
            [self.train_step, self.abs_errors, self.q_eval, self.cost],
            feed_dict={
                self.q_target : q_target,
                self.action_input : action_batch,
                self.eval_net_input : state_batch,
                self.ISWeights: ISWeights_value
        })
        self.replayMemory.batch_update(tree_idx, abs_errors)
        self.lost_hist.append(loss)
        self.q_targets.append(q_target)
        self.q_evals.append(q_evals)

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


    def setPerception(self, nextObserv, action, reward, terminal, curScore):
        self.total_rewards_this_episode += reward
        # 把nextObserv放到最下面，把最上面的抛弃
        newState = np.append(self.currentState[:, :, 1:], nextObserv, axis=2)
        # store the last 50000(REPLAY_MEMORY) transitions in memory
        terminal_int = self.transform_terminal(terminal)
        transition = np.hstack((self.currentState.flatten(), action, reward, terminal_int, newState.flatten()))
        self.replayMemory.store(transition)  # have high priority for newly arrived transition
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

        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ ACTION", action[1], "/ EPSILON", self.epsilon,
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

    def transform_terminal(self, terminal):
        if terminal:
            return 1
        else:
            return 0


