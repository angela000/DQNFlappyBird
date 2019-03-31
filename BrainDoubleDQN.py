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
from BrainDQNNature import BrainDQNNature
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
RECORD_STEP = (500000, 1000000, 1500000, 2000000, 2500000)       # the time steps to draw pics.
REPLACE_TARGET_ITER = 500                       # number of steps when target net parameters update

# BrainDoubleDQN: BrainDQNNature的改进版（防止过估计产生的较差策略）
class BrainDoubleDQN(BrainDQNNature):

    def _setDirName(self):
        self.dir_name = '/double_dqn/'

    def trainQNetwork(self):
        # Train the network
        if self.timeStep % REPLACE_TARGET_ITER == 0:
            self.sess.run(self.target_replace_op)
        # Step1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step2: calculate q_target
        q_target = []
        '''Double DQN'''
        readout_j1_batch = self.readout_t.eval(feed_dict={self.target_net_input: next_state_batch})  # (32, 2)
        readout_j1_batch_for_action = self.readout_e.eval(feed_dict={self.eval_net_input: next_state_batch})  # (32, 2)
        max_act4next = np.argmax(readout_j1_batch_for_action, axis=1)
        selected_q_next = readout_j1_batch[range(len(max_act4next)), max_act4next]  # (batch_size, 1)
        '''Double DQN'''
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                q_target.append(reward_batch[i])
            else:
                q_target.append(reward_batch[i] + GAMMA * selected_q_next[i])

        _, self.lost = self.sess.run(
            [self.train_step, self.cost],
            feed_dict={
                self.q_target : q_target,
                self.action_input : action_batch,
                self.eval_net_input : state_batch
        })
        self.lost_hist.append(self.lost)
        self.q_target_list.append(q_target)
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