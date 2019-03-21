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
from BrainDQNNature_CC import BrainDQNNature
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
SAVE_PATH = "./saved_parameters/double_dqn/"   # store network parameters and other parameters for pause.
STOP_STEP = 1500000.                    # the only way to exit training. 1,500,000 time steps.
DIR_NAME = '/double_dqn/'               # name of the log directory (be different with other networks).
REPLACE_TARGET_ITER = 500               # number of steps when target net parameters update

# BrainDoubleDQN: BrainDQNNature的改进版（防止过估计）      此处完全可以与BrainPrioritizedReplyDQN结合一下写出新的架构
class BrainDoubleDQN(BrainDQNNature):

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

        _, q_evals, self.lost = self.sess.run(
            [self.train_step, self.q_eval, self.cost],
            feed_dict={
                self.q_target : q_target,
                self.action_input : action_batch,
                self.eval_net_input : state_batch
        })
        self.lost_hist.append(self.lost)
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
        if self.timeStep == STOP_STEP:
            self._record_by_pic()