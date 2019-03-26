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
RECORD_STEP = (500000, 1000000, 1500000, 2000000, 2500000)   # the time steps to draw pics.

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
        self.epsilon = INITIAL_EPSILON          # 需要在这里判断self的具体的类。才能知道文件夹是哪个。
        self._setDirName()
        # logs, append to file every SAVER_ITER
        self.save_path = "./saved_parameters" + self.dir_name  # store network parameters and other parameters for pause.
        self.saved_parameters_file_path = self.save_path + self.gameName + '-saved-parameters.txt'
        self.logs_path = "./logs_" + self.gameName + self.dir_name   # "logs_bird/dqn/"

        self.lost_hist = []
        self.lost_hist_file_path = self.logs_path + 'lost_hist.txt'
        self.q_target = []
        self.q_target_file_path = self.logs_path + 'q_targets.txt'
        self.score_every_episode = []
        self.score_every_episode_file_path = self.logs_path + 'score_every_episode.txt'
        self.time_steps_when_episode_end = []
        self.time_steps_when_episode_end_file_path = self.logs_path + 'time_steps_when_episode_end.txt'
        self.reward_every_time_step = []
        self.reward_every_time_step_file_path = self.logs_path + 'reward_every_time_step.txt'
        # init Q network
        self._createQNetwork()

    def _setDirName(self):
        self.dir_name = "/dqn/"


    def setPerception(self, nextObserv, action, reward, terminal, curScore):
        # 把nextObserv放到最下面，把最上面的抛弃
        newState = np.append(self.currentState[:, :, 1:], nextObserv, axis = 2)
        transition = (self.currentState, action, reward, newState, terminal)
        self.replayMemory.append(transition)
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

        print("TIMESTEP", self.timeStep, "/ STATE", state, "/ ACTION", action[1], "/ EPSILON", self.epsilon,
                  "/ REWARD", reward, "/ SCORE", curScore)

        self.reward_every_time_step.append(reward)
        if terminal:
            self.gameTimes += 1
            print("GAME_TIMES:" + str(self.gameTimes))
            self.score_every_episode.append(curScore)
            self.time_steps_when_episode_end.append(self.timeStep)
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
        checkpoint = tf.train.get_checkpoint_state(self.save_path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            # restore other params.
            if os.path.exists(self.saved_parameters_file_path) and os.path.getsize(self.saved_parameters_file_path) > 0:
                with open(self.saved_parameters_file_path, 'rb') as saved_parameters_file:
                    self.gameTimes = pickle.load(saved_parameters_file)
                    self.timeStep = pickle.load(saved_parameters_file)
                    self.epsilon = pickle.load(saved_parameters_file)
        else:
            # Re-train the network from zero.
            print("Could not find old network weights")


    def _trainQNetwork(self):
        # Step1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replayMemory, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step2: calculate q_target
        q_target = []
        QValue_batch = self.QValue.eval(
            feed_dict={
                self.stateInput: next_state_batch
            }
        )
        for i in range(0, BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                q_target.append(reward_batch[i])
            else:
                q_target.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

        _, self.lost = self.sess.run(
            [self.trainStep, self.cost],
            feed_dict={
                self.q_target : q_target,
                self.actionInput : action_batch,
                self.stateInput : state_batch
        })
        self.lost_hist.append(self.lost)
        self.q_target.append(q_target)
        # save network and other data every 100,000 iteration
        if self.timeStep % 100000 == 0:
            self.saver.save(self.sess, self.save_path + self.gameName, global_step=self.timeStep)
            with open(self.saved_parameters_file_path, 'wb') as saved_parameters_file:
                pickle.dump(self.gameTimes, saved_parameters_file)
                pickle.dump(self.timeStep, saved_parameters_file)
                pickle.dump(self.epsilon, saved_parameters_file)
            self._save_loss_score_timestep_reward_qtarget_to_file()
        if self.timeStep in RECORD_STEP:
            self._record_by_pic()


    def setInitState(self, observ):
        self.currentState = np.stack((observ, observ, observ, observ), axis = 2)

    # Called at the record times.
    def _record_by_pic(self):
        self._save_loss_score_timestep_reward_qtarget_to_file()
        loss, scores, time_step_when_episode_end, reward_every_time_step, q_target = self._get_loss_score_timestep_reward_qtarget_from_file()
        plt.figure()
        plt.plot(loss, '-')
        plt.ylabel('loss')
        plt.xlabel('time_step')
        plt.savefig(self.logs_path + str(self.timeStep) + "_lost_hist_total.png")

        plt.figure()
        plt.plot(scores, '-')
        plt.ylabel('score')
        plt.xlabel('episode')
        plt.savefig(self.logs_path + str(self.timeStep) + "_scores_episode_total.png")

        plt.figure()
        plt.plot(q_target, '-')
        plt.ylabel('q target')
        plt.xlabel('time_step')
        plt.savefig(self.logs_path + str(self.timeStep) + "_q_target_total.png")

        plt.figure()
        plt.plot(time_step_when_episode_end, scores, '-')
        plt.ylabel('score')
        plt.xlabel('time_step')
        plt.savefig(self.logs_path + str(self.timeStep) + "_scores_time_step_total.png")

    # save loss/score/time_step/reward/q_target to file
    def _save_loss_score_timestep_reward_qtarget_to_file(self):
        with open(self.lost_hist_file_path, 'a') as lost_hist_file:
            for l in self.lost_hist:
                lost_hist_file.write(str(l) + ' ')
        del self.lost_hist[:]

        with open(self.score_every_episode_file_path, 'a') as score_every_episode_file:
            for s in self.score_every_episode:
                score_every_episode_file.write(str(s) + ' ')
        del self.score_every_episode[:]

        with open(self.time_steps_when_episode_end_file_path, 'a') as time_step_when_episode_end_file:
            for t in self.time_steps_when_episode_end:
                time_step_when_episode_end_file.write(str(t) + ' ')
        del self.time_steps_when_episode_end[:]

        with open(self.reward_every_time_step_file_path, 'a') as reward_every_time_step_file:
            for r in self.reward_every_time_step:
                reward_every_time_step_file.write(str(r) + ' ')
        del self.reward_every_time_step[:]

        with open(self.q_target_file_path, 'a') as q_target_file:
            for q in self.q_target:
                q_target_file.write(str(q) + ' ')
        del self.q_target[:]


    def _get_loss_score_timestep_reward_qtarget_from_file(self):
        with open(self.lost_hist_file_path, 'r') as lost_hist_file:
            lost_hist_list_str = lost_hist_file.readline().split(" ")
            lost_hist_list_str = lost_hist_list_str[0:-1]
            loss = list(map(eval, lost_hist_list_str))

        with open(self.score_every_episode_file_path, 'a') as score_every_episode_file:
            scores_str = score_every_episode_file.readline().split(" ")
            scores_str = scores_str[0:-1]
            scores = list(map(eval, scores_str))

        with open(self.time_steps_when_episode_end_file_path, 'a') as time_step_when_episode_end_file:
            time_step_when_episode_end_str = time_step_when_episode_end_file.readline().split(" ")
            time_step_when_episode_end_str = time_step_when_episode_end_str[0:-1]
            time_step_when_episode_end = list(map(eval, time_step_when_episode_end_str))

        with open(self.reward_every_time_step_file_path, 'a') as reward_every_time_step_file:
            reward_every_time_step_str = reward_every_time_step_file.readline().split(" ")
            reward_every_time_step_str = reward_every_time_step_str[0:-1]
            reward_every_time_step = list(map(eval, reward_every_time_step_str))

        with open(self.q_target_file_path, 'a') as q_target_file:
            q_target_str = q_target_file.readline().split(" ")
            q_target_str = q_target_str[0:-1]
            q_target = list(map(eval, q_target_str))

        return loss, scores, time_step_when_episode_end, reward_every_time_step, q_target
