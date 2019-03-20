#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import pickle
import sys
sys.path.append("game/")
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Hyper Parameters:
FRAME_PER_ACTION = 1                                # number of frames per action.
GAMMA = 0.99                                        # decay rate of past observations.
SAVE_PATH = "./saved_parameters/actor_critic/"      # store network parameters and other parameters for pause.
STOP_STEP = 1500000.                                # the only way to exit training. 1,500,000 time steps.
DIR_NAME = '/actor_critic/'                         # name of the log directory (be different with other networks).


# Brain重要接口:
# getAction():      根据self.currentState选择action
# setPerception():  得到新observation之后进行记忆学习
class BrainDQN:
    def __init__(self, actionNum, gameName):
        self.actionNum = actionNum
        self.gameName = gameName
        # init other parameters
        self.onlineTimeStep = 0
        # saved parameters every SAVER_ITER
        self.gameTimes = 0
        self.timeStep = 0
        self.saved_parameters_file_path = SAVE_PATH + self.gameName + '-saved-parameters.txt'
        # logs, append to file every SAVER_ITER
        self.logs_path = "./logs_" + self.gameName + DIR_NAME   # "logs_bird/actor_critic/"
        self.lost_hist_actor = []
        self.lost_hist_actor_file_path = self.logs_path + 'lost_hist_actor.txt'
        self.lost_hist_critic = []
        self.lost_hist_critic_file_path = self.logs_path + 'lost_hist_critic.txt'
        self.scores = []
        self.scores_file_path = self.logs_path + 'scores.txt'
        self.total_rewards_this_episode = 0
        self.rewards = []
        self.rewards_file_path = self.logs_path + 'reward.txt'
        # init Q network
        self.createQNetwork()


    def createQNetwork(self):
        # build actor network
        with tf.variable_scope('Actor'):
            # input layer
            self.state_input_a = tf.placeholder("float", [1, 80, 80, 4])
            # conv layer 1
            W_conv1_a = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
            b_conv1_a = tf.Variable(tf.constant(0.01, shape=[32]))
            h_conv1_a = tf.nn.conv2d(self.state_input_a, W_conv1_a, strides=[1, 4, 4, 1], padding="SAME")
            h_relu1_a = tf.nn.relu(h_conv1_a + b_conv1_a)
            # [1, 20, 20, 32]
            h_pool1_a = tf.nn.max_pool(h_relu1_a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # [1, 10, 10, 32]
            # conv layer 2
            W_conv2_a = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
            b_conv2_a = tf.Variable(tf.constant(0.01, shape=[64]))
            h_conv2_a = tf.nn.conv2d(h_pool1_a, W_conv2_a, strides=[1, 2, 2, 1], padding="SAME")
            h_relu2_a = tf.nn.relu(h_conv2_a + b_conv2_a)
            # [1, 5, 5, 64]
            # conv layer 3
            W_conv3_a = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
            b_conv3_a = tf.Variable(tf.constant(0.01, shape=[64]))
            h_conv3_a = tf.nn.conv2d(h_relu2_a, W_conv3_a, strides=[1, 1, 1, 1], padding="SAME")
            h_relu3_a = tf.nn.relu(h_conv3_a + b_conv3_a)
            # [1, 5, 5, 64]
            # reshape layer
            h_conv3_flat_a = tf.reshape(h_relu3_a, [-1, 1600])
            # [1, 1600]
            # full layer
            W_fc1_a = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.01))
            b_fc1_a = tf.Variable(tf.constant(0.01, shape=[512]))
            h_fc1_a = tf.nn.relu(tf.matmul(h_conv3_flat_a, W_fc1_a) + b_fc1_a)
            # [1, 512]
            # reader layer 1
            W_fc2_a = tf.Variable(tf.truncated_normal([512, self.actionNum], stddev=0.01))
            b_fc2_a = tf.Variable(tf.constant(0.01, shape=[self.actionNum]))

            self.QValue_a = tf.matmul(h_fc1_a, W_fc2_a) + b_fc2_a
            # [1, 2]
            self.action_prob_a = tf.nn.softmax(self.QValue_a)

            # build actor train network
            self.td_error_a = tf.placeholder(tf.float32, None, "td_error")      # TD_error
            self.action_a = tf.placeholder(tf.int32, None, "act")               # action_a
            log_prob_a = tf.log(self.action_prob_a[0, self.action_a])
            self.loss_a = tf.reduce_mean(log_prob_a * self.td_error_a)  # reduce_mean是否有意义？？？
            self.train_step_a = tf.train.AdamOptimizer(1e-6).minimize(self.loss_a)   # loss

        # build critic network
        with tf.variable_scope('critic'):
            # input layer
            self.state_input_c = tf.placeholder("float", [1, 80, 80, 4])
            # conv layer 1
            W_conv1_c = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
            b_conv1_c = tf.Variable(tf.constant(0.01, shape=[32]))
            h_conv1_c = tf.nn.conv2d(self.state_input_c, W_conv1_c, strides=[1, 4, 4, 1], padding="SAME")
            h_relu1_c = tf.nn.relu(h_conv1_c + b_conv1_c)
            # [1, 20, 20, 32]
            h_pool1_c = tf.nn.max_pool(h_relu1_c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            # [1, 10, 10, 32]
            # conv layer 2
            W_conv2_c = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
            b_conv2_c = tf.Variable(tf.constant(0.01, shape=[64]))
            h_conv2_c = tf.nn.conv2d(h_pool1_c, W_conv2_c, strides=[1, 2, 2, 1], padding="SAME")
            h_relu2_c = tf.nn.relu(h_conv2_c + b_conv2_c)
            # [1, 5, 5, 64]
            # conv layer 3
            W_conv3_c = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
            b_conv3_c = tf.Variable(tf.constant(0.01, shape=[64]))
            h_conv3_c = tf.nn.conv2d(h_relu2_c, W_conv3_c, strides=[1, 1, 1, 1], padding="SAME")
            h_relu3_c = tf.nn.relu(h_conv3_c + b_conv3_c)
            # [1, 5, 5, 64]
            # reshape layer
            h_conv3_flat_c = tf.reshape(h_relu3_c, [-1, 1600])
            # [1, 1600]
            # full layer
            W_fc1_c = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.01))
            b_fc1_c = tf.Variable(tf.constant(0.01, shape=[512]))
            h_fc1_c = tf.nn.relu(tf.matmul(h_conv3_flat_c, W_fc1_c) + b_fc1_c)
            # [1, 512]
            # reader layer 1
            W_fc2_c = tf.Variable(tf.truncated_normal([512, 1], stddev=0.01))
            b_fc2_c = tf.Variable(tf.constant(0.01, shape=[1]))

            self.state_value_c = tf.matmul(h_fc1_c, W_fc2_c) + b_fc2_c
            # [1, 1]

            # build critic train network
            self.reward_c = tf.placeholder(tf.float32, None)
            self.next_state_value_c = tf.placeholder(tf.float32, [1, 1])
            self.td_error_c = self.reward_c + GAMMA * self.next_state_value_c - self.state_value_c
            # 没有tf.reduce_sum，reduce_mean，也就是说这是一种online-learning.没有batch
            self.loss_c = tf.square(self.td_error_c)
            self.train_step_c = tf.train.AdamOptimizer(1e-6).minimize(self.loss_c)

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


    def trainQNetwork(self, action, reward, next_state):
        td_error, _, self.lost_c = self.sess.run(
            [self.td_error_c, self.train_step_c, self.loss_c],
            feed_dict={
                self.reward_c: reward,
                self.next_state_value_c: next_state,
                self.state_value_c: self.currentState
            }
        )
        # train on episode
        _, self.lost_a = self.sess.run(
            [self.train_step_a, self.loss_a],
            feed_dict={
                self.td_error_a: td_error,
                self.action_a: action,
                self.state_input_a: self.currentState
            }
        )
        self.lost_hist_actor.append(self.lost_a)
        self.lost_hist_critic.append(self.lost_c)

        # save network and other data every 100,000 iteration
        if self.timeStep % 100000 == 0:
            self.saver.save(self.sess, SAVE_PATH + self.gameName, global_step=self.timeStep)
            saved_parameters_file = open(self.saved_parameters_file_path, 'wb')
            pickle.dump(self.gameTimes, saved_parameters_file)
            pickle.dump(self.timeStep, saved_parameters_file)
            pickle.dump(self.epsilon, saved_parameters_file)
            saved_parameters_file.close()
            self.save_lsr_to_file()
        if self.timeStep == STOP_STEP:
            self.end_the_game()


    # observ != state. game环境可以给observ，但是state需要自己构造（最近的4个observ）
    def setPerception(self, nextObserv, action, reward, terminal, curScore):
        self.total_rewards_this_episode += reward
        # 把nextObserv放到最下面，把最上面的抛弃
        newState = np.append(self.currentState[:, :, 1:], nextObserv, axis = 2)
        print("TIMESTEP", self.timeStep, "/ ACTION", action[1], "/ EPSILON", self.epsilon, "/ REWARD", reward)
        self.trainQNetwork(reward, nextObserv)
        if terminal:
            self.gameTimes += 1
            print("GAME_TIMES:" + str(self.gameTimes))
            self.scores.append(curScore)
            self.rewards.append(self.total_rewards_this_episode)
            self.total_rewards_this_episode = 0
        self.currentState = newState
        self.timeStep += 1
        self.onlineTimeStep += 1


    # 这个getAction可以多观察观察。细节上还没理解。很容易出错。
    def getAction(self):
        act_prob = self.action_prob_a.eval(feed_dict = {self.state_input_a: self.currentState})
        action = np.random.choice(range(act_prob.shape[1]), p=act_prob.ravel())
        return action


    def setInitState(self, observ):
        self.currentState = np.stack((observ, observ, observ, observ), axis = 2)


    # Called when the game ends.
    def end_the_game(self):
        self.save_lsr_to_file()
        self.get_lsr_from_file()
        plt.figure()
        plt.plot(self.lost_hist_actor)
        plt.ylabel('actor_lost')
        plt.savefig(self.logs_path + "lost_hist_actor_total.png")

        plt.figure()
        plt.plot(self.lost_hist_critic)
        plt.ylabel('critic_lost')
        plt.savefig(self.logs_path + "lost_hist_critic_total.png")

        plt.figure()
        plt.plot(self.scores)
        plt.ylabel('score')
        plt.savefig(self.logs_path + "scores_total.png")

        plt.figure()
        plt.plot(self.rewards)
        plt.ylabel('rewards')
        plt.savefig(self.logs_path + "rewards_total.png")


    # save lost/score/reward to file
    def save_lsr_to_file(self):
        list_hist_actor_file = open(self.lost_hist_actor_file_path, 'a')
        for la in self.lost_hist_actor:
            list_hist_actor_file.write(str(la) + ' ')
        list_hist_actor_file.close()
        del self.lost_hist_actor[:]

        list_hist_critic_file = open(self.lost_hist_critic_file_path, 'a')
        for lc in self.lost_hist_critic:
            list_hist_critic_file.write(str(lc) + ' ')
        list_hist_critic_file.close()
        del self.lost_hist_critic[:]

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


    def get_lsr_from_file(self):
        scores_file = open(self.scores_file_path, 'r')
        scores_str = scores_file.readline().split(" ")
        scores_str = scores_str[0:-1]
        self.scores = list(map(eval, scores_str))
        scores_file.close()

        lost_hist_actor_file = open(self.lost_hist_actor_file_path, 'r')
        lost_hist_actor_list_str = lost_hist_actor_file.readline().split(" ")
        lost_hist_actor_list_str = lost_hist_actor_list_str[0:-1]
        self.lost_hist_actor = list(map(eval, lost_hist_actor_list_str))
        lost_hist_actor_file.close()

        lost_hist_critic_file = open(self.lost_hist_critic_file_path, 'r')
        lost_hist_critic_list_str = lost_hist_critic_file.readline().split(" ")
        lost_hist_critic_list_str = lost_hist_critic_list_str[0:-1]
        self.lost_hist_critic = list(map(eval, lost_hist_critic_list_str))
        lost_hist_critic_file.close()

        rewards_file = open(self.rewards_file_path, 'r')
        rewards_str = rewards_file.readline().split(" ")
        rewards_str = rewards_str[0:-1]
        self.rewards = list(map(eval, rewards_str))
        rewards_file.close()