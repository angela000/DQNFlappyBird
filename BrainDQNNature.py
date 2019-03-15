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
SAVE_PATH = "./saved_parameters/dqn_nature/"   # store network parameters and other parameters for pause.
COUNTERS_SIZE = 10                      # calculate the average score for every 10 episodes.
STOP_STEP = 1500000.                    # the only way to exit training. 1,500,000 time steps.
DIR_NAME = '/dqn_nature/'                      # name of the log directory (be different with other networks).


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



#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import game.wrapped_flappy_bird as game
import random
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

import shutil
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import pdb

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
FRAME_PER_ACTION = 1  # number of frames per action
BATCH = 32  # size of minibatch

OBSERVE = 10000.  # 100000 timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
GAMMA = 0.99  # decay rate of past observations
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
REPLACE_TARGET_ITER = 500  # number of steps when target net parameters update

SAVER_ITER = 10000  # number of steps when save checkpoint.
COUNTERS_SIZE = 2  # the number of episodes to average for evaluation. 10
AVERAGE_SIZE = 400  # the length of average_score to print a png. 500

# Evaluation: store the average scores of ten last episodes.
average_score = []

LOGS_PATH = "./logs_" + GAME + "/dqn_nature/"
SAVE_PATH = "./saved_parameters/dqn_nature/"
SAVE_BACK_PATH = "./saved_back/dqn_nature/"


def createNetwork():
    # input layer
    with tf.variable_scope("eval_net"):
        eval_net_input = tf.placeholder("float", [None, 80, 80, 4])
        # conv layer 1
        W_conv1_e = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
        b_conv1_e = tf.Variable(tf.constant(0.01, shape=[32]))

        h_conv1_e = tf.nn.conv2d(eval_net_input, W_conv1_e, strides=[1, 4, 4, 1], padding="SAME")
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
        W_fc2_e = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.01))
        b_fc2_e = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

        readout_e = tf.matmul(h_fc1_e, W_fc2_e) + b_fc2_e  # [None, 2]

    with tf.variable_scope("target_net"):
        target_net_input = tf.placeholder("float", [None, 80, 80, 4])
        # conv layer 1
        W_conv1_t = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01), trainable=False)
        b_conv1_t = tf.Variable(tf.constant(0.01, shape=[32]), trainable=False)

        h_conv1_t = tf.nn.conv2d(target_net_input, W_conv1_t, strides=[1, 4, 4, 1], padding="SAME")
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
        W_fc2_t = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.01), trainable=False)
        b_fc2_t = tf.Variable(tf.constant(0.01, shape=[ACTIONS]), trainable=False)

        readout_t = tf.matmul(h_fc1_t, W_fc2_t) + b_fc2_t  # [None, 2]

    return eval_net_input, target_net_input, readout_e, readout_t, h_fc1_e, h_fc1_t


def trainNetwork(eval_net_input, target_net_input, readout_eval, readout_target, h_fc1_eval, h_fc1_target, sess):
    # parameter transfer
    t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    with tf.variable_scope('soft_replacement'):
        target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    # store the previous observations in replay memory
    D = deque()

    # Evaluation: store the last ten episodes' scores
    counter = []

    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    q_target = tf.placeholder("float", [None])

    q_eval = tf.reduce_sum(tf.multiply(readout_eval, a), axis=1)  # [None]
    # readout_action -- reward of selected action by a.
    cost = tf.reduce_mean(tf.square(q_target - q_eval))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    # do_nothing[0] == 1: do nothing
    # do_nothing[1] == 1: flap the bird
    do_nothing[0] = 1
    # game_state.frame_step return: {image_data, reward, terminal}
    x_t, r_0, terminal, score_current = game_state.frame_step(do_nothing)

    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    observation = np.stack((x_t, x_t, x_t, x_t), axis=2)  # observation 80x80x4

    sess.run(tf.global_variables_initializer())

    # saving and loading networks, step determines when to save a png file.
    saver, step = store_parameters()

    # tensorboard
    writer = tf.summary.FileWriter(LOGS_PATH, sess.graph)

    # start training
    epsilon = INITIAL_EPSILON  # 0.0001
    t = 0
    while "ZR love EV" != "False":  # A no-stop circulation.
        # choose an action epsilon greedily
        a_t, action_q_value, action_index = epsilon_select_action(t, epsilon, observation)

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal, score_current = game_state.frame_step(a_t)

        # Training:
        if (step+t) % 1000000 == 0:
            shutil.copytree(SAVE_PATH, SAVE_BACK_PATH + str((step + t)))
        # Testing:
        # store the score to counter when crash
        # (step+t) > 200000, so that the 0 pts at the beginning could be filtered.
        # if terminal and (step + t) > 250000:
            # counter_add(counter, score_current, t+step)

        # preprocess the image.
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        # observation_ = np.stack((x_t1, x_t1, x_t1, x_t1), axis=2)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        observation_ = np.append(x_t1, observation[:, :, :3], axis=2)  # (80x80x4)

        # store the last 50000(REPLAY_MEMORY) transitions in deque D
        D.append((observation, a_t, r_t, observation_, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # aa = sess.run(e_params)
            # bb = sess.run(t_params)
            # check to replace target parameters
            if t % REPLACE_TARGET_ITER == 0:
                sess.run(target_replace_op)
                print('\ntarget_params_replaced\n')
                # aa = sess.run(e_params)
                # bb = sess.run(t_params)

            # sample a minibatch(32) to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            # (d[0], d[1], d[2], d[3], d[4])
            # (observation, a_t, r_t, observation_, terminal)
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []  # max_length = 32
            readout_j1_batch = readout_target.eval(feed_dict={target_net_input: s_j1_batch})  # (32, 2)
            for i in range(0, len(minibatch)):  # len(minibatch) -- 32
                terminal = minibatch[i][4]  # terminal -- type:bool
                # if terminal, only equals reward
                # terminal: true -- crash
                # terminal: false -- right
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

            # perform gradient step to minimize cost.
            train_step.run(feed_dict={q_target: y_batch, a: a_batch, eval_net_input: s_j_batch})
        '''end'''

        # update the old values
        observation = observation_
        t += 1

        # save progress every 10000 iterations
        if t % SAVER_ITER == 0:
            saver.save(sess, SAVE_PATH + GAME + '-dqn', global_step=(t+step))

        # print info
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index,
              "/ REWARD", r_t, "/ Q_MAX %e" % np.max(action_q_value))


def store_parameters():
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(SAVE_PATH)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0

    return saver, step


def counter_add(counters, count, steps):
    counters.append(count)
    # calculate the mean score and clear the counter.
    if len(counters) >= COUNTERS_SIZE:
        average_score.append(np.mean(counters))
        # get a scores png and clear average_score.
        if steps >= 1000000:
            a = steps//1000000
            max_size = AVERAGE_SIZE//(2**a)
        else:
            max_size = AVERAGE_SIZE
        if len(average_score) >= max_size:
            # plt.figure()
            # plt.plot(average_score)
            # plt.ylabel('score')
            # plt.savefig("logs_" + GAME + "/dqn_target_net/" + str(steps) + "_average_score.png")
            fo = open(LOGS_PATH + str(steps) + "_average_score.txt", "w")
            fo.write(str(average_score))
            fo.close()
            del average_score[:]
        del counters[:]


def epsilon_select_action(step, epsilon, observation):
    action_q_value = readout_eval.eval(feed_dict={eval_net_input: [observation]})[0]
    a_t = np.zeros([ACTIONS])
    # e.g. a_t -- array([0., 0.], dtype=float32)
    action_index = 0
    if step % FRAME_PER_ACTION == 0:
        # epsilon-greedy to balance exploration and exploitation.
        if random.random() <= epsilon:
            # print("----------Random Action----------")
            action_index = random.randrange(ACTIONS)
            a_t[action_index] = 1
        else:
            action_index = np.argmax(action_q_value)
            a_t[action_index] = 1
    else:
        a_t[0] = 1  # do nothing

    return a_t, action_q_value, action_index


if __name__ == "__main__":
    sess = tf.InteractiveSession()
    eval_net_input, target_net_input, readout_eval, readout_target, h_fc1_eval, h_fc1_target = createNetwork()
    trainNetwork(eval_net_input, target_net_input, readout_eval, readout_target, h_fc1_eval, h_fc1_target, sess)
