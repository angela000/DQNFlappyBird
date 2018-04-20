#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import pdb

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
FRAME_PER_ACTION = 1  # number of frames per action
BATCH = 32  # size of minibatch

OBSERVE = 100000.  # 100000 timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
GAMMA = 0.99  # decay rate of past observations
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
REPLACE_TARGET_ITER = 500  # number of steps when target net parameters update

SAVER_ITER = 10000  # number of steps when save checkpoint.
COUNTERS_SIZE = 10  # the number of episodes to average for evaluation. 10
AVERAGE_SIZE = 500  # the length of average_score to print a png. 500

# Evaluation: store the average scores of ten last episodes.
average_score = []


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

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

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
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    observation = np.stack((x_t, x_t, x_t, x_t), axis=2)  # observation 80x80x4

    sess.run(tf.global_variables_initializer())

    # saving and loading networks, step determines when to save a png file.
    saver, step = store_parameters()

    # tensorboard
    writer = tf.summary.FileWriter("./logs_bird/", sess.graph)

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
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        # store the score to counter when crash
        # (step+t) > 200000, so that the 0 pts at the beginning could be filtered.
        if terminal and (step + t) > 300000:
            counter_add(counter, game_state.score, t + step)

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
            # check to replace target parameters
            if t % REPLACE_TARGET_ITER == 0:
                sess.run(target_replace_op)
                print('\ntarget_params_replaced\n')

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
                    # e.g. readout_j1_batch[i] -- array([12.69..., 3.03...], dtype=float32)
                    # e.g. np.max(readout_j1_batch[i]) -- 12.69...

            # perform gradient step to minimize cost.
            train_step.run(feed_dict={q_target: y_batch, a: a_batch, eval_net_input: s_j_batch})
        '''end'''

        # update the old values
        observation = observation_
        t += 1

        # save progress every 10000 iterations
        if t % SAVER_ITER == 0:
            saver.save(sess, 'saved_networks/dqn_target_net/' + GAME + '-dqn', global_step=t)

        # print info
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index,
              "/ REWARD", r_t, "/ Q_MAX %e" % np.max(action_q_value))

        # write info to files
        # if t % 10000 <= 100:
        #     a_file.write(",".join([str(x) for x in action_q_value]) + '\n')
        #     h_file.write(",".join([str(x) for x in h_fc1_eval.eval(feed_dict={eval_net_input: [observation]})[0]]) + '\n')
        #     cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)


def store_parameters():
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks/dqn_target_net/")
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
        if len(average_score) >= AVERAGE_SIZE:
            plt.figure()
            plt.plot(average_score)
            plt.ylabel('score')
            plt.savefig("logs_" + GAME + "/" + str(steps) + "_average_score.png")
            average_score.clear()
        counters.clear()


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
