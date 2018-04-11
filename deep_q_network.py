#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import pdb

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
FRAME_PER_ACTION = 1  # number of frames per action
BATCH = 32  # size of minibatch

OBSERVE = 10000.  # timesteps to observe before training
EXPLORE = 3000000.  # frames over which to anneal epsilon
GAMMA = 0.99  # decay rate of past observations
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.1  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)  # [None, 20, 20, 32]
    h_pool1 = max_pool_2x2(h_conv1)  # [None, 10, 10, 32]

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)  # [None, 5, 5, 64]
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)  # [None, 5, 5, 64]
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])  # [None, 1600]
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)  # [None, 512]
    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2  # [None, 2]

    return s, readout, h_fc1


def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])

    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)  # [None]
    # readout_action -- reward of selected action by a.
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    # train_step.run(feed_dict={y: y_batch, a: a_batch, s: s_j_batch})

    # # tf.reduce_sum(...;reduction_indices: The old (deprecated) name for axis;...)
    # x = tf.constant([[1, 1, 1], [1, 1, 1]])
    # tf.reduce_sum(x)  # 6
    # tf.reduce_sum(x, 0)  # [2, 2, 2]
    # tf.reduce_sum(x, 1)  # [3, 3]
    # tf.reduce_sum(x, 1, keepdims=True)  # [[3], [3]]
    # tf.reduce_sum(x, [0, 1])  # 6

    # # tf.reduce_mean()
    # x = tf.constant([[1., 1.], [2., 2.]])
    # tf.reduce_mean(x)  # 1.5
    # tf.reduce_mean(x, 0)  # [1.5, 1.5]
    # tf.reduce_mean(x, 1)  # [1.,  2.]

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()
    # deque is a double-ended queue, or deque, supports adding and removing elements from either end.
    # More about deque, refer to https://www.jianshu.com/p/6928e420edb0

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    # do_nothing[0] == 1: do nothing
    # do_nothing[1] == 1: flap the bird
    do_nothing[0] = 1
    # game_state.frame_step return: {image_data, reward, terminal}
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    # # cv2.cvtColor()
    # Converts an image from one color space to another.
    # The conversion from a RGB image to gray is done with:
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    # The custom threshold is 1, which is greater than 1 is white and black is less than 1.
    # refer to: http://blog.csdn.net/u011321546/article/details/79593195
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)  # s_t 80x80x4

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")

    # start training
    epsilon = INITIAL_EPSILON  # 0.0001
    t = 0
    while "flappy bird" != "angry bird":  # A no-stop circulation.
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s: [s_t]})[0]
        # e.g. readout.eval(feed_dict={s: [s_t]}) -- array([[11.83..., 11.62...]], dtype=float32)
        # e.g. readout_t -- array([11.83..., 11.62...], dtype=float32)
        # readout_t means two actions's assumed reward.
        a_t = np.zeros([ACTIONS])
        # e.g. a_t -- array([0., 0.], dtype=float32)
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            # epsilon-greedy to balance exploration and exploitation.
            if random.random() <= epsilon:
                # print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # scale down epsilon
        # OBSERVE -- 100000
        # EXPLORE -- 2000000
        # INITIAL_EPSILON = FINAL_EPSILON = 0.0001
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        # preprocess the image.
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)  # (80x80x4)

        # store the last 50000(REPLAY_MEMORY) transitions in deque D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > OBSERVE:
            # sample a minibatch(32) to train on
            minibatch = random.sample(D, BATCH)

            # get the batch variables
            # (d[0], d[1], d[2], d[3], d[4])
            # (s_t, a_t, r_t, s_t1, terminal)
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []  # max_length = 32
            readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})  # (32, 2)
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
            train_step.run(feed_dict={y: y_batch, a: a_batch, s: s_j_batch})

        # update the old values
        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step=t)

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION", action_index,
              "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t))

        # write info to files
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s: [s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)


def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
