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
N_FEATURES = 80 * 80 * 4  # number of features
FRAME_PER_ACTION = 1  # number of frames per action
BATCH = 32  # size of minibatch
'''OBSERVE must > REPLY_MEMORY to full the sumTree'''
OBSERVE = 2000.  # 100000 timesteps to observe before training
EXPLORE = 3000000.  # 3000000 frames over which to anneal epsilon
GAMMA = 0.99  # 0.99 decay rate of past observations
FINAL_EPSILON = 0.0001  # 0.0001 final value of epsilon
INITIAL_EPSILON = 0.0001  # 0.0001 starting value of epsilon
REPLAY_MEMORY = 1000  # 50000 number of previous transitions to remember
REPLACE_TARGET_ITER = 500  # 500 number of steps when target net parameters update

SAVER_ITER = 10000  # number of steps when save checkpoint.
COUNTERS_SIZE = 2  # the number of episodes to average for evaluation. 10
AVERAGE_SIZE = 400  # the length of average_score to print a png. 500

# Evaluation: store the average scores of ten last episodes.
average_score = []


class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):  # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        # tt = self.tree.tree
        # dd = self.tree.data
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty(
            (n, 1))
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p  # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            # aa = prob
            # bb = min_prob
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


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
    memory = Memory(capacity=REPLAY_MEMORY)

    # Evaluation: store the last COUNTERS_SIZE episodes' scores
    counter = []

    # define the cost function
    # a = tf.placeholder("float", [None, ACTIONS])
    # q_target = tf.placeholder("float", [None])
    q_target = tf.placeholder("float", [None, ACTIONS])  # for calculating loss
    ISWeights_holder = tf.placeholder("float", [None, 1])

    # define the cost
    abs_errors = tf.reduce_sum(tf.abs(q_target - readout_eval), axis=1)  # for updating Sumtree
    cost = tf.reduce_mean(ISWeights_holder * tf.squared_difference(q_target, readout_eval))
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
    writer = tf.summary.FileWriter("./logs_bird/prioritized_reply_dqn/", sess.graph)

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

        # store the score to counter when crash
        # (step+t) > 250000, so that the 0 pts at the beginning could be filtered.
        if terminal and (step + t) > 250000:
            counter_add(counter, score_current, t + step)

        # preprocess the image.
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        observation_ = np.append(x_t1, observation[:, :, :3], axis=2)  # (80x80x4)

        # store the last 50000(REPLAY_MEMORY) transitions in deque D
        transition = np.hstack((observation.flatten(), a_t, r_t, observation_.flatten()))
        memory.store(transition)  # have high priority for newly arrived transition

        # only train if done observing
        if t > OBSERVE:
            # check to replace target parameters
            if t % REPLACE_TARGET_ITER == 0:
                sess.run(target_replace_op)
                print('\ntarget_params_replaced\n')

            tree_idx, batch_memory, ISWeights_value = memory.sample(BATCH)

            q_next, q_eval = sess.run([readout_target, readout_eval],
                                      feed_dict={target_net_input: np.reshape(batch_memory[:, -N_FEATURES:],
                                                                              (BATCH, 80, 80, 4)),
                                                 eval_net_input: np.reshape(batch_memory[:, :N_FEATURES],
                                                                            (BATCH, 80, 80, 4))})
            q_target_value = q_eval.copy()

            batch_index = np.arange(BATCH, dtype=np.int32)
            eval_act_index = batch_memory[:, N_FEATURES].astype(int)
            reward = batch_memory[:, N_FEATURES + 1]

            q_target_value[batch_index, eval_act_index] = reward + GAMMA * np.max(q_next, axis=1)

            _, abs_errors_, cost_ = sess.run([train_step, abs_errors, cost],
                                             feed_dict={eval_net_input: np.reshape(batch_memory[:, :N_FEATURES],
                                                                                   (BATCH, 80, 80, 4)),
                                                        q_target: q_target_value,
                                                        ISWeights_holder: ISWeights_value})
            memory.batch_update(tree_idx, abs_errors_)  # update priority
        '''end'''

        # update the old values
        observation = observation_
        t += 1

        # save progress every 10000 iterations
        if t % SAVER_ITER == 0:
            saver.save(sess, 'saved_networks/prioritized_reply_dqn/' + GAME + '-dqn', global_step=t)

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
    checkpoint = tf.train.get_checkpoint_state("saved_networks/prioritized_reply_dqn/")
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
            a = steps // 1000000
            max_size = AVERAGE_SIZE // (2 ** a)
        else:
            max_size = AVERAGE_SIZE
        if len(average_score) >= max_size:
            fo = open("logs_" + GAME + "/prioritized_reply_dqn/" + str(steps) + "_average_score.txt", "w")
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
