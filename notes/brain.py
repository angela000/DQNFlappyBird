"""
This part of code is the Q learning brain, which is a brain of the agent.
"""

import numpy as np
import tensorflow as tf

REPLY_START_SIZE = 1000


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.flag = True  # output signal
        self.summary_flag = output_graph  # tf.summary flag

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # start a session
        self.sess = tf.Session()

        if self.summary_flag:
            self.writer = tf.summary.FileWriter("./logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        # store a newest set of q_eval
        # self.eval_dqn = None

    def _build_net(self):
        # ------------------ all inputs ------------------------
        # input for target net
        self.eval_net_input = tf.placeholder(tf.float32, [None, self.n_features], name='eval_net_input')
        # input for eval net
        self.target_net_input = tf.placeholder(tf.float32, [None, self.n_features], name='target_net_input')
        # q_target for loss
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')

        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.eval_net_input, 10, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='e1')
            self.q_eval_net_out = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                                  bias_initializer=b_initializer, name='q')

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.target_net_input, 10, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name='t1')
            self.q_target_net_out = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer, name='t2')

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_net_out, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s[0], [a, r], s_[0]))
        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, step):
        # at the very beginning, only take actions randomly
        if step < REPLY_START_SIZE:
            action = np.random.randint(0, self.n_actions)
        else:
            # if self.flag:
            #     print("Start epsilon-greedy policy")
            #     self.flag = False
            if np.random.uniform() < self.epsilon:
                # forward feed the observation and get q value for every actions
                actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
                action = np.argmax(actions_value)
            else:
                action = np.random.randint(0, self.n_actions)

        return action

    def choose_action_eval(self, observation):
        actions_value = self.sess.run(self.q_eval_net_out, feed_dict={self.eval_net_input: observation})
        action = np.argmax(actions_value)

        return actions_value, action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        '''start from here'''
        # input is all next observation
        q_target_select_a, q_target_out = \
            self.sess.run([self.q_eval_net_out, self.q_target_net_out],
                          feed_dict={self.eval_net_input: batch_memory[:, -self.n_features:],
                                     self.target_net_input: batch_memory[:, -self.n_features:]})
        # real q_eval, input is the current observation
        q_eval = self.sess.run(self.q_eval_net_out, {self.eval_net_input: batch_memory[:, :self.n_features]})
        tf.summary.histogram("q_eval", q_eval)

        # self.eval_dqn = q_eval

        q_target = q_eval.copy()

        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        '''Double DQN'''
        # max_act4next = np.argmax(q_target_select_a, axis=1)
        # selected_q_next = q_target_out[batch_index, max_act4next]
        '''DQN'''
        selected_q_next = np.max(q_target_out, axis=1)

        # real q_target
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        tf.summary.histogram("q_target", q_target)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        '''end here'''

        self.cost_his.append(self.cost)
        tf.summary.scalar("cost", self.cost)

        if self.summary_flag:
            # merge_all() must follow all tf.summary
            self.merge_op = tf.summary.merge_all()
            self.summary_flag = False

        merge_all = self.sess.run(self.merge_op, feed_dict={self.eval_net_input: batch_memory[:, :self.n_features],
                                                            self.q_target: q_target})
        self.writer.add_summary(merge_all, self.learn_step_counter)
        # epsilon-decay
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
