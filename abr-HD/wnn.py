import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
ACTION_EPS = 1e-4
POOL_MAX_SIZE = 100000
# PPO2
EPS = 0.2

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            split_0 = tflearn.fully_connected(
                inputs[:, 0:1, -1], FEATURE_NUM, activation='relu')
            split_1 = tflearn.fully_connected(
                inputs[:, 1:2, -1], FEATURE_NUM, activation='relu')
            split_2 = tflearn.conv_1d(
                inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
            split_3 = tflearn.conv_1d(
                inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(
                inputs[:, 4:5, :self.a_dim], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(
                inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            w_net = tflearn.fully_connected(
                merge_net, FEATURE_NUM, activation='relu')
            w = tflearn.fully_connected(w_net, self.a_dim, activation='tanh')
            return w
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self._entropy = 1.
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.R = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.pool = []
        self.entropy_weight = tf.placeholder(tf.float32)
        self.w = self.CreateNetwork(inputs = self.inputs)
        # self.real_out = tf.clip_by_value(tf.nn.softmax(self.w * self.entropy_weight), ACTION_EPS, 1. - ACTION_EPS)
        self.loss = tflearn.mean_square( \
            tf.reduce_sum(self.w * self.acts, reduction_indices=1, keepdims=True),
            self.R)
        # Get all network parameters
        self.network_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(
                self.network_params[idx].assign(param))

        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)

    def predict(self, input, argmax=False):
        q_prob = self.sess.run(self.w, feed_dict={
            self.inputs: input
        })
        prob = q_prob[0]
        if argmax:
            return np.argmax(prob)
        if np.random.uniform() > self._entropy:
            return np.argmax(prob)
        else:
            return np.random.randint(self.a_dim)

    def set_entropy_decay(self, decay = 0.9):
        self._entropy *= decay
        self._entropy = np.clip(self._entropy, 1e-10, 1.)

    def get_entropy(self):
        return self._entropy

    def train(self, s_batch, a_batch, p_batch, v_batch):
        for (s, a, w) in zip(s_batch, a_batch, v_batch):
            self.pool.append([s, a, w])
            if len(self.pool) > POOL_MAX_SIZE:
                self.pool.pop(0)
        
        pool_num = len(self.pool)
        if pool_num >= 1000:
            s_batch, a_batch, w_batch = [], [], []
            for _iter in range(10):
                for _sample in range(32):
                    _idx = np.random.randint(pool_num)
                    _s, _a , _w = self.pool[_idx]
                    s_batch.append(_s)
                    a_batch.append(_a)
                    w_batch.append(_w)

                self.sess.run(self.optimize, feed_dict={
                    self.inputs: s_batch,
                    self.acts: a_batch,
                    self.R: w_batch, 
                    self.entropy_weight: self.get_entropy()
                })
