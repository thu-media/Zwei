import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 64
EPS = 1e-4

class Network():
    def CreateNetwork(self, inputs):
        with tf.variable_scope('actor'):
            split_0 = tflearn.conv_1d(inputs[:, 0:1, :], FEATURE_NUM, 4, activation='relu')
            split_1 = tflearn.conv_1d(inputs[:, 1:2, :], FEATURE_NUM, 4, activation='relu')
            split_2 = tflearn.conv_1d(inputs[:, 2:3, :], FEATURE_NUM, 4, activation='relu')
            split_3 = tflearn.conv_1d(inputs[:, 3:4, :], FEATURE_NUM, 4, activation='relu')
            split_4 = tflearn.conv_1d(inputs[:, 4:5, :], FEATURE_NUM, 4, activation='relu')
            split_5 = tflearn.fully_connected(inputs[:, 5:6, -1], FEATURE_NUM, activation='relu')
            
            split_0_flat = tflearn.flatten(split_0)
            split_1_flat = tflearn.flatten(split_1)
            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge([split_0_flat, split_1_flat, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')
            net = tflearn.fully_connected(merge_net, FEATURE_NUM, activation='relu')
                
            pi = tflearn.fully_connected(net, self.a_dim, activation='softmax')
            value = tflearn.fully_connected(net, 1, activation='tanh')
            return pi, value
            
    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })
        
    def __init__(self, sess, state_dim, action_dim, learning_rate):
        self.quality = 0
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate
        self.sess = sess
        self.outputs = tf.placeholder(tf.float32, [None, 1])
        self.inputs = tf.placeholder(tf.float32, [None, self.s_dim[0], self.s_dim[1]])
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])
        self.pi, self.val = self.CreateNetwork(inputs=self.inputs)
        self.real_out = tf.clip_by_value(self.pi, EPS, 1. - EPS)

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
        
        self.loss = 0.5 * tflearn.mean_square(self.val, self.outputs) \
            + tflearn.objectives.categorical_crossentropy(self.real_out, self.acts * (self.outputs - tf.stop_gradient(self.val))) \
            - 0.05 * tflearn.objectives.categorical_crossentropy(self.real_out, self.real_out)
        
        self.optimize = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)
    
    def predict(self, input):
        action = self.sess.run(self.real_out, feed_dict={
            self.inputs: input
        })
        return action[0]
    
    def train(self, s_batch, a_batch, g_batch):
        s_batch, a_batch, g_batch = tflearn.data_utils.shuffle(s_batch, a_batch, g_batch)
        self.sess.run(self.optimize, feed_dict={
            self.inputs: s_batch,
            self.acts: a_batch,
            self.outputs: g_batch
        })
