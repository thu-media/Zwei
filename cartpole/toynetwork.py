import math
import numpy as np
import tensorflow as tf
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import tflearn

FEATURE_NUM = 128
EPS = 1e-4

class Network():
    def CreateNetwork(self, inputs, acts):
        with tf.variable_scope('actor'):
            net = tflearn.fully_connected(inputs, 32, activation='relu')
            net = tflearn.fully_connected(net, 16, activation='relu')
            net = tflearn.fully_connected(net, 8, activation='relu')
            pi = tflearn.fully_connected(net, self.a_dim, activation='softmax')
            net_act = tflearn.fully_connected(acts, 8, activation='relu')
            net_val = tflearn.merge([net, net_act], 'concat')
            value = tflearn.fully_connected(net_val, 1, activation='sigmoid')
            return pi ,value
            
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
        self.pi, self.val = self.CreateNetwork(inputs=self.inputs, acts=self.acts)
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
            - tf.reduce_mean(tf.log(tf.reduce_sum(tf.multiply(self.real_out, self.acts), reduction_indices=1, keepdims=True)) * (
            self.outputs - tf.stop_gradient(self.val))) + 0.1 * tf.reduce_mean(tf.multiply(self.real_out, tf.log(self.real_out)))
        
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
