import socket
import time
import thread
import os
import struct
import sys
import tensorflow as tf
import tflearn
from tflearn.layers.conv import global_max_pool
import numpy as np
import warnings
import json
from ../../a3c_off import a3c

#from socket import *

socket_buffer_len = 1500
ip = '0.0.0.0'
PORT = 23583
ALPHA = 0.3
if len(sys.argv) > 1:
    ip = sys.argv[1]
LOGFILENAME = ip
if len(sys.argv) > 2:
    ALPHA = float(sys.argv[2])
if len(sys.argv) > 3:
    LOGFILENAME = sys.argv[3]
if len(sys.argv) > 4:
    PORT = int(sys.argv[4])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
INITAL = 0.125 * 2
FEATURE_NUM = 64
S_INFO = 3
S_LEN = 8
INTERVAL = 1000
LOG = 'hybrid-final'


def build_error_net(x, sess=None):
    inputs = tflearn.input_data(placeholder=x)
    dense_net = tflearn.fully_connected(inputs, FEATURE_NUM, activation='relu')
    dense_net = tflearn.fully_connected(dense_net, 1, activation='linear')
    return dense_net


def build_hybrid_net(x, sess=None):
    inputs = tflearn.input_data(placeholder=x)
    with tf.name_scope('1d-cnn'):
        network_array = []
        for p in xrange(S_INFO - 1):
            branch = tflearn.conv_1d(
                inputs[:, :, p:p+1], FEATURE_NUM, 3, activation='relu')
            branch = tflearn.flatten(branch)
            network_array.append(branch)
        out_cnn = tflearn.merge(network_array, 'concat')
    with tf.name_scope('gru'):
        net = tflearn.gru(x, FEATURE_NUM, return_seq=True)
        out_gru = tflearn.gru(x, FEATURE_NUM)

    header = tflearn.merge([out_cnn, out_gru], 'concat')
    dense_net = tflearn.fully_connected(inputs, FEATURE_NUM, activation='relu')
    out = tflearn.fully_connected(header, 1, activation='linear')
    return out, header


def tickcount():
    return int(time.time() * 1000)


def generate_packet(nextbitrate, nextrange):
    #print (nextbitrate), (nextrange)
    nextbitrate *= 1024 * 1024
    nextrange *= 1024 * 1024
    _json = {}
    _json['baseline'] = int(nextbitrate)
    _json['variable'] = int(nextrange)
    return json.dumps(_json)


def analysis_packet(msg, f=None):
    # print len(msg)
    _realen = len(msg) - 8
    timestamp, _bytes = struct.unpack(
        'Q' + str(_realen) + 's', msg)
    if f is not None:
        f.write(_bytes)
    return timestamp, 0, _bytes, _realen


def DelayController(DelayArray, k=S_LEN, alpha=ALPHA):
    return  0.0 - (alpha - 1) / (alpha + k)* np.sum(DelayArray[-k:])
    #return -np.mean(DelayArray)
    # return -np.sum(DelayArray)


def main():
    precvtick = -1
    bytesbuf = 0
    send_len = 0
    delay_grad = []
    saver = tf.train.Saver()
    state = np.zeros((S_INFO, S_LEN))
    # delay_buffer = np.zeros(S_LEN)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        x = tf.placeholder(shape=(None, S_LEN, S_INFO), dtype=tf.float32)
        y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        # merge = tf.placeholder(shape=(None, FEATURE_NUM * 2), dtype=tf.float32)

        hybrid_net, hybrid_header = build_hybrid_net(x)
        hybrid_net_loss = tf.abs(hybrid_net - y_) / y_
        hybrid_train_op = tf.train.AdamOptimizer(
            learning_rate=0.0000625).minimize(hybrid_net_loss)
        hybrid_accuracy = tf.reduce_mean(hybrid_net_loss)
        hybrid_err = tf.abs(hybrid_net - y_)

        err_net = build_error_net(hybrid_header)
        err_net_loss = tflearn.objectives.mean_square(err_net, hybrid_err)
        err_train_op = tf.train.AdamOptimizer(
            learning_rate=0.00001).minimize(err_net_loss)
        err_accuracy = tf.reduce_mean(err_net_loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, LOG + "_model/nn_model_ep_" + str(190) + ".ckpt")
        #print 'model restored'
        address = ('0.0.0.0', 0)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        # s.setblocking(0)
        s.bind(address)
        #print 'start request', address
        _send_address = (ip, PORT)
        # for i in range(S_LEN):
        s.sendto(generate_packet(INITAL, 0.0), _send_address)
        #_time = 0
        _writer = open(LOGFILENAME + '.csv', 'w')
        _last_delay_grad_mean = -1
        _last_delay_checked = False
        _index = 0
        state = np.zeros((S_INFO, S_LEN))
        last_bitrate = INITAL * 1024 * 1024
        last_variable = 0
        last_estimated = -1
        _tick = 0
        _f = open('test.h264', 'w')
        while True:
            try:
                data, addr = s.recvfrom(socket_buffer_len)
            except socket.timeout:
                s.sendto(generate_packet(last_bitrate / (1024.0 * 1024.0),
                                         last_variable / (1024.0 * 1024.0)), _send_address)
                continue
            _ts, _sendlen, _bytes, _len = analysis_packet(data, _f)
            _now = tickcount()
            if precvtick < 0:
                precvtick = _now
            _delay = _now - _ts
            if _now - precvtick >= INTERVAL:
                #_time += 1
                _tick += 1
                if _tick == 60:
                    break
                _delay_grad_mean = np.mean(delay_grad)
                if _last_delay_checked == False:
                    _last_delay_grad_mean = _delay_grad_mean
                    _last_delay_checked = True
                _delay_gradient = _delay_grad_mean - _last_delay_grad_mean
                _last_delay_grad_mean = _delay_grad_mean
                #_bytes_buf = bytesbuf
                state = np.roll(state, -1, axis=1)
                state[0, -1] = bytesbuf / (1024.0 * 1024.0)
                state[1, -1] = _delay_gradient / 1000.0
                state[2] = np.zeros(S_LEN)
                #print last_estimated, _delay_gradient / \
                #    1000.0, abs(last_estimated - _delay_gradient / 1000.0)
                if _tick < S_LEN:
                    last_estimated = DelayController(state[1], _tick)
                else:
                    last_estimated = DelayController(state[1])
                state[2, -1] = last_estimated

                _writer.write(str(bytesbuf) + ',' +
                              str(last_bitrate) + ',' + str(last_variable) + ',' + str(_delay_gradient))
                _writer.write('\n')
                #_writer.write(state)
                if _index < 1:
                    s.sendto(generate_packet(INITAL, 0.0), addr)
                    _index += 1
                else:
                    #print state
                    _state = [state.T]
                    _predict_value = sess.run(
                        hybrid_net, feed_dict={x: _state})
                    _predict_err = sess.run(
                        err_net, feed_dict={x: _state})

                    if _predict_value > 0:
                        #_packet = generate_packet(INITAL, 0.0)
                        _packet = generate_packet(
                            _predict_value[0][0], _predict_err[0][0])
                        #print _predict_value[0][0], _predict_err[0][0]
                        last_bitrate = _predict_value[0][0] * 1024.0 * 1024.0
                        last_variable = _predict_err[0][0] * 1024.0 * 1024.0
                    else:
                        _packet = generate_packet(INITAL, 0.0)
                        last_bitrate = INITAL * 1024.0 * 1024.0
                        last_variable = 0.0

                    s.sendto(_packet, addr)
                precvtick = _now
                delay_grad = []
                bytesbuf = 0
            else:
                bytesbuf += _len
                # bytesbuf /= 1000.0
                send_len = _sendlen
                delay_grad.append(_delay)
                # s.sendto(generate_packet(last_bitrate / (1024.0 * 1024.0),
                #                         last_variable / (1024.0 * 1024.0)), _send_address)
        s.close()


def load_data(filename):
    _x, _y = [], []
    f = open(filename, 'r')
    for line in f:
        _jsondata = json.loads(line)
        if _jsondata is not None:
            state = np.zeros((S_INFO, S_LEN))
            state[0] = np.array(_jsondata['b'])
            state[1] = np.array(_jsondata['d'])
            state[2, -1] = _jsondata['x']
            _x.append(state.T)
            _y.append(_jsondata['y'])
    _x = np.array(_x)
    _y = np.array(_y)
    _y = np.reshape(_y, [_y.shape[0], 1])
    return _x, _y


def test():

    state = np.zeros((S_INFO, S_LEN))
    # delay_buffer = np.zeros(S_LEN)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        x = tf.placeholder(shape=(None, S_LEN, S_INFO), dtype=tf.float32)
        y_ = tf.placeholder(shape=(None, 1), dtype=tf.float32)
        # merge = tf.placeholder(shape=(None, FEATURE_NUM * 2), dtype=tf.float32)

        hybrid_net, hybrid_header = build_hybrid_net(x)
        hybrid_net_loss = tf.abs(hybrid_net - y_) / y_
        hybrid_train_op = tf.train.AdamOptimizer(
            learning_rate=0.000625).minimize(hybrid_net_loss)
        hybrid_accuracy = tf.reduce_mean(hybrid_net_loss)
        hybrid_err = tf.abs(hybrid_net - y_)

        err_net = build_error_net(hybrid_header)
        err_net_loss = tf.sqrt(
            tflearn.objectives.mean_square(err_net, hybrid_err))
        err_train_op = tf.train.AdamOptimizer(
            learning_rate=0.0001).minimize(err_net_loss)
        err_accuracy = tf.reduce_mean(err_net_loss)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, LOG + "_model/nn_model_ep_" + str(190) + ".ckpt")
        print 'model restored'
        _x, _y = load_data('3')

        _predict_value = sess.run(
            hybrid_net, feed_dict={x: _x})
        _predict_err = sess.run(
            err_net, feed_dict={x: _x})
        print np.mean(1.0 - np.abs(_predict_value - _y) / _y)


if __name__ == '__main__':
    # test()
    main()
    # harmonic()
