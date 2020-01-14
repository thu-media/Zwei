import numpy as np
import tensorflow as tf
import network as a3c
import sys
from test_env import NetworkEnv
from matplotlib import pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

S_DIM = [6, 8]
A_DIM = 7
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RAND_RANGE = 1000
# NN_MODEL = './models/nn_model_eps_len_300.ckpt'
NN_MODEL = sys.argv[1]
EP = sys.argv[2]
TEST_TRACES = './test_traces/'


def main():

    #env = gym.make("CartPole-v0")
    # env.force_mag = 100.0
    os.system('mkdir test_results')
    with tf.Session() as sess:
        actor = a3c.Network(
            sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        # sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, NN_MODEL)
        rew_list = []
        for _file in os.listdir(TEST_TRACES):
            env = NetworkEnv(TEST_TRACES + _file)
            obs = env.reset()
            _f = open('test_results/' + _file + '.csv', 'w')
            while True:
                action_prob = actor.predict(
                    np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
                #print(np.round(action_prob, 2))
                a = np.argmax(action_prob)
                # action_cumsum = np.cumsum(action_prob)
                # a = (action_cumsum > np.random.randint(
                   1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                obs, rew, done, _info = env.step(a)
                rew_list.append(rew)
                if done:
                    _f.close()
                    break
                #rew_list.append(rew)
                sending_rate, recv_bitrate, available_bandwidth, rtt, loss = _info['sending_rate'], _info[
                    'recv_bitrate'], _info['limbo_bytes_len'], _info['rtt'],  _info['loss']
                _f.write(str(sending_rate) + ',' +
                         str(recv_bitrate) + ',' + str(available_bandwidth) + ',' + str(loss) + ',' + str(rew) + '\n')
                _f.flush()
        rew_mean = np.mean(rew_list)
        _file = open('figs/test.txt','a')
        _file.write(str(rew_mean) + '\n')
        _file.close()



if __name__ == '__main__':
    main()
    os.system('mkdir figs/' + EP)
    os.system('python show.py figs/' + EP)
    os.system('python draw.py figs/test.txt')
