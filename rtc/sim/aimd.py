import numpy as np
import sys
from aimd_env import NetworkEnv
from matplotlib import pyplot as plt
import os

S_DIM = [6, 8]
A_DIM = 7
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RAND_RANGE = 1000
EP = sys.argv[1]
TEST_TRACES = './test_traces_1/'


def main():

    # os.system('mkdir test_results')
    rew_list, sending_list, loss_list, rtt_list = [], [], [], []
    for _file in os.listdir(TEST_TRACES):
        rtt_tmp, loss_tmp = [], []
        env = NetworkEnv(TEST_TRACES + _file)
        obs, _info = env.reset()
        _f = open('aimd_results/' + _file + '.csv', 'w')
        sending_rate, recv_bitrate, available_bandwidth, rtt, loss = _info['sending_rate'], _info[
                'recv_bitrate'], _info['limbo_bytes_len'], _info['rtt'],  _info['loss']
        last_sending_rate = 0.1
        while True:
            if loss < 0.3:
                if sending_rate < last_sending_rate:
                    sending_rate += 0.01
                else:
                    sending_rate += 0.005
            else:
                buff_rate = np.maximum(sending_rate - last_sending_rate, 0.)
                last_sending_rate = sending_rate
                sending_rate = (sending_rate + buff_rate) / 2.
            sending_rate = np.clip(sending_rate, 0., 0.5)
            last_sending_rate = np.clip(last_sending_rate, 0., 0.5)
            obs, rew, done, _info = env.step(sending_rate)
            rew_list.append(rew)
            sending_list.append(sending_rate)
            loss_list.append(loss)
            rtt_list.append(rtt)
            if done:
                rtt_tmp.append(np.percentile(rtt_list, 95))
                loss_tmp.append(np.percentile(loss_list, 95))
                rtt_list, loss_list = [], []
                _f.close()
                break
            #rew_list.append(rew)
            sending_rate, recv_bitrate, available_bandwidth, rtt, loss = _info['sending_rate'], _info[
                'recv_bitrate'], _info['limbo_bytes_len'], _info['rtt'],  _info['loss']
            _f.write(str(sending_rate) + ',' +
                        str(recv_bitrate) + ',' + str(available_bandwidth) + ',' + str(loss) + ',' + str(rew) + '\n')
            _f.flush()
    rew_mean = np.mean(rew_list)
    sending_mean = np.mean(sending_list)
    loss_mean = np.mean(loss_tmp)
    rtt_mean = np.mean(rtt_tmp)
    _file = open('figs/test.txt','a')
    _file.write(str(EP) + ',' + str(rew_mean) + ',' + str(sending_mean) + ',' + \
        str(loss_mean) + ',' + str(rtt_mean) + '\n')
    _file.close()



if __name__ == '__main__':
    main()
    # os.system('mkdir figs/' + EP)
    # os.system('python show.py figs/' + EP)
    # os.system('python draw.py figs/test.txt')
