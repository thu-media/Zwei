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
TEST_TRACES = './test_traces/'


def main():

    # os.system('mkdir test_results')
    rew_list, sending_list, loss_list, rtt_list = [], [], [], []
    for _file in os.listdir(TEST_TRACES):
        env = NetworkEnv(TEST_TRACES + _file)
        obs, _info = env.reset()
        _f = open('webrtc_results/' + _file + '.csv', 'w')
        sending_rate, recv_bitrate, available_bandwidth, rtt, loss = _info['sending_rate'], _info[
                'recv_bitrate'], _info['limbo_bytes_len'], _info['rtt'],  _info['loss']
        last_sending_rate = 0.1
        last_rtt = rtt
        rtt_queue = []
        rtt_tmp, loss_tmp = [], []
        rtt_cum = 0.
        while True:
            send_bit = 0.
            base_rtt = rtt - last_rtt
            
            # we delete kalman-filter
            # only use trend-line filter.
            rtt_queue.append(base_rtt)
            if len(rtt_queue) > 20:
                rtt_queue.pop(0)
            inv_rtt = 1. / (np.array(rtt_queue) + 1e-8)
            rtt_normal = len(rtt_queue) / np.sum(inv_rtt)
            rtt_normal = rtt_normal / 1000.
            rtt_normal *= 0.08
            
            # delay-based
            if rtt_normal < 0.0018:
                #increase
                send_bit = 1.05 * sending_rate
            elif rtt_normal < 0.001:
                #hold
                send_bit = sending_rate
            else:
                #decrease
                send_bit = 0.85 *sending_rate
            # loss-based
            if loss < 0.02:
                loss_send = 1.05 * sending_rate
            elif loss < 0.1:
                loss_send = sending_rate
            else:
                loss_send = sending_rate * (1 - 0.5 * loss)
            # print(loss, loss_send, send_bit, sending_rate)
            sending_rate = np.minimum(loss_send, send_bit)
            sending_rate = np.clip(sending_rate, 0.01, 0.5)
            last_rtt = rtt
            obs, rew, done, _info = env.step(sending_rate)
            rew_list.append(rew)
            if done:
                rtt_tmp.append(np.percentile(rtt_list, 95))
                loss_tmp.append(np.percentile(loss_list, 95))
                #loss_list.sort()
                #print(loss_list)
                #print(rtt_tmp[-1], loss_tmp[-1])
                rtt_list, loss_list = [], []
                _f.close()
                break
            #rew_list.append(rew)
            sending_rate, recv_bitrate, available_bandwidth, rtt, loss = _info['sending_rate'], _info[
                'recv_bitrate'], _info['limbo_bytes_len'], _info['rtt'],  _info['loss']
            
            sending_list.append(sending_rate)
            loss_list.append(loss)
            rtt_list.append(rtt)

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
    os.system('mkdir figs/' + EP)
    # os.system('python show.py figs/' + EP)
    # os.system('python draw.py figs/test.txt')
