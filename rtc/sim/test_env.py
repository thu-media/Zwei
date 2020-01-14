# add queuing delay into halo
import numpy as np
from test_delay_queue import NoloopDelayQueue as DelayQueue
#from SimTime import SimTime
import time

MTU_PACKET_SIZE = 1500

RANDOM_SEED = 42
PACKET_PAYLOAD_PORTION = 0.95

NOISE_LOW = 0.9
NOISE_HIGH = 1.1

FPS = 25
#ACTION_SPACE = [-0.5, -0.2, -0.1, -0., 0.1, 0.2, 0.5]
ACTION_SPACE = [-0.4, -0.2, -0.1, 0., 0.1, 0.2, 0.3]
A_DIM = len(ACTION_SPACE)
DEFAULT_ACTION = 4
S_INFO = 6
S_LEN = 8  # take how many frames in the past
BUFFER_NORM_FACTOR = 8.0

RATE_PENALTY = 1.
RTT_PENALTY = 0.5
LOSS_PENALTY = 0.5


class NetworkEnv:
    def __init__(self, trace_file, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.delay_queue = None
        self.simtime = 0.0
        self.trace_file = trace_file
        self.last_sending_rate = 0.1  # 50kBps
        self.last_encoding_rate = 0.25  # 50kBps
        self.sending_rate_min = 0.025
        self.sending_rate_max = 0.5
        self.last_bit_rate = DEFAULT_ACTION
        self.bit_rate = DEFAULT_ACTION
        self.last_rtt = -1

        self.last_state = np.zeros((S_INFO, S_LEN))
        
    def fec(self, rtt, loss):
        delta = 140. / (rtt * 1000.)
        alpha = loss ** (1 + delta)
        if alpha < 0.02:
            return 0.2
        elif alpha < 0.05:
            return 0.3
        elif alpha < 0.1:
            return 0.5
        else:
            return 0.7

    def reset(self):
        if self.delay_queue is not None:
            self.delay_queue.clear()
        self.delay_queue = None
        #queue_len = np.random.randint(1, 100)
        self.delay_queue = DelayQueue(self.trace_file)  # queue_len * 10)
        self.simtime = 0.
        self.last_sending_rate = 0.1
        # ACTION_SPACE[DEFAULT_ACTION]
        self.last_bit_rate = DEFAULT_ACTION
        bit_rate = DEFAULT_ACTION
        self.last_rtt = -1
        self.last_state = np.zeros((S_INFO, S_LEN))
        sending_rate = self.last_sending_rate
        delay, loss, throughput, rtt, recv_bitrate, limbo_bytes_len = self.get_video_chunk(
            sending_rate)
        rtt /= 1000.
        if self.last_rtt < 0:
            self.last_rtt = rtt
        state = self.last_state
        _delay_gradient = (rtt - self.last_rtt)

        state[0, -1] = sending_rate  # last quality
        state[1, -1] = float(recv_bitrate) / delay
        state[2, -1] = float(rtt)
        state[3, -1] = _delay_gradient  # max:500ms
        state[4, -1] = float(loss)  # changed loss
        state[5, -1] = float(bit_rate) / len(ACTION_SPACE)  # action

        self.last_rtt = rtt
        self.last_state = state
        self.last_bit_rate = bit_rate
        self.last_sending_rate = sending_rate
        return state

    def step(self, action):
        state = self.last_state
        bit_rate = action

        state = np.roll(state, -1, axis=1)
        #sending_rate = 0.3
        # ACTION_SPACE[action]
        if ACTION_SPACE[bit_rate] < 0:
            sending_rate = max(self.last_sending_rate *
                           (1. + ACTION_SPACE[bit_rate]), self.sending_rate_min)
            sending_rate = min(sending_rate, self.sending_rate_max)
        else:
            sending_rate = max(self.last_sending_rate + 0.5 * ACTION_SPACE[bit_rate] * self.sending_rate_max, self.sending_rate_min)
            sending_rate = min(sending_rate, self.sending_rate_max)

        # sending_rate *= np.random.uniform(0.95, 1.05)

        delay, loss, throughput, rtt, recv_bitrate, limbo_bytes_len = self.get_video_chunk(
            sending_rate)
        #state = self.last_state
        _info = {}
        _info['sending_rate'] = sending_rate
        _info['recv_bitrate'] = recv_bitrate
        _info['limbo_bytes_len'] = limbo_bytes_len
        _info['rtt'] = rtt
        _info['loss'] = loss
        if delay is None:
            done = True
            return None, 0., done, _info
        else:
            rtt /= 1000.
            _delay_gradient = (rtt - self.last_rtt)

            state[0, -1] = sending_rate  # last quality
            state[1, -1] = float(recv_bitrate) / delay
            state[2, -1] = float(rtt)
            state[3, -1] = _delay_gradient  # max:500ms
            state[4, -1] = float(loss)  # changed loss
            state[5, -1] = float(bit_rate) / len(ACTION_SPACE)  # action

            reward = RATE_PENALTY * float(recv_bitrate) / delay \
                - RTT_PENALTY * 10. * (_delay_gradient) \
                - LOSS_PENALTY * np.minimum(loss, 0.3) \
                - 3. * LOSS_PENALTY * np.maximum(loss - 0.3, 0.)
            reward -= np.abs(sending_rate - self.last_sending_rate)
            reward *= 10.
            # reward = float(recv_bitrate) / delay - 0.5 * rtt - 0.5 * loss
            #reward = np.power(sending_rate, 0.9) - 900. * _delay_gradient - 11.35 * sending_rate * loss
            # print float(recv_bitrate) / delay, rtt, loss
            self.last_rtt = rtt
            self.last_state = state
            self.last_bit_rate = bit_rate
            self.last_sending_rate = sending_rate
            # if rtt >= 10.:
            #    done = True
            #reward = -100.
            # else:
            done = False
            return state, reward, done, _info

    def send_video_queue(self, video_quality, timeslot):
        # another fast algorithm with random walk - poisson process
        video_quality *= 1024 * 1024
        video_quality = int(video_quality)
        _packet_count = int(video_quality / MTU_PACKET_SIZE)
        _last_packet_len = video_quality % MTU_PACKET_SIZE
        if _last_packet_len > 0:
            _packet_count += 1
        _temp = np.random.randint(0, int(timeslot), _packet_count)
        _d_ms_array = _temp[np.argsort(_temp)] + self.simtime

        for _t in range(len(_d_ms_array) - 1):
            self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[_t])

        if _last_packet_len > 0:
            self.delay_queue.write(_last_packet_len, _d_ms_array[-1])
        else:
            self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[-1])

        self.simtime += timeslot

        _total_delay, _total_bytes_len, _limbo_bytes_len = self.delay_queue.syncread(
            timeslot)
        #assert  _total_delay < 100 * 1000
        return _total_delay, _total_bytes_len, _limbo_bytes_len

    def get_video_chunk(self, quality, timeslot=1000):

        choose_quality = quality  # self.video_size[quality]

        queuing_delay, _total_bytes_len, _limbo_bytes_len = self.send_video_queue(
            choose_quality, timeslot)
        if queuing_delay is not None:
            _total_bytes_len = float(_total_bytes_len) / float(1024 * 1024)
            _limbo_bytes_len = float(_limbo_bytes_len) / float(1024 * 1024)

            #throughput * duration * PACKET_PAYLOAD_PORTION
            packet_payload = _total_bytes_len 
            #* \
            #    np.random.uniform(NOISE_LOW, NOISE_HIGH)
            # use the delivery opportunity in mahimahi
            loss = 0.0  # in ms
            if packet_payload > choose_quality:
                loss = 0
                # add a multiplicative noise to loss
                _real_packet_payload = choose_quality * \
                    np.random.uniform(NOISE_LOW, NOISE_HIGH)
            else:
                loss = 1 - packet_payload / choose_quality
                _real_packet_payload = packet_payload

            return timeslot / 1000.0, loss, \
                packet_payload, \
                queuing_delay, _real_packet_payload, _limbo_bytes_len
        else:
            return None, None, None, None, None, None


if __name__ == "__main__":
    env = NetworkEnv()
    env.reset()
