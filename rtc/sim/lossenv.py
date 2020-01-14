# add queuing delay into halo
import numpy as np
from scipy import interpolate
from LossDelayQueue import DelayQueue
#from SimTime import SimTime
import time

MTU_PACKET_SIZE = 1500

RANDOM_SEED = 42
PACKET_PAYLOAD_PORTION = 0.95

NOISE_LOW = 0.95
NOISE_HIGH = 1.05

FPS = 25
#ACTION_SPACE = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
ACTION_SPACE = [-0.5, -0.35, -0.2, -0.15, -
                0.05, -0., 0.05, 0.15, 0.2, 0.35, 0.5]
A_DIM = len(ACTION_SPACE)
DEFAULT_ACTION = 6
S_INFO = 8
S_LEN = 10  # take how many frames in the past
BUFFER_NORM_FACTOR = 8.0

RATE_PENALTY = 1.
RTT_PENALTY = 0.25
LOSS_PENALTY = 0.25


class NetworkEnv:
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.delay_queue = None
        self.simtime = 0.0

        self.last_sending_rate = 0.05  # 50kBps
        self.sending_rate_min = 0.05
        self.sending_rate_max = 0.5
        self.last_bit_rate = DEFAULT_ACTION
        self.bit_rate = DEFAULT_ACTION
        self.last_rtt = -1

        self.last_state = np.zeros((S_INFO, S_LEN))

    def reset(self):
        if self.delay_queue is not None:
            self.delay_queue.clear()
        self.delay_queue = None
        self.delay_queue = DelayQueue()
        self.simtime = 0.
        self.last_sending_rate = 0.3
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
        #_norm_send_bitrate = float(bit_rate) / float(BITRATE_LEVELS)
        _delay_gradient = np.abs(rtt - self.last_rtt)
        state[0, -1] = sending_rate  # last quality
        state[1, -1] = float(recv_bitrate) / delay / \
            BUFFER_NORM_FACTOR  # kilo byte / ms
        state[2, -1] = rtt  # max:500ms
        state[3, -1] = _delay_gradient  # max:500ms
        state[4, -1] = float(loss)  # changed loss
        state[5, -1] = bit_rate / len(ACTION_SPACE)  # action

        _fft = np.fft.fft(state[1])
        state[6] = _fft.real
        state[7] = _fft.imag

        self.last_rtt = rtt
        self.last_state = state
        self.last_bit_rate = bit_rate
        self.last_sending_rate = sending_rate
        return state

    def step(self, action):
        state = self.last_state
        bit_rate = action

        state = np.roll(state, -1, axis=1)
        sending_rate = 0.3
        # ACTION_SPACE[action]
        sending_rate = max(self.last_sending_rate *
                           (1. + ACTION_SPACE[bit_rate]), self.sending_rate_min)
        sending_rate = min(sending_rate, self.sending_rate_max)
        delay, loss, throughput, rtt, recv_bitrate, limbo_bytes_len = self.get_video_chunk(
            sending_rate)
        state = self.last_state
        rtt /= 1000.
        #_norm_send_bitrate = float(bit_rate) / float(BITRATE_LEVELS)
        _delay_gradient = (rtt - self.last_rtt)
        state[0, -1] = sending_rate  # last quality
        state[1, -1] = float(recv_bitrate) / delay / \
            BUFFER_NORM_FACTOR  # kilo byte / ms
        state[2, -1] = rtt  # max:500ms
        state[3, -1] = _delay_gradient  # max:500ms
        state[4, -1] = float(loss)  # changed loss
        state[5, -1] = action / len(ACTION_SPACE)  # action

        _fft = np.fft.fft(state[1])
        state[6] = _fft.real
        state[7] = _fft.imag

        reward = RATE_PENALTY * float(recv_bitrate) / delay \
            - RTT_PENALTY * rtt \
            - LOSS_PENALTY * loss
        reward *= 10.

        # print float(recv_bitrate) / delay, rtt, loss
        self.last_rtt = rtt
        self.last_state = state
        self.last_bit_rate = bit_rate
        self.last_sending_rate = sending_rate
        if rtt >= 10.:
            done = True
        else:
            done = False
        return state, reward, done, sending_rate, recv_bitrate, limbo_bytes_len, rtt, loss

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

        # for _t in range(len(_d_ms_array) - 1):
        #     self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[_t])

        # if _last_packet_len > 0:
        #     self.delay_queue.write(_last_packet_len, _d_ms_array[-1])
        # else:
        #     self.delay_queue.write(MTU_PACKET_SIZE, _d_ms_array[-1])

        self.simtime += timeslot
        _total_delay, _total_bytes_len, _limbo_bytes_len = self.delay_queue.syncread(_d_ms_array,
                                                                                     timeslot)
        #assert  _total_delay < 100 * 1000
        return _total_delay, _total_bytes_len, _limbo_bytes_len

    def get_video_chunk(self, quality, timeslot=1000):

        choose_quality = quality  # self.video_size[quality]

        queuing_delay, _total_bytes_len, _limbo_bytes_len = self.send_video_queue(
            choose_quality, timeslot)
        _total_bytes_len = float(_total_bytes_len) / float(1024 * 1024)
        _limbo_bytes_len = float(_limbo_bytes_len) / float(1024 * 1024)

        #throughput * duration * PACKET_PAYLOAD_PORTION
        packet_payload = _total_bytes_len * \
            np.random.uniform(NOISE_LOW, NOISE_HIGH)
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
