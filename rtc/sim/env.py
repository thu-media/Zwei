# add queuing delay into halo
import numpy as np
from LossDelayQueue import LossDelayQueue as DelayQueue
#from SimTime import SimTime
import time
from traceloader import TraceLoader

MTU_PACKET_SIZE = 1500

RANDOM_SEED = 42
PACKET_PAYLOAD_PORTION = 0.95

NOISE_LOW = 0.9
NOISE_HIGH = 1.1

FPS = 25
#ACTION_SPACE = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
#ACTION_SPACE = [-0.5, -0.35, -0.2, -0.15, -
#                0.05, -0., 0.05, 0.15, 0.2, 0.35, 0.5]
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

EPS = 0.2

class NetworkEnv:
    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.delay_queue = None
        self.simtime = 0.0

        self.last_sending_rate = 0.1  # 50kBps
        self.last_encoding_rate = 0.25
        self.sending_rate_min = 0.025
        self.sending_rate_max = 0.5
        self.last_bit_rate = DEFAULT_ACTION
        self.last_recv_rate = 0.1
        self.bit_rate = DEFAULT_ACTION
        self.last_rtt = -1

        self.queue_len = np.random.randint(1, 1000) * 10.
        self.base_loss = np.random.uniform(0., 0.1)
        self.base_rtt = np.random.random() * 600. + 20.
        self.idx = 0.
        self.limbo_idx = 0.
        
        self.last_state = np.zeros((S_INFO, S_LEN))
        self.base_loss = np.random.uniform(0., 0.1)

        self.trace_loader = TraceLoader()
        self.reset_trace()

    def reset_trace(self):
        if self.delay_queue is not None:
            self.delay_queue.clear()
        self.delay_queue = None
        self.queue_len = np.random.randint(1, 1000) * 10.
        self.base_loss = np.random.uniform(0., 0.1)
        self.base_rtt = np.random.random() * 600. + 20.
        self.idx = np.random.random()
        self.limbo_idx = np.random.random()
        self.c_t, self.c_bw = self.trace_loader.read_trace()
        # self.delay_queue = DelayQueue(self.random_seed, self.base_loss, self.queue_len, self.base_rtt)

    def reset(self):
        self.delay_queue = DelayQueue(self.random_seed, self.base_loss, self.queue_len, \
            self.base_rtt, self.c_t, self.c_bw)
        self.simtime = 0.

        self.last_sending_rate = 0.1
        self.last_encoding_rate = 0.25
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
        _delay_gradient = (rtt - self.last_rtt)
        #last_encoding_rate = sending_rate * (1 - self.fec(rtt, loss)) * np.random.uniform(0.95, 1.05)
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
        self.last_recv_rate = recv_bitrate

        _info = {}
        #self.base_loss = np.random.uniform(0., 0.5)
        return state

    def step(self, action):
        state = self.last_state
        bit_rate = action

        state = np.roll(state, -1, axis=1)
        # AIMD
        if ACTION_SPACE[bit_rate] < 0:
            sending_rate = np.clip(self.last_sending_rate * (1. + ACTION_SPACE[bit_rate]), self.sending_rate_min, self.sending_rate_max)
        else:
            sending_rate = np.clip(self.last_sending_rate + 0.5 * ACTION_SPACE[bit_rate] * self.sending_rate_max, self.sending_rate_min, self.sending_rate_max)
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
            # print('hi')
            done = True
            return None, 0.,done, _info
        else:
            rtt /= 1000.
            _delay_gradient = (rtt - self.last_rtt)
            state[0, -1] = sending_rate  # last quality
            state[1, -1] = float(recv_bitrate) / delay
            state[2, -1] = float(rtt)
            state[3, -1] = _delay_gradient  # max:500ms
            state[4, -1] = float(loss)  # changed loss
            state[5, -1] = float(action) / len(ACTION_SPACE)  # action

            reward = RATE_PENALTY * float(recv_bitrate) / delay \
               - RTT_PENALTY * 10. * np.abs(_delay_gradient) \
               - LOSS_PENALTY * np.minimum(loss, 0.3) \
               - 3. * LOSS_PENALTY * np.maximum(loss - 0.3, 0.)
            reward -= np.abs(sending_rate - self.last_sending_rate)
            reward *= 10.
            
            self.last_rtt = rtt
            self.last_state = state
            self.last_bit_rate = bit_rate
            self.last_sending_rate = sending_rate
            self.last_recv_rate = recv_bitrate

            if rtt >= 10.:
                done = True
                # a strong penalty
                reward = 0.
            else:
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
        return _total_delay, _total_bytes_len, _limbo_bytes_len

    def get_video_chunk(self, quality, timeslot=1000):

        choose_quality = quality  # self.video_size[quality]

        queuing_delay, _total_bytes_len, _limbo_bytes_len = self.send_video_queue(
            choose_quality, timeslot)
            
        if queuing_delay is not None:
            _total_bytes_len = float(_total_bytes_len) / float(1024 * 1024)
            _limbo_bytes_len = float(_limbo_bytes_len) / float(1024 * 1024)

            #throughput * duration * PACKET_PAYLOAD_PORTION
            packet_payload = _total_bytes_len #* \
            #    np.random.uniform(NOISE_LOW, NOISE_HIGH)
            # use the delivery opportunity in mahimahi
            loss = 0.  # in ms
            if packet_payload > choose_quality:
                loss = 0.
                # add a multiplicative noise to loss
                _real_packet_payload = choose_quality #* \
                    #np.random.uniform(NOISE_LOW, NOISE_HIGH)
            else:
                loss = 1. - packet_payload / choose_quality
                _real_packet_payload = packet_payload #* \
                    #np.random.uniform(NOISE_LOW, NOISE_HIGH)

            #loss += self.base_loss

            return timeslot / 1000.0, loss, \
                packet_payload, \
                queuing_delay, _real_packet_payload, _limbo_bytes_len
        else:
            return None, None, None, None, None, None

if __name__ == "__main__":
    env = NetworkEnv()
    env.reset()
    for i in range(100):
        env.step(0)

