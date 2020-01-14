# add queuing delay into halo
import os
import numpy as np

RANDOM_SEED = 42
NORM = 40000.
S_INFO = 3  # seq
S_ALL_INFO = 2*S_INFO + 2
S_LEN = 20  # feature
N_STEP = 1440

DEFAULT_ACTION = 0

ACTION_SPACE = np.array([[0,0,0],[0.01,0,0],[0.05,0,0],[0.1,0,0],[0.2,0,0], \
                          [0,0.01,0], [0,0.05,0],[0,0.1,0],[0,0.2,0],\
                          [0,0,0.01], [0,0,0.05],[0,0,0.1],[0,0,0.2]])

class manmade_peak:
    def __init__(self, peak, block_ratio, peak_ratio, peak_workload):
        self.peak = np.array(peak)
        self.block_ratio = np.array(block_ratio)
        self.peak_ratio = np.array(peak_ratio)
        self.peak_workload = np.array(peak_workload)

    def change(self, peak, block_ratio, peak_ratio, peak_workload):
        self.peak = np.array(peak)
        self.block_ratio = np.array(block_ratio)
        self.peak_ratio = np.array(peak_ratio)
        self.peak_workload = np.array(peak_workload)

    def blockratio(self, workload):
        if(workload < self.peak):
            return self.block_ratio
        else:
            return min(1.0, (workload-self.peak)*(self.peak_ratio - self.block_ratio)/(self.peak_workload-self.peak) + self.block_ratio)


class LTSEnv():

    def __init__(self, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)
        self.epoch = 0
        self.total_iter = 0
        self.trace = []
        with open('./trace.txt', 'r') as f:
            for line in f:
                line = line.strip()
                self.trace.append(float(line.split(' ')[0]))
        self.trace_len = len(self.trace)
        self._iter = np.random.randint(0, self.trace_len)
        self._iter_rand = self._iter
        self.ratio = np.array([0.35, 0.35, 0.3])
        self.model1 = manmade_peak(5000, 0.4, 1.0, 20000)
        self.model2 = manmade_peak(11000, 0.12, 1.0, 30000)
        self.model3 = manmade_peak(30000, 0.25, 1.0, 75000)
        self.model_use1 = self.model1
        self.model_use2 = self.model2
        self.model_use3 = self.model3

        self.last_action = DEFAULT_ACTION
        self.state = np.zeros([S_ALL_INFO, S_LEN])
        self.reset()

    def seed(self, num):
        np.random.seed(num)

    def reset_trace(self):
        self._iter = self._iter_rand

    def reset(self):
        self._iter = 0
        #np.random.randint(0, self.trace_len)
        self._iter_rand = self._iter
        self.ratio = np.array([0.35, 0.35, 0.3])
        self.model1 = manmade_peak(5000, 0.4, 1.0, 20000)
        self.model2 = manmade_peak(11000, 0.12, 1.0, 30000)
        self.model3 = manmade_peak(30000, 0.25, 1.0, 75000)
        self.model_use1 = self.model1
        self.model_use2 = self.model2
        self.model_use3 = self.model3

        self.last_action = DEFAULT_ACTION
        self.state = np.zeros([S_ALL_INFO, S_LEN])

        workload, block_ratio = self.single_step(DEFAULT_ACTION)
        # reward = -np.dot(block_ratio, np.transpose(workload))
        state = np.roll(self.state, -1, axis=1)

        for i in range(S_INFO):
            state[i * 2, -1] = workload[i] / NORM
            state[i * 2 + 1, -1] = block_ratio[i]
        state[-2, -1] = np.sum(workload) / NORM
        state[-1, 0:S_INFO] = self.ratio
        self.state = state

        return self.state

    def single_step(self, action):
        choose_action = ACTION_SPACE[action]
        _worst_add = np.zeros(self.ratio.shape[0])
        _worst = np.argsort(-self.ratio)
        _remain = np.max(choose_action)
        _current = np.argmax(choose_action)
        _index = 0
        _flag = False

        while _remain > 0.0:
            _worst_index = _worst[_index]
            if _worst_index == _current and _flag == False:
                _flag = True
                _index += 1
            else:
                _delta = min(_remain, self.ratio[_worst_index] - 0.01)
                _worst_add[_worst_index] = _delta
                _remain -= _delta
                _index += 1
            _index %= self.ratio.shape[0]

        self.ratio = self.ratio + choose_action - _worst_add
        _workload = self.ratio * self.trace[self._iter]

        self.total_iter += 1
        self.total_iter = self.total_iter % N_STEP
        self._iter += 1
        self._iter = self._iter % self.trace_len

        block_ratio = np.array([self.model_use1.blockratio(_workload[0]), self.model_use2.blockratio(
            _workload[1]), self.model_use3.blockratio(_workload[-1])])
        return _workload, block_ratio

    def render(self):
        return

    def step(self, action):
        workload, block_ratio = self.single_step(action)
        reward = -np.dot(block_ratio, np.transpose(workload))
        state = np.roll(self.state, -1, axis=1)

        for i in range(S_INFO):
            state[i * 2, -1] = workload[i] / NORM
            state[i * 2 + 1, -1] = block_ratio[i]
        state[-2, -1] = np.sum(workload) / NORM
        state[-1, 0:S_INFO] = self.ratio
        self.state = state
        # hard-core
        # We assume that each person watch the video streaming with the bitrate of 1.2Mbps.
        w = workload * 1.2 / 8. / 1000. * np.array([0.72, 0.4, 0.8])
        return state, reward, False, {'workload': 0. - np.sum(w), 'ratio': block_ratio}
