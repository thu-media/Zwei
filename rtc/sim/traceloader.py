import os
import numpy as np

class TraceLoader:
    def __init__(self, Floder='./traces/'):
        self.floder = Floder
    
    def read_trace(self):
        trace_files = os.listdir(self.floder)
        _test_num = len(trace_files)
        idx = np.random.randint(_test_num)
        cooked_file = trace_files[idx]
        f = open(self.floder + cooked_file, "r")
        if not f:
            print("error:open file")
        cooked_time, cooked_bw = [], []
        _base_time = 0.0
        for line in f:
            parse = line.split()
            if len(parse) > 1:
                _time = float(parse[0]) - _base_time
                _bw = float(parse[1])
                cooked_time.append(_time)
                cooked_bw.append(_bw / 3.)
                _base_time = float(parse[0])
        f.close()
        cooked_rand_idx = int(len(cooked_time) * np.random.random())
        cooked_time = cooked_time[cooked_rand_idx:] + cooked_time[:cooked_rand_idx]
        cooked_bw = cooked_bw[cooked_rand_idx:] + cooked_bw[:cooked_rand_idx]
        return cooked_time, cooked_bw
