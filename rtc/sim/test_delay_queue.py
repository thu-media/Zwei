import numpy as np
#import time
from TinyQueue import TinyQueue
from Packet import DelayedPacket, PartialPacket
import random
import os
import time
TRACE_FLODER = "./test_traces/"
TRAIN_SIM_FLODER = "./test_traces/"
INTERVAL = 50
BITRATE_TIME = 1.0
MTU_PACKET_SIZE = 1500


class NoloopDelayQueue:
    def reset_trace_core(self, cooked_file):
        _schedule = []
        f = open(cooked_file, "r")
        #print 'opening file ', TRAIN_SIM_FLODER + cooked_file
        if not f:
            print("error:open file")
        # temp = 0
        cooked_time, cooked_bw = [], []
        _base_time = 0.0
        for line in f:
            parse = line.split()
            if len(parse) > 1:
                cooked_time.append(float(parse[0]) - _base_time)
                cooked_bw.append(float(parse[1]) / 3.0)
                _base_time = float(parse[0])
        f.close()
        _basic_ms = 0.0
        for (_time, _bw) in zip(cooked_time, cooked_bw):
            if _time < 0.05:
                continue
            _bw *= _time
            _time *= 1000  # random_seed
            _bw *= 1024 * 1024
            _packets_count = int(_bw / MTU_PACKET_SIZE)
            if _packets_count == 0:
                _packets_count = 1
            _temp = np.random.randint(0, int(_time), _packets_count)
            _d_ms_array = _temp[np.argsort(_temp)] + _basic_ms
            for _d_ms in _d_ms_array:
                self._limbo.push(_d_ms)
            _basic_ms += _time
        return _base_time

    def reset_trace(self):
        #_start = time.time()
        _trace_file = self.s_name
        self.reset_trace_core(self.s_name)
    
    def reset_trace_core_v2(self, cooked_file):
        #_schedule = []
        f = open(cooked_file, "r")
        if not f:
            print("error:open file")
        self._basic_ms = 0
        #if len(self._limbo) > 0:
        #    self._basic_ms = self._limbo[-1]
        for _line in f:
            _d_ms = int(_line) +self._basic_ms
            self._limbo.push(_d_ms)

    def reset_trace_v2(self):
        _trace_file = self.s_name
        self.reset_trace_core_v2(self.s_name)

    def __init__(self, s_name=None, loss_rate=0.0, limbo_size=100):
        # np.random.seed(222)
        self.SERVICE_PACKET_SIZE = 1500
        self.s_name = s_name
        self._sender = TinyQueue()
        self._real_limbo = []
        self._real_limbo_size = limbo_size
        self._limbo_count = 0
        self._limbo = TinyQueue()

        #self._ms_delay = s_ms_delay
        self._loss_rate = loss_rate
        self._queued_bytes = 0
        #self._base_timestamp = base_timestamp
        self._packets_added = 0
        self._packets_dropped = 0
        #self._time = TinyTime()
        self._basic_ms = 0

        self.last_queuing_delay = 0

        self.reset_trace()
        self.rtt = np.random.random() * 200.
        self._out_ms, self._sender_ms = [], []
        self.time_start = -1.

    def write(self, packet, now):
        r = random.random()
        self._packets_added += 1
        if (r < self._loss_rate):
            self._packets_dropped += 1
            # print("%s,Stochastic drop of packet,packets_added so far %d,packets_dropped %d,drop rate %f" %
            #      (self._name, self._packets_added, self._packets_dropped,
            #       float(self._packets_dropped) / float(self._packets_added)))
        else:
            p = DelayedPacket(now, now, packet)
            self._sender.push(p)
            self._queued_bytes += packet

    def getfront(self):
        if self._limbo.size() <= 0:
            return None
        else:
            return self._limbo.front()

    def syncread(self, duration):
        # fix, is it right?
        if self.time_start < 0.:
            self.time_start = self.getfront()
        _timestart = self.time_start
        #print(_timestart)
        if self.getfront() is None:
            return None, None, None
        _out_bytes, _sender_bytes = 0., 0.
        _rtt = 0.
        _bytes_send = 0.

        for p in range(duration):
            _time = _timestart + p
            while True:
                _front = self.getfront()
                if _front is None:
                    break
                if _front >= _time:
                    break
                self._out_ms.append(_front)
                self._limbo.pop()
            while True:
                _delay_front = self._sender.front()
                if _delay_front is None:
                    break
                if _delay_front.entry_time >= _time:
                    break
                if len(self._sender_ms) < self._real_limbo_size:
                    self._sender_ms.append(_delay_front)
                # _sender_ms.append(_delay_front)
                self._sender.pop()
            while True:
                if len(self._out_ms) == 0:
                    break
                if len(self._sender_ms) == 0:
                    # state machine; drain
                    #_out_bytes += len(_out_ms)
                    break
                _sender_head = self._sender_ms[0].entry_time
                _out_head = self._out_ms[0]
                #assert _sender_head <= _out_head
                _delta = (_out_head - _sender_head)
                if _delta >= 0:
                    _rtt += (_out_head - _sender_head)
                    _out_bytes += 1.
                    _bytes_send += self._sender_ms[0].contents
                    _sender_bytes += 1.
                    self._sender_ms.pop(0)
                    self._out_ms.pop(0)
                else:
                    _sender_bytes += 1.
                    self._out_ms.pop(0)
        #_bytes_send = self.SERVICE_PACKET_SIZE * _out_bytes
        _av_bytes_send = self.SERVICE_PACKET_SIZE * _sender_bytes
        self.time_start += duration
        if _out_bytes > 0:
            return _rtt / _out_bytes, _bytes_send, _av_bytes_send
        else:
            return 0., _bytes_send, _av_bytes_send


if __name__ == "__main__":
    pass