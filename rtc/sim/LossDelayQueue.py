import numpy as np
from TinyQueue import TinyQueue
from Packet import DelayedPacket, PartialPacket
import random
import os
import time
#from matplotlib import pyplot as plt
TRAIN_SIM_FLODER = "./traces/"
RAW_FLODER = "./raw_traces/"


class LossDelayQueue:
    
    def reset_trace(self):
        for (_time, _bw) in zip(self.cooked_time, self.cooked_bw):
            if _time < 0.01:
                continue
            _bw *= _time
            _time *= 1000  # random_seed
            _bw *= 1024 * 1024  # * _rand
            _packets_count = int(_bw / self.SERVICE_PACKET_SIZE)
            if _packets_count == 0:
                _packets_count = 1
            _temp = np.random.randint(0, int(_time), _packets_count)
            _d_ms_array = _temp[np.argsort(_temp)] + self._basic_ms
            for _d_ms in _d_ms_array:
                self._limbo.append(_d_ms)
            self._basic_ms += _time

    def __init__(self, random_seed = 42, loss_rate=0.0, 
        limbo_size=100, base_rtt = 300.,
        cooked_time = [], cooked_bw = []):

        np.random.seed(random_seed)
        self.SERVICE_PACKET_SIZE = 1500
        #self.s_name = s_name
        self._sender = TinyQueue('sender')
        self._real_limbo = []
        self._real_limbo_size = limbo_size
        self._limbo_count = 0
        #self._limbo = TinyQueue('out')
        self._limbo = []
        self._limbo_idx = 0

        self.cooked_time = cooked_time
        self.cooked_bw = cooked_bw

        #self._ms_delay = s_ms_delay
        self._loss_rate = loss_rate
        self._queued_bytes = 0
        #self._base_timestamp = base_timestamp
        self._packets_added = 0
        self._packets_dropped = 0
        #self._time = TinyTime()
        self._basic_ms = 0.

        self.last_queuing_delay = 0

        self.reset_trace()
        #if limbo_idx >= 0:
        #    self._limbo_idx = int(limbo_idx * len(self._limbo))
        self.rtt = base_rtt
        self._out_ms, self._sender_ms = [], []
        self.time_start = -1.

    def clear(self):
        if self._sender:
            self._sender.clear()
        #if self._limbo:
        #    self._limbo.clear()
        self._limbo = []
        self._out_ms = []
        self._sender_ms = []
        self.time_start = -1.

    def write(self, packet, now):
        r = random.random()
        self._packets_added += 1
        if (r < self._loss_rate * np.random.uniform(0.95, 1.05)):
            self._packets_dropped += 1
            # print("%s,Stochastic drop of packet,packets_added so far %d,packets_dropped %d,drop rate %f" %
            #      (self._name, self._packets_added, self._packets_dropped,
            #       float(self._packets_dropped) / float(self._packets_added)))
        else:
            p = DelayedPacket(now, now, packet)
            self._sender.push(p)
            self._queued_bytes += packet

    def getfront(self):
        _v = self._limbo[self._limbo_idx]
        return _v

    def syncread(self, duration):
        # fix, is it right?
        if self.time_start < 0.:
            self.time_start = self.getfront()
        _timestart = self.time_start
        _out_bytes, _sender_bytes = 0., 0.
        _rtt = 0.
        _bytes_send = 0.

        for p in range(duration):
            _time = _timestart + p
            while True:
                _front = self.getfront()
                if _front is None:
                    break
                    # return None, None, None
                if _front >= _time:
                    break
                self._out_ms.append(_front)
                self._limbo_idx += 1
                if self._limbo_idx >= len(self._limbo):
                    _tail = self._limbo[-1] - self._limbo[0]
                    for i in range(len(self._limbo)):
                        self._limbo[i] += _tail
                    self._limbo_idx = 0
                #self._limbo.pop()
            while True:
                _delay_front = self._sender.front()
                if _delay_front is None:
                    break
                    # return None, None, None
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
            return _rtt / _out_bytes + self.rtt * np.random.uniform(0.95, 1.05), _bytes_send, _av_bytes_send
        else:
            return 0. + self.rtt, _bytes_send, _av_bytes_send


if __name__ == "__main__":
    pass
