# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:00:16 2017

@author: RickZz
"""

import numpy as np
import gc


class TinyQueue:
    def __init__(self, s_name = 'Queue', maxsize=1024 * 1024 * 5):
        self._queue = [0] * maxsize
        self._front = 0
        self._end = 0
        self._maxsize = maxsize
        self._size = 0
        self._starttime = 0
        self.s_name = s_name

    def IsEmpty(self):
        # if(self._front - self._end == 0):
        return self.size() == 0

    def IsFull(self):
        return self.size() == self._maxsize - 1

    def debug(self):
        return self._queue

    def push(self, pkt):
        if(self.IsFull()):
            print("Queue Is full")
            self._queue += [0] * 1024 * 1024
        self._end = (self._end + 1) % self._maxsize
        self._queue[self._end] = pkt
        self._size += 1

    def push_back(self, pkt):
        self._queue.append(pkt)

    def pop(self):
        if(self.IsEmpty()):
            #self._queue += [0] * 1024 * 1024
            print(self.s_name + ":Queue is empty")
        else:
            pkt = self.front()
            del pkt
            self._front = (self._front + 1) % self._maxsize
            self._size -= 1
            self._starttime += 1
            # gc.collect()

    def size(self):
        return max(self._size, 0)

    def front(self):
        if(self.IsEmpty()):
            #print(self.s_name + ":Queue is empty")
            return None
        else:
            return self._queue[(self._front + 1) % self._maxsize]

    def clear(self):
        while self.size() > 0:
            self.pop()
        #self._queue = []

    def start(self):
        self._starttime = 0

    def stop(self):
        return self._starttime
