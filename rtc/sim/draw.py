
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
import sys

LW = 3
SCALE = 300



def read_csv(filename):
    _file = open(filename, 'r')
    _tmp = []
    for _line in _file:
        _tmp.append(float(_line))
    return np.array(_tmp)

def moving_average(data, alpha=0.15):
    global max_val
    _tmp = []
    _val = data[0]
    for p in data:
        #max_val = max(max_val, p)
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return np.array(_tmp)

plt.switch_backend('Agg')
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['axes.labelweight'] = 'bold'
font = {'size': 28}
matplotlib.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(15, 6), dpi=100)
y_ = read_csv(sys.argv[1])
#ax1.grid(True)
ax1.set_title(sys.argv[1])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.set_ylim(1.0,1.95)
l4 = ax1.plot(y_, color='#49b8ff', lw=LW, label='original')
# for p in range(3):
#l4 = ax1.plot(moving_average(y_), color='red', lw=LW, label='ma')
ax1.legend(loc="upper left", framealpha=1, frameon=False, fontsize=14)
savefig(sys.argv[1] + '.png')
print('done')
