
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
#from pykalman import KalmanFilter
import sys

LW = 1.5
SCALE = 300


def read_csv(filename='elo.txt'):
    _file = open(filename, 'r')
    _tmpA = []
    for _line in _file:
        _lines = _line.split(' ')
        _tmpA.append(float(_lines[0]))
    _file.close()
    return np.array(_tmpA)


def read_history(filename='elo_baseline.txt'):
    _file = open(filename, 'r')
    _tmpA = []
    for _line in _file:
        for _l in _line.split(' '):
            if _l is not '':
                _tmpA.append(float(_l))
    _file.close()
    return np.array(_tmpA)


def moving_average(data, alpha=0.4):
    _tmp = []
    _val = data[0]
    for p in data:
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return np.array(_tmp)

plt.switch_backend('Agg')

plt.rcParams['axes.labelsize'] = 12
font = {'size': 12}
matplotlib.rc('font', **font)
fig, ax1 = plt.subplots(figsize=(7, 5))
plt.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.91)
_a = read_csv()
_tmp = read_history()

ax1.set_ylabel('ELO')
ax1.set_xlabel('Step')
l4 = ax1.plot(_a, color='darkblue', lw=LW, alpha=0.3)
l4 = ax1.plot(moving_average(_a), color='black', lw=LW, label='Zwei')
_label = ['Rate-based', 'Buffer-based', 'MPC', 'HYB', 'Pensieve', 'Comyco']
_color = ['darkred', 'darkblue', 'salmon', 'gray', 'pink', 'darkgreen']

for index, p in enumerate(_tmp):
    ax1.hlines(p, 0, len(_a), linestyles="dashed", color = _color[index], lw = LW, label=_label[index])

ax1.legend(loc='upper center', ncol=4,
          bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=12)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
savefig('elo.png')
