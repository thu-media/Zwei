
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import plot, savefig
import sys
TEST_FOLDER = 'test_results/'
LW = 3
SCALE = 100



def read_csv(filename):
    _file = open(TEST_FOLDER + filename, 'r')
    _tmp, _thr, _loss = [], [], []
    _tmp_idx = 0
    _idx = []
    for _line in _file:
        _sp_lines = _line.split(',')
        _tmp.append(float(_sp_lines[0]))
        _thr.append(float(_sp_lines[2]))
        _loss.append(float(_sp_lines[3]))
        _idx.append(_tmp_idx)
        _tmp_idx += 1
    return np.array(_idx), np.array(_tmp), np.array(_thr), np.array(_loss)

def moving_average(data, alpha=0.6):
    global max_val
    _tmp = []
    _val = data[0]
    for p in data:
        #max_val = max(max_val, p)
        _val = p * (1.0 - alpha) + _val * alpha
        _tmp.append(_val)
    return np.array(_tmp)

#os.system('mkdir figs')
dirs = sys.argv[1]
for p in os.listdir(TEST_FOLDER):
    plt.switch_backend('Agg')
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['axes.labelweight'] = 'bold'
    font = {'size': 18}
    matplotlib.rc('font', **font)
    fig, ax1 = plt.subplots(figsize=(15, 6))
    #plt.subplots_adjust(left=0.12, bottom=0.15, right=0.97, top=0.98)
    idx_, x_, y_, loss_ = read_csv(p)
    ax1.set_title(p)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    #ax1.set_ylim(0.,5.)
    plt.subplots_adjust(left=0.08, bottom=0.05, right=0.99, top=0.95)
    l4 = ax1.plot(y_, '--', color='red', lw=LW, label='available bandwidth')
    ax1.plot(x_, color='black', lw=LW, label='sending rate')
    ax1.scatter(idx_, loss_, s=25., color='blue', label='loss')
    # ax1.plot(loss_, color='lightblue', lw=LW, label='loss')
    ax1.legend(framealpha=1, frameon=False, fontsize=20)
    savefig(dirs + '/' + p + '.png')
