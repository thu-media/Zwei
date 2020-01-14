import os
import numpy as np
from rules import rules, update_elo, update_elo_2

class tracepool(object):
    def __init__(self, testdir='./hd-fs-test/'):
        self.test_dir = testdir
        self.abr_list = ['rb', 'bb', 'rmpc', 'hyb', 'rl']
        #[sabre.ThroughputRule, sabre.ConstrainRule]
        self.sample_list = []
        self.trace_list = []
        self.test_list = []

        for p in os.listdir(self.test_dir):
            self.test_list.append(p)

        self.elo_score = []

        for p in self.abr_list:
            self.sample_list.append([])
            self.elo_score.append(1000.0)
        self.sample()

    def sample(self):
        print('generating samples')
        for _trace in self.get_test_set():
            for _index, _abr in enumerate(self.abr_list):
                _f = open('./norway/log_sim_' + _abr + '_' + _trace, 'r')
                _bitrate, _rebuffer = [], []
                for lines in _f:
                    #110.64486915972032	2850	19.235901151929067	0.0	1341201	5257.885326692943	0.5	2.85
                    sp = lines.split()
                    if(len(sp) > 0):
                        _bitrate.append(float(sp[1]))
                        _rebuffer.append(float(sp[3]))
                _bitrate_mean = np.mean(_bitrate[1:])
                _rebuffer_mean = np.mean(_rebuffer[1:])

                self.sample_list[_index].append([_bitrate_mean, _rebuffer_mean])

        for _index0 in range(len(self.abr_list)):
            _battle = []
            for _index in range(len(self.abr_list)):
                tmp = [0, 0, 0]
                for _trace_index in range(len(self.get_test_set())):
                    res = rules([self.sample_list[_index0][_trace_index],
                                 self.sample_list[_index][_trace_index]])
                    if _index0 < _index:
                        self.elo_score = update_elo(self.elo_score,
                                                    _index0, _index, res)
                    tmp[np.argmax(res)] += 1
                    tmp[-1] += 1
                _battle.append(round(tmp[0] * 100.0 / tmp[-1], 2))
            print(_index0, _battle)
        log_file = open('elo_baseline.txt', 'w')
        for p in self.elo_score:
            log_file.write(str(p) + ' ')
        log_file.close()
        print(self.elo_score)

    def get_test_set(self):
        return self.test_list

    def get_list(self):
        return self.trace_list

    def get_list_shuffle(self, sample=15):
        np.random.shuffle(self.trace_list)
        return self.trace_list[:sample]

    def battle(self, agent_elo, agent_result):
        ret = []
        for p in range(len(agent_result[0])):
            res, agent_elo = self._battle_index(agent_elo, agent_result, p)
            ret.append(res)
        return ret, agent_elo

    def _battle_index(self, agent_elo, agent_result, index):
        ret = []
        for _index in range(len(self.abr_list)):
            tmp = [0, 0, 0]
            for _trace_index in range(len(self.get_test_set())):
                res = rules(
                    [agent_result[_trace_index][index], self.sample_list[_index][_trace_index]])
                agent_elo = update_elo_2(
                    agent_elo, self.elo_score, index, _index, res)
                # if res[0] != 0:
                tmp[np.argmax(res)] += 1
                tmp[-1] += 1
            ret.append(round(tmp[0] * 100.0 / tmp[-1], 2))
        return ret, agent_elo
