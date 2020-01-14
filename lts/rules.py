import numpy as np
import elo
EPS = 100.0


def rules(agent_result):
    return pareto_rules(agent_result)

def multi_rules(agent_result):
    p = 0
    for q in range(1, len(agent_result)):
        res = rules([agent_result[p], agent_result[q]])
        res_idx = np.argmax(res)
        if res_idx == 0:
            p = q
    return p
    
def pareto_rules(agent_results):
    r_0, b_0 = agent_results[0]
    r_1, b_1 = agent_results[1]
    _tmp = [0., 0.]
    if np.abs(b_0 - b_1) < 1e-2 and np.abs(r_0 - r_1) < 10.:
        _idx = np.random.randint(2)
        _tmp[_idx] = 1.
    elif np.abs(b_0 - b_1) < 1e-2:
        _tmp[np.argmax([r_0, r_1])] = 1.
    else:
        _tmp[np.argmin([b_0, b_1])] = 1.
    return _tmp

def poor_rules(agent_results):
    b_0, r_0, _ = agent_results[0]
    b_1, r_1, _ = agent_results[1]
    tmp = [0, 0]
    tmp[np.argmax([b_0, b_1])] = 1.0
    return tmp

def threshold_rules(agent_results, threshold=0.01 * 198 * 3000):
    b_0, r_0, _ = agent_results[0]
    b_1, r_1, _ = agent_results[1]
    _tmp = [0, 0]
    if b_0 == b_1 and r_0 == r_1:
        return [1, 1]
    else:
        _win = np.argmax([b_0 / (r_0 + EPS), b_1 / (r_1 + EPS)])
        _tmp[_win] = 1.0
    return _tmp


def update_elo(elo_list, i0, i1, res):
    if res[0] > 0:
        elo_list[i0], elo_list[i1] = elo.rate_1vs1(elo_list[i0], elo_list[i1])
    elif res[0] == 0.5:
        elo_list[i0], elo_list[i1] = elo.rate_1vs1(
            elo_list[i0], elo_list[i1], True)
    else:
        elo_list[i1], elo_list[i0] = elo.rate_1vs1(elo_list[i1], elo_list[i0])
    return elo_list


def update_elo_2(agent_list, elo_list, i0, i1, res):
    #print(agent_list, elo_list)
    if res[0] > 0:
        agent_list[i0], _ = elo.rate_1vs1(
            agent_list[i0], elo_list[i1])
    elif res[0] == 0.5:
        agent_list[i0], _ = elo.rate_1vs1(
            agent_list[i0], elo_list[i1], True)
    else:
        _, agent_list[i0] = elo.rate_1vs1(
            elo_list[i1], agent_list[i0])
    return agent_list


def basic_rules(agent_result):
    total_bitrate0, total_rebuffer0, total_smoothness0 = agent_result[0]
    total_bitrate1, total_rebuffer1, total_smoothness1 = agent_result[1]
    total_smoothness0 = total_smoothness0 / total_bitrate0
    total_smoothness1 = total_smoothness1 / total_bitrate1
    if total_rebuffer0 < total_rebuffer1:
        if total_bitrate0 > total_bitrate1:
            return [1, 0]
        elif total_bitrate0 == total_bitrate1:
            return [1, 0]
        else:
            _cof0 = total_rebuffer0 / total_bitrate0
            _cof1 = total_rebuffer1 / total_bitrate1
            if _cof0 > _cof1:
                return [0, 1]
            elif _cof0 == _cof1:
                tmp = [0,0]
                tmp[np.random.randint(2)] = 1
                return tmp
            else:
                return [1, 0]
    elif total_rebuffer0 == total_rebuffer1:
        if total_bitrate0 > total_bitrate1:
            return [1, 0]
        elif total_bitrate0 == total_bitrate1:
            tmp = [0,0]
            tmp[np.random.randint(2)] = 1
            return tmp
        else:
            return [0, 1]
    else:
        if total_bitrate0 > total_bitrate1:
            _cof0 = total_rebuffer0 / total_bitrate0
            _cof1 = total_rebuffer1 / total_bitrate1
            if _cof0 > _cof1:
                return [0, 1]
            elif _cof0 == _cof1:
                tmp = [0,0]
                tmp[np.random.randint(2)] = 1
                return tmp
            else:
                return [1, 0]
        elif total_bitrate0 == total_bitrate1:
            return [0, 1]
        else:
            return [0, 1]