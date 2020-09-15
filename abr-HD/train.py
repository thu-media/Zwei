import multiprocessing as mp
import numpy as np
import logging
import os
import sys
from abr import ABREnv
# import network_ppo_naive as network
import lsac as network
import tensorflow as tf
import rules
import tracepool
import itertools

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE =1e-4
NUM_AGENTS = 12
TRAIN_SEQ_LEN = 300  # take as a train batch
TRAIN_EPOCH = 1000000
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './lsac-1e-3'
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './lsac-1e-3/log'
BATTLE_ROUND = 16

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
    
NN_MODEL = None
# './results/nn_model_ep_18700.ckpt'

def testing(epoch, pool, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.mkdir(TEST_LOG_FOLDER)
    
    # run test script
    os.system('python rl_test.py ' + nn_model)
    agent_elo = []
    agent_list = []
    for p in range(1):
        agent_list.append(str(p))
        agent_elo.append(1000.0)
    _trace_result = []
    entropies = []
    print(epoch, 'start testing...')
    for _trace in pool.get_test_set():
        agent_result = []
        entropy = []
        for _agent in agent_list:
            _f = open('./' + TEST_LOG_FOLDER + 'log_sim_zwei_' + _trace, 'r')
            _bitrate, _rebuffer = [], []
            for lines in _f:
                #110.64486915972032	2850	19.235901151929067	0.0	1341201	5257.885326692943	0.5	2.85
                sp = lines.split()
                if(len(sp) > 0):
                    entropy.append(float(sp[-2]))
                    _bitrate.append(float(sp[1]))
                    _rebuffer.append(float(sp[3]))
            _bitrate_mean = np.mean(_bitrate[1:])
            _rebuffer_mean = np.mean(_rebuffer[1:])
            _smo_mean = np.mean(np.abs(np.diff(_bitrate[1:])))
            
            agent_result.append(
                (_bitrate_mean, _rebuffer_mean, _smo_mean))
            _trace_result.append(agent_result)
            _f.close()
            entropies.append(np.mean(entropy[1:]))
    
    _rate, agent_elo = pool.battle(agent_elo, _trace_result)
    for p in agent_elo:
        print(p)
        log_file.write(str(p))
    log_file.write('\n')
    log_file.flush()
    os.system('python draw.py')
    return agent_elo[0], np.mean(entropies)

def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("Beta", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", eps_total_reward)
    entropy = tf.Variable(0.)
    tf.summary.scalar("Entropy", entropy)

    summary_vars = [td_loss, eps_total_reward, entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    pool = tracepool.tracepool()
    with tf.Session() as sess, open('elo.txt', 'w') as test_log_file:
        summary_ops, summary_vars = build_summaries()

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM,
                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # training monitor
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        max_reward, max_epoch = -10000., 0
        tick_gap = 0
        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(1, TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                g += g_

            for _ in range(actor.training_epo):
                actor.train(s, a, p, g)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                avg_reward, avg_entropy = testing(epoch, pool, 
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)
                
                if avg_reward > max_reward:
                    max_reward = avg_reward
                    max_epoch = epoch
                    tick_gap = 0
                else:
                    tick_gap += 1
                
                if tick_gap >= 5:
                    # saver.restore(sess, SUMMARY_DIR + "/nn_model_ep_" + str(max_epoch) + ".ckpt")
                    actor.set_entropy_decay()
                    tick_gap = 0

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: actor.get_entropy(epoch),
                    summary_vars[1]: avg_reward,
                    summary_vars[2]: avg_entropy
                })
                writer.add_summary(summary_str, epoch)
                writer.flush()

def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        time_stamp = 0
        obs = env.reset()
        # env.reset()
        for epoch in range(TRAIN_EPOCH):
            env.reset_trace()
            tmp_buffer = []
            for i in range(BATTLE_ROUND):
                obs = env.reset()
                s_batch, a_batch, p_batch, bitrate_batch, rebuffer_batch = [], [], [], [], []
                for step in range(TRAIN_SEQ_LEN):
                    s_batch.append(obs)
                    action_prob = actor.predict(
                        np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
                    
                    action_cumsum = np.cumsum(action_prob)
                    bit_rate = (action_cumsum > np.random.randint(
                        1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                    obs, rew, done, info = env.step(bit_rate)

                    action_vec = np.zeros(A_DIM)
                    action_vec[bit_rate] = 1
                    a_batch.append(action_vec)
                    p_batch.append(action_prob)

                    bitrate_batch.append(info['bitrate'])
                    rebuffer_batch.append(info['rebuffer'])
                    if done:
                        break
                tmp_buffer.append(
                    [s_batch, a_batch, p_batch, bitrate_batch, rebuffer_batch])
            s, a, p, g = [], [], [], []
            for i in range(BATTLE_ROUND):
                w_arr = []
                for j in range(BATTLE_ROUND):
                    if i != j:
                        tmp_agent_results = []
                        # i
                        s_batch, a_batch, p_batch, bitrate_batch, rebuffer_batch = tmp_buffer[i]
                        bit_rate_ = np.mean(bitrate_batch)
                        rebuffer_ = np.mean(rebuffer_batch)
                        smoothness_ = np.mean(np.abs(np.diff(bitrate_batch)))
                        tmp_agent_results.append([bit_rate_, rebuffer_, smoothness_])
                        # j
                        s_batch, a_batch, p_batch, bitrate_batch, rebuffer_batch = tmp_buffer[j]
                        bit_rate_ = np.mean(bitrate_batch)
                        rebuffer_ = np.mean(rebuffer_batch)
                        smoothness_ = np.mean(np.abs(np.diff(bitrate_batch)))
                        tmp_agent_results.append([bit_rate_, rebuffer_, smoothness_])
                        # battle
                        w_rate_imm = rules.rules(tmp_agent_results)[0]
                        w_arr.append(w_rate_imm)
                w_rate = np.sum(w_arr) / len(w_arr)
                s_batch, a_batch, p_batch, bitrate_batch, rebuffer_batch = tmp_buffer[i]
                # Policy invariance under reward 
                for s_, a_, p_ in zip(s_batch, a_batch, p_batch):
                    s.append(s_)
                    a.append(a_)
                    p.append(p_)
                    g.append([w_rate])
            exp_queue.put([s, a, p, g])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)

def compute_entropy(x):
    """
    Given vector x, computes the entropy
    H(x) = - sum( p * log(p))
    """
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H

def main():

    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
