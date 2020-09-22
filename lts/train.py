import multiprocessing as mp
import numpy as np
import logging
import os
import sys
from ltsenv import LTSEnv
import dualppo as network
import tensorflow as tf
import rules

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [8, 20]
A_DIM = 13
ACTOR_LR_RATE =1e-4
CRITIC_LR_RATE = 1e-3
NUM_AGENTS = 12
TRAIN_SEQ_LEN = 1440  # take as a train batch
TRAIN_EPOCH = 1000000
MODEL_SAVE_INTERVAL = 20
RANDOM_SEED = 42
SUMMARY_DIR = './results'
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './results/log'
BATTLE_ROUND = 16

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
    
NN_MODEL = None

def testing(epoch, pool, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.mkdir(TEST_LOG_FOLDER)
    
    # run test script
    os.system('python test.py ' + nn_model)
    f = open(TEST_LOG_FOLDER + 'log','r')
    reward, step = 0., 0.
    for p in f:
        sp = p.split(',')
        reward = float(sp[0])
        step = float(sp[1])
    f.close()
    log_file.write(str(reward) + ',' + str(step))
    log_file.write('\n')
    log_file.flush()

def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    
    with tf.Session() as sess, open('elo.txt', 'w') as test_log_file:

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM,
                learning_rate=ACTOR_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
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
                testing(epoch, None,
                    SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
                    test_log_file)

def agent(agent_id, net_params_queue, exp_queue):
    env = LTSEnv(agent_id)
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
                s_batch, a_batch, workload_batch, ratio_batch = [], [], [], []
                p_batch = []
                for step in range(TRAIN_SEQ_LEN):
                    s_batch.append(obs)
                    action_prob = actor.predict(
                        np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

                    noise = np.random.gumbel(size=len(action_prob))
                    act = np.argmax(np.log(action_prob) + noise)

                    obs, rew, done, info = env.step(act)

                    action_vec = np.zeros(A_DIM)
                    action_vec[act] = 1
                    a_batch.append(action_vec)
                    p_batch.append(action_prob)

                    workload_batch.append(info['workload'])
                    ratio_batch.append(0. - rew)
                    if done:
                        break
                tmp_buffer.append(
                    [s_batch, a_batch, p_batch, workload_batch, ratio_batch])
                    
            s, a, p, g = [], [], [], []
            for i in range(BATTLE_ROUND):
                w_arr = []
                for j in range(BATTLE_ROUND):
                    if i != j:
                        tmp_agent_results = []
                        # i
                        s_batch, a_batch, p_batch, workload_batch, ratio_batch = tmp_buffer[i]
                        bit_rate_ = np.mean(workload_batch)
                        rebuffer_ = np.mean(ratio_batch)
                        smoothness_ = np.mean(np.abs(np.diff(workload_batch)))
                        tmp_agent_results.append([bit_rate_, rebuffer_, smoothness_])
                        # j
                        s_batch, a_batch, p_batch, workload_batch, ratio_batch = tmp_buffer[j]
                        bit_rate_ = np.mean(workload_batch)
                        rebuffer_ = np.mean(ratio_batch)
                        smoothness_ = np.mean(np.abs(np.diff(workload_batch)))
                        tmp_agent_results.append([bit_rate_, rebuffer_, smoothness_])
                        # battle
                        w_rate_imm = rules.rules(tmp_agent_results)[0]
                        w_arr.append(w_rate_imm)
                w_rate = np.sum(w_arr) / len(w_arr)
                s_batch, a_batch, p_batch, workload_batch, ratio_batch = tmp_buffer[i]
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
