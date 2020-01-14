import multiprocessing as mp
import numpy as np
import logging
import os
import sys
import tensorflow as tf
import toynetwork as network
import gym

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [1, 4]
A_DIM = 2
ACTOR_LR_RATE =1e-4
CRITIC_LR_RATE = 1e-3
NUM_AGENTS = 4
TRAIN_SEQ_LEN = 300  # take as a train batch
TRAIN_EPOCH = 1000000
MODEL_SAVE_INTERVAL = 10
RANDOM_SEED = 42
RAND_RANGE = 10000
SUMMARY_DIR = './results'
MODEL_DIR = './models'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = './results/log'
BATCH_SIZE = 128

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)
    
NN_MODEL = None
EPS = 0.8

def pareto_rules(agent_results):
    r_0, step_0 = agent_results[0]
    r_1, step_1 = agent_results[1]
    _tmp = [0., 0.]
    if np.abs(step_0 - step_1) < 1e-6 and np.abs(r_0 - r_1) < 1e-6:
        _idx = np.random.randint(2)
        _tmp[_idx] = 1.
    else:
        _tmp[np.argmax([r_0, r_1])] = 1.
    return _tmp

def testing(epoch, nn_model, log_file):
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
    # os.system('python draw.py')    

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

            s, a, g, r = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, g_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                g += g_
                r.append(r_)
            actor.train(s, a, g)
            f = open('elo.txt', 'a')
            f.write(str(np.mean(r)))
            f.write('\n')
            f.flush()
            f.close()
            # if epoch % MODEL_SAVE_INTERVAL == 0:
            #     # Save the neural net parameters to disk.
            #     save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" +
            #                            str(epoch) + ".ckpt")
            #     testing(epoch,
            #         SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt", 
            #         test_log_file)

def agent(agent_id, net_params_queue, exp_queue):
    env = gym.make("CartPole-v0")
    env.force_mag = 100.0

    with tf.Session() as sess, open(SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        time_stamp = 0
        for epoch in range(TRAIN_EPOCH):
            # env.reset_trace()
            tmp_buffer = []
            tmp_agent_results = []
            for i in range(2):
                obs = env.reset()
                reward = 0.
                s_batch, a_batch, r_batch = [], [], []
                for step in range(TRAIN_SEQ_LEN):
                    obs = np.reshape(obs, (S_DIM[0], S_DIM[1]))
                    s_batch.append(obs)
                    action_prob = actor.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
                    action_cumsum = np.cumsum(action_prob)
                    action = (action_cumsum > np.random.randint(
                        1, RAND_RANGE) / float(RAND_RANGE)).argmax()
                    obs, rew, done, info = env.step(action)
                    
                    reward += rew
                    action_vec = np.zeros(A_DIM)
                    action_vec[action] = 1
                    a_batch.append(action_vec)

                    if done:
                        break
                tmp_buffer.append(
                    [s_batch, a_batch])
                tmp_agent_results.append([reward, step])
                r_batch.append(reward)

            g_ = pareto_rules(tmp_agent_results)
            s, a, g = [], [], []
            for t, p in zip(tmp_buffer, g_):
                s_batch, a_batch = t
                for s_, a_ in zip(s_batch, a_batch):
                    s.append(s_)
                    a.append(a_)
                    g.append([p])
            r = np.mean(r_batch)
            exp_queue.put([s, a, g, r])

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
