import numpy as np
import tensorflow as tf
import gym
import toynetwork as network
import sys


S_DIM = [1, 2]
A_DIM = 3
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RAND_RANGE = 1000
# NN_MODEL = './models/nn_model_eps_len_300.ckpt'
NN_MODEL = sys.argv[1]


def main():

    env = gym.make("MountainCar-v0")
    env._max_episode_steps = 10000
    env.seed(42)

    with tf.Session() as sess:
        actor = network.Network(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        saver = tf.train.Saver()
        saver.restore(sess, NN_MODEL)
    
        obs = env.reset()
        reward = 0.
        pos = 10000.
        for step in range(10000):
            
            action_prob = actor.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
            # action_cumsum = np.cumsum(action_prob)
            a = np.argmax(action_prob)
            #(action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            pos = np.minimum(pos, obs[0])
            obs, rew, done, info = env.step(a)

            reward += rew
            if done:
                break
        f = open('./test_results/log', 'w')
        f.write(str(reward) + ',' + str(pos))
        f.write('\n')
        f.close()


if __name__ == '__main__':
    main()
