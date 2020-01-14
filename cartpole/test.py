import numpy as np
import tensorflow as tf
import gym
import toynetwork as network
import sys


S_DIM = [1, 4]
A_DIM = 2
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
RAND_RANGE = 1000
# NN_MODEL = './models/nn_model_eps_len_300.ckpt'
NN_MODEL = sys.argv[1]


def main():

    env = gym.make("CartPole-v0")
    # env.force_mag = 100.0

    with tf.Session() as sess:
        actor = network.Network(sess, state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)
        saver = tf.train.Saver()
        saver.restore(sess, NN_MODEL)
    
        obs = env.reset()
        reward = 0.
        for step in range(300):

            action_prob = actor.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))
            # action_cumsum = np.cumsum(action_prob)
            a = np.argmax(action_prob)
            #(action_cumsum > np.random.randint(1, RAND_RANGE) / float(RAND_RANGE)).argmax()

            obs, rew, done, info = env.step(a)

            reward += rew
            if done:
                break
        f = open('./test_results/log', 'w')
        f.write(str(reward) + ',' + str(step))
        f.write('\n')
        f.close()


if __name__ == '__main__':
    main()