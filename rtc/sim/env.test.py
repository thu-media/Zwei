import env
import numpy as np

netenv = env.NetworkEnv(0)
netenv.reset_trace()
for p in range(2):
    obs = netenv.reset()
    rew_ = []
    for i in range(10):
        obs, rew, done, info = netenv.step(4)
        rew_.append(rew)
    print(np.mean(rew_))