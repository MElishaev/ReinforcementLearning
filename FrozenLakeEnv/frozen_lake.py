import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

n_games = 1000
win_pct = []
win_pct_interval = 10
scores = []

for i in range(n_games):
    done = False
    obs = env.reset()
    score = 0
    while not done:
        action = env.action_space.sample() # gets random action from action space
        obs, reward, done, truncated, info = env.step(action)
        score += reward
    scores.append(score)

    if i % win_pct_interval == 0:
        avg = np.mean(scores[-10:])
        win_pct.append(avg)

plt.plot(win_pct)
plt.show()
