import numpy as np
from numpy import save
from numpy import load
import gym
from sarsa import SarsaLambda, StateActionFeatureVectorWithTile
import flappy_bird_gym
import time

def test_sarsa_lamda():
    env = flappy_bird_gym.make("FlappyBird-v0")
    gamma = 1

    env.observation_space.high = [2, 2]
    env.observation_space.low = [-2, -2]
    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=4,
        # tile_width=np.array([.1105,.00875]))
        # tile_width=np.array([.00875*2,.1105]))
        tile_width=np.array([.1, .1]))
        # tile_width=np.array([.0451,.0351]))

    w = SarsaLambda(env, gamma, 0.8, 0.01, X, 2000)
    # w = load('weights_new.npy')
    # w = load('weights.npy')
    # save('weights_new.npy', w)

    def greedy_policy(s,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    obs = env.reset()

    done = False
    while True:
        # Next action:
        action = greedy_policy(obs, done)
        # Processing:
        obs, reward, done, info = env.step(action)
        
        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS
        
        # Checking if the player is still alive
        if done:
            print(info['score'])
            break

    env.close()

if __name__ == "__main__":
    test_sarsa_lamda()

    