from ast import arg
from operator import mod
import numpy as np
from numpy import save
from numpy import load
import gym
from SARSA_Lambda.sarsa_lambda import SarsaLambda, StateActionFeatureVectorWithTile
import flappy_bird_gym
import time
import statistics as stats
import sys

args = sys.argv
if len(args) <= 0:
    mode = "SINGLE_GAME"
elif args[1] == '-t':
    mode = "TEST"
elif args[1] == '-l':
    mode = "LEARN"
else:
    mode = "SINGLE_GAME"

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
        tile_width=np.array([.1, .1])
    )

    if mode == "LEARN":
        w = SarsaLambda(env, gamma, 0.7, 0.01, X, 2000)
        save('SARSA_Lambda/weights_new.npy', w)
    if mode == "TEST":
        w = load('SARSA_Lambda/weights_lam_0_7.npy')
    if mode == "SINGLE_GAME":
        w = load('SARSA_Lambda/weights_lam_0_7.npy')

    def greedy_policy(s,done):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    obs = env.reset()

    done = False

    if mode == "LEARN_HEAVY":
        while True:
            w = SarsaLambda(env, gamma, 0.8, 0.01, X, 2000)
            obs = env.reset()
            for i in range(10):
                action = greedy_policy(obs, done)
                # Processing:
                obs, reward, done, info = env.step(action)
                # env.render()
                # time.sleep(1 / 30)  # FPS
                
                # Checking if the player is still alive
                if done:
                    obs = env.reset()
                if info['score'] > 50:
                    break
            if(info['score'] > 50):
                break
    elif mode == "TEST":
        avg = []
        ep = 0
        while True:
            # Next action:
            action = greedy_policy(obs, done)
            # Processing:
            obs, reward, done, info = env.step(action)
            
            # Checking if the player is still alive
            if done:
                avg.append(info['score'])
                obs = env.reset()
                ep += 1
                if ep > 2000:
                    break
        print(np.mean(avg),max(avg), min(avg))

    elif mode == "SINGLE_GAME":
        while True:
            # Next action:
            action = greedy_policy(obs, done)
            # Processing:
            obs, reward, done, info = env.step(action)

            # Rendering the game:
            # (remove this two lines during training)
            env.render()
            time.sleep(1 / 60)  # FPS
            
            # Checking if the player is still alive
            if done:
                break
    env.close()

if __name__ == "__main__":
    test_sarsa_lamda()

    