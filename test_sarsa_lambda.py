import numpy as np
from numpy import save
from numpy import load
import gym
from SARSA_Lambda.sarsa_lambda import SarsaLambda, StateActionFeatureVectorWithTile
import flappy_bird_gym
import time
import statistics as stats

# mode = "TEST"
# mode = "LEARN"
mode = "SINGLE_GAME"

def test_sarsa_lamda():

    env = flappy_bird_gym.make("FlappyBird-v0")
    gamma = 1
    # env.observation_space.high = [1.8, .8]
    # env.observation_space.low = [0, -.8]

    env.observation_space.high = [2, 2]
    env.observation_space.low = [-2, -2]


    X = StateActionFeatureVectorWithTile(
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        num_tilings=4,
        # tile_width=np.array([.1105,.00875]))
        # tile_width=np.array([.00875*2,.1105]))
        tile_width=np.array([.1, .1])
        # tile_width=np.array([.08, .08])
    )

    if mode == "LEARN":
        w = SarsaLambda(env, gamma, 0.75, 0.01, X, 2000)
        save('weights_lam_0_75.npy', w)
        # mode = "SINGLE_GAME"
    if mode == "TEST":
        w = load('weights_lam_0_7.npy')
    if mode == "SINGLE_GAME":
        w = load('weights_lam_0_7.npy')
    # w = SarsaLambda(env, gamma, 0.8, 0.01, X, 2000)
    # w = load('weights_new.npy')
    # w = load('weights_151.npy')

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

            # Rendering the game:
            # (remove this two lines during training)
            # env.render()
            # time.sleep(1 / 60)  # FPS
            
            # Checking if the player is still alive
            if done:
                # break
                avg.append(info['score'])
                obs = env.reset()
                ep += 1
                print(ep)
                if ep > 2000:
                    break
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

        # print(info['score'])
    print(np.mean(avg),max(avg), min(avg))
    env.close()

if __name__ == "__main__":
    print("Starting")
    test_sarsa_lamda()

    