import numpy as np
from numpy import save
from numpy import load
import gym
from N_Step_Sarsa.n_step_sarsa import n_step_sarsa, OptimalPolicy
import flappy_bird_gym
import time
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

def test_n_step_sarsa():

    TEST_MODE = True

    env = flappy_bird_gym.make("FlappyBird-v0")
    gamma = 1
    # env.observation_space.high = [1.8, .8]
    # env.observation_space.low = [0, -.8]

    env.observation_space.high = [2, 2]
    env.observation_space.low = [-2, -2]

    num_actions = env.action_space.n
    # pi = OptimalPolicy(num_actions)
    target_score = 25

    if mode == "LEARN":
        while True:
            pi = OptimalPolicy(num_actions)
            pi, avg_score = n_step_sarsa(env = env, n = 2, alpha = 0.005, gamma = 1, num_episodes = 3000, pi = pi, target_score = target_score, num_consecutive_scores = 1)

            if avg_score >= target_score:
                break

        save('N_Step_Sarsa/sarsa_q_table_new.npy', pi.get_q())

        obs = env.reset()
        done = False

        while True:
            # Next action:
            action = pi.action(obs)
            # print(action)
            # Processing:
            obs, reward, done, info = env.step(action)

            # Rendering the game:
            # (remove this two lines during training)
            env.render()
            time.sleep(1 / 60)  # FPS
            
            # Checking if the player is still alive
            if done:
                break

        print(info['score'])

    elif mode == "TEST":
        pi = OptimalPolicy(num_actions)
        q = load('N_Step_Sarsa/sarsa_q_table_best_agent.npy')
        pi.set_q_table(q)

        scores = []
        for episode in range(2000):
            obs = env.reset()
            while True:
                # Next action:
                action = pi.action(obs)
                # print(action)
                # Processing:
                obs, reward, done, info = env.step(action)

                # Rendering the game:
                # (remove this two lines during training)
                # env.render()
                # time.sleep(1 / 60)  # FPS
                
                # Checking if the player is still alive
                if done:
                    break
            scores.append(info["score"])
        
        print("Avg Score: " + str(np.mean(scores)))
        print("Max Score: " + str(np.max(scores)))
    
    elif mode == "SINGLE_GAME":
        pi = OptimalPolicy(num_actions)
        q = load('N_Step_Sarsa/sarsa_q_table_best_agent.npy')
        pi.set_q_table(q)

        scores = []
        for episode in range(1):
            obs = env.reset()
            while True:
                # Next action:
                action = pi.action(obs)
                # print(action)
                # Processing:
                obs, reward, done, info = env.step(action)

                # Rendering the game:
                # (remove this two lines during training)
                env.render()
                time.sleep(1 / 60)  # FPS
                
                # Checking if the player is still alive
                if done:
                    break
            scores.append(info["score"])
        
        print("Score: " + str(np.mean(scores)))

    env.close()
    env.close()

if __name__ == "__main__":
    test_n_step_sarsa()

    