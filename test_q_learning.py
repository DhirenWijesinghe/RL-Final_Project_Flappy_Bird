import numpy as np
from numpy import save
from numpy import load
import gym
from Q_Learning.q_learning import Q_Learning, Q_Table
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

def test_q_learning():

    env = flappy_bird_gym.make("FlappyBird-v0")

    env.observation_space.high = [2, 2]
    env.observation_space.low = [-2, -2]

    num_actions = env.action_space.n
    target_score = 5

    if mode == "LEARN":
        while True:
            pi = Q_Table(num_actions)
            pi, avg_score = Q_Learning(env = env, gamma = 1, alpha = 0.5, num_actions = num_actions, num_episodes = 3000, target_score = target_score, num_consecutive_scores = 2)
            if avg_score >= target_score:
                break

        save('Q_Learning/q_learning_q_table_new.npy', pi.get_q())

        obs = env.reset()
        done = False

        while True:
            # Next action:
            action = pi.action(obs)
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
        pi = Q_Table(num_actions)
        q = load('Q_Learning/q_learning_best_q_table.npy')
        pi.set_q_table(q)

        scores = []
        for episode in range(2000):
            obs = env.reset()
            while True:
                # Next action:
                action = pi.action(obs)
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
        pi = Q_Table(num_actions)
        q = load('Q_Learning/q_learning_best_q_table.npy')
        pi.set_q_table(q)

        scores = []
        for episode in range(1):
            obs = env.reset()
            while True:
                # Next action:
                action = pi.action(obs)
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

if __name__ == "__main__":
    test_q_learning()

    