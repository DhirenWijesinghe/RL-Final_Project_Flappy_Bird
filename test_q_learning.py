import numpy as np
from numpy import save
from numpy import load
import gym
from q_learning import Q_Learning, Q_Table
import flappy_bird_gym
import time

def test_q_learning():

    TEST_MODE = True

    env = flappy_bird_gym.make("FlappyBird-v0")
    gamma = 1
    # env.observation_space.high = [1.8, .8]
    # env.observation_space.low = [0, -.8]

    env.observation_space.high = [2, 2]
    env.observation_space.low = [-2, -2]

    num_actions = env.action_space.n
    # pi = OptimalPolicy(num_actions)
    target_score = 3

    if not TEST_MODE:
        while True:
            pi = Q_Table(num_actions)
            # pi, avg_score = n_step_sarsa(env = env, n = 1, alpha = 0.005, gamma = 1, num_episodes = 3000, pi = pi, target_score = target_score, num_consecutive_scores = 3)
            pi, avg_score = Q_Learning(env = env, gamma = 1, alpha = 0.005, num_actions = num_actions, num_episode = 3000)
            if avg_score >= target_score:
                break

        save('q_learning_q_table.npy', pi.get_q())
    
    if TEST_MODE:
        pi = Q_Table(num_actions)
        q = load('q_learning_q_table.npy')
        # q = load('sarsa_q_table_n_2_target_25.npy')
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

    if not TEST_MODE:
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
    env.close()

if __name__ == "__main__":
    test_q_learning()

    