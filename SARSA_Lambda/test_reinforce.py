import sys
import numpy as np
import gym
from matplotlib import pyplot as plt

from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN

import flappy_bird_gym
import time

import torch
from torch import save
from torch import load

def test_reinforce(with_baseline):
    env = flappy_bird_gym.make("FlappyBird-v0")
    gamma = 1.
    alpha = 3e-4

    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.reset_default_graph()

    pi = PiApproximationWithNN(
        env.observation_space.shape[0],
        env.action_space.n,
        alpha)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.shape[0],
            alpha)
    else:
        B = Baseline(0.)

    total_return = REINFORCE(env,gamma,1000,pi,B)

    print("About to test")

    obs = env.reset()

    print(pi(obs))
    while True:
        # Next action:
        # (feed the observation to your agent here)
        # action =  env.action_space.sample() #for a random action

        action = pi(obs)
        # print(action)

        # Processing:
        obs, reward, done, info = env.step(action)
        
        # Rendering the game:
        # (remove this two lines during training)
        env.render()
        time.sleep(1 / 30)  # FPS
        
        # Checking if the player is still alive
        if done:
            break

    env.close()

    return total_return


if __name__ == "__main__":
    num_iter = 5

    # Test REINFORCE without baseline

    # Comment start
    without_baseline = []
    for _ in range(num_iter):
        training_progress = test_reinforce(with_baseline=False)
        without_baseline.append(training_progress)
    without_baseline = np.mean(without_baseline,axis=0)

    # # Test REINFORCE with baseline
    # with_baseline = []
    # for _ in range(num_iter):
    #     training_progress = test_reinforce(with_baseline=True)
    #     with_baseline.append(training_progress)
    # with_baseline = np.mean(with_baseline,axis=0)

    # Plot the experiment result
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(without_baseline)),without_baseline, label='without baseline')
    # ax.plot(np.arange(len(with_baseline)),with_baseline, label='with baseline')

    ax.set_xlabel('iteration')
    ax.set_ylabel('G_0')
    ax.legend()

    plt.show()

    #Comment end
    

    # Note:
    # For CartPole-v0, without-baseline version could be more stable than with-baseline.
    # Can you guess why? How can you stabilize it? (This is not the part of the assignment, but just
    # the food for thought)

    # Note:
    # Due to stochasticity and unstable nature of REINFORCE, the algorithm might not stable and
    # doesn't converge. General rule of thumbs to check your algorithm works is that see whether
    # your algorithm hits the maximum possible return, in our case 200.
