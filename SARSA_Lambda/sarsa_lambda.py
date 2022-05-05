import numpy as np
# import cupy as np
import math
import pandas as pd


import torch

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        # TODO: implement here

        self.state_low = state_low 
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_actions = num_actions

        self.num_tiles_x = math.ceil((state_high[0] - state_low[0]) / tile_width[0]) + 1
        self.num_tiles_y = math.ceil((state_high[1] - state_low[1]) / tile_width[1]) + 1
        # print(state_high[0], state_low[0], tile_width[0])

        print(self.num_tiles_x, self.num_tiles_y)

        self.tiling_start_coords = []

        # Get start coordinates for tilings
        for tiling_idx in range(0, num_tilings, 1):
            # Compute start coordinates for tilings
            x_coord = state_low[0] - (tiling_idx / num_tilings) * tile_width[0]
            y_coord = state_low[1] - (tiling_idx / num_tilings) * tile_width[1]
            self.tiling_start_coords.append((x_coord, y_coord))

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        # TODO: implement this 
        return (self.num_actions * self.num_tilings * self.num_tiles_x * self.num_tiles_y)
        raise NotImplementedError()

    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        # TODO: implement this method
        feature_vector = np.zeros((self.feature_vector_len()))
        if done:
            return feature_vector
        else:

            tiling_idx = 0

            for tiling_idx in range(0, self.num_tilings, 1):
                # print(s[0],s[1])
                # calculate x tile
                x_tile = (s[0] + abs(self.tiling_start_coords[tiling_idx][0])) // self.tile_width[0]

                # calculate y tile
                y_tile = (s[1] + abs(self.tiling_start_coords[tiling_idx][1])) // self.tile_width[1]

                # Encode feature vector
                vec_idx = np.ravel_multi_index((a, tiling_idx, int(x_tile), int(y_tile)), (self.num_actions, self.num_tilings, self.num_tiles_x, self.num_tiles_y))
                # vec_idx = a + tiling_idx * self.num_actions + int(x_tile) * self.num_actions * self.num_tilings + int(y_tile) * self.num_actions * self.num_tilings * self.num_tiles_x
                
                feature_vector[int(vec_idx)] = 1

        return feature_vector
        raise NotImplementedError()

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=0.01):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]
        # Q = np.dot(w, X(s,done,w))
        # print(Q)
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            # if(Q[0] == Q[1]):
            #     return np.random.randint(nA)
            return np.argmax(Q)

    def reward_func(R, done, score, score_prev):
        if not done:
            if score > score_prev:
                return 1, score
            else:
                return 0, score_prev
        else:
            return -10, score_prev
    
    # def reward_func(R,done):
    #     if not done:
    #         return 1
    #     else:
    #         return -10
    
    # def reward_func(R):
    #     if R == 1:
    #         return 1
    #     else:
    #         return -1000

    # def reward_func(score, score_prev):
    #     if score > score_prev:
    #         return 5, score
    #     else:
    #         return 1, score_prev

    w = np.zeros((X.feature_vector_len()))
    # w = np.load('weights 70.npy')
    mean_scores = [0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    episode = 0
    while np.mean(mean_scores[-3:]) < 20 and episode < 10000:
    # for episode in range(0, num_episode, 1):

        print("Episode:", episode, "Avg score: ", np.mean(mean_scores[-2:]), end='\r')

        done = False

        S = env.reset()

        A = epsilon_greedy_policy(S,done, w)

        x = X(S,done,A)
        # print(x.shape)

        z = np.zeros((X.feature_vector_len()))

        Q_old = 0

        # step = 0
        score_prev = 0
        while not done:
            # print(step)
            S_prime, R, done, info = env.step(A)
            score = info['score']
            R, score_prev = reward_func(R, done, score, score_prev)
            # R = reward_func(R)
            # R, score_prev = reward_func(score, score_prev)
            A_prime = epsilon_greedy_policy(S_prime, done, w)
            x_prime = X(S_prime, done, A_prime)
            Q = np.dot(x, w)
            Q_prime = np.dot(x_prime, w)
            delta = R + (gamma * Q_prime) - Q
            z = (gamma * lam * z) + ((1 - ((alpha*gamma*lam) * np.dot(x,z))) * x )
            w = w + alpha * (delta + Q - Q_old) * z - (alpha * (Q - Q_old) * x)
            Q_old = Q_prime
            x = x_prime
            A = A_prime
            # S = S_prime
            # Break when S' terminal
        mean_scores.append(score)
        episode += 1
    dataframe = pd.DataFrame(mean_scores) 
    dataframe.to_csv('scores.csv')
    return w
