from typing import Iterable, Tuple

import numpy as np
import pandas as pd

class OptimalPolicy():
    def __init__(self,nA):
        self.indices_x = {}
        self.indices_y = {}
        index = 0
        for i in np.arange(-2, 2.01, 0.01):
            new_i = round(i, 2)
            self.indices_x[new_i] = index
            self.indices_y[new_i] = index
            index += 1
        self.nS = index
        self.q = np.empty((self.nS * self.nS, nA))

    def state_to_index(self, state):
        x_index = self.indices_x[round(state[0], 2)]
        y_index = self.indices_y[round(state[1], 2)]
        index = (x_index * self.nS) + y_index
        return index

    def action_prob(self,state,action=None):
        index = self.state_to_index(state)
        return action == np.argmax(self.q[index])

    def action(self,state):
        index = self.state_to_index(state)
        return np.argmax(self.q[index])

    def set_q_value(self, state, action, val):
        index = self.state_to_index(state)
        self.q[index, action] = val
    
    def get_q(self):
        return self.q
    
    def get_q_at_pos(self, state, action=-1):
        index = self.state_to_index(state)
        if action == -1:
            return self.q[index]
        return self.q[index, action]
    
    def set_q_table(self, q_table):
        self.q = q_table

def n_step_sarsa(
    env,
    n:int,
    alpha:float,
    gamma: float,
    num_episodes: int,
    pi: OptimalPolicy,
    target_score: int,
    num_consecutive_scores: int
) -> Tuple[np.array]:
    """
    input:
        env_spec: environment spec
        trajs: N trajectories generated using
            list in which each element is a tuple representing (s_t,a_t,r_{t+1},s_{t+1})
        n: how many steps?
        alpha: learning rate
        initV: initial V values; np array shape of [nS]
    ret:
        V: $v_pi$ function; numpy array shape of [nS]
    """

    def epsilon_greedy_policy(s,epsilon=0.001):
        nA = env.action_space.n
        Q = pi.get_q_at_pos(s)
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)

    def reward_func(R, done, score, score_prev):
        if not done:
            if score > score_prev:
                return 1, score
            else:
                return 0, score_prev
        else:
            return -10, score_prev

    tau_val = 0
    episode = 0
    mean_scores = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    while np.mean(mean_scores[-1*num_consecutive_scores:]) < target_score and episode < num_episodes:
        print("Episode:", episode, "Avg score: ", np.mean(mean_scores[-1*num_consecutive_scores:]), end='\r')
        T = np.Infinity
        t = 0
        state = env.reset()
        action = epsilon_greedy_policy(state)
        memory = [(state, action, 0)]
        score_prev = 0
        while tau_val != T-1:
            if t < T:
                action = memory[t][1]
                s_next, reward_next, done_next, info = env.step(action)
                score = info['score']
                reward_next, score_prev = reward_func(reward_next, done_next, score, score_prev)
                if done_next:
                    T = t+1
                    action_next = -1
                else:
                    action_next = epsilon_greedy_policy(s_next)
                memory.append((s_next, action_next, reward_next))
            tau_val = t-n+1
            if tau_val >= 0:
                G = 0
                for i in range(tau_val+1, min(tau_val+n, T)+1):
                    G += (gamma ** (i-tau_val-1)) * memory[i][2]
                if tau_val + n < T:
                    G += (gamma ** n) * pi.get_q_at_pos(memory[tau_val+n][0], memory[tau_val+n][1])
                current_q = pi.get_q_at_pos(memory[tau_val][0], memory[tau_val][1])
                new_q = current_q + (alpha * (G-current_q))
                # if tau_val > 0 and memory[tau_val-1][1] > memory[tau_val][1]:
                #     new_q += 0.001
                pi.set_q_value(memory[tau_val][0], memory[tau_val][1], new_q)
            t += 1
        mean_scores.append(score)
        episode += 1
    dataframe = pd.DataFrame(mean_scores) 
    dataframe.to_csv('sarsa_scores.csv')
    return pi, np.mean(mean_scores[-1*num_consecutive_scores:])