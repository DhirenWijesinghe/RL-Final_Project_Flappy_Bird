import numpy as np

class Q_Table():
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

def reward_func(R, done, score, score_prev):
        if not done:
            if score > score_prev:
                return 1, score
            else:
                return 0, score_prev
        else:
            return -1000, score_prev

def Q_Learning(
    env, # openai gym environment
    gamma:float, # discount factor
    alpha:float, # step size
    num_actions:int,
    num_episodes:int,
    target_score: int,
    num_consecutive_scores: int,
) -> np.array:

    Q = Q_Table(num_actions)

    def epsilon_greedy_policy(s,epsilon=0.01):
        nA = env.action_space.n
        Q_vals = Q.get_q_at_pos(s)
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q_vals)

    mean_scores = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    episode = 0
    mean_index = -1*num_consecutive_scores
    while np.mean(mean_scores[mean_index:]) < target_score and episode < num_episodes:
        print("Episode:", episode, "Avg score: ", np.mean(mean_scores[mean_index:]), end='\r')
        done = False
        S = env.reset()
        score = 0
        score_prev = 0
        while not done:
            A = epsilon_greedy_policy(S)
            S_prime, R, done, info = env.step(A)
            score = info['score']
            R, score_prev = reward_func(R, done, score, score_prev)
            max_Q = max(Q.get_q_at_pos(S_prime, 0), Q.get_q_at_pos(S_prime, 1))
            current_Q_value = Q.get_q_at_pos(S,A)
            new_Q_value = (current_Q_value) + (alpha * (R + (gamma * max_Q) - current_Q_value))
            Q.set_q_value(S, A, new_Q_value)
            S = S_prime
        mean_scores.append(score)
        episode += 1
    return Q, np.mean(mean_scores[mean_index])