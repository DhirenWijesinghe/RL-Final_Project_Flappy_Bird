import numpy as np

class Q_Table():
    def __init__(self,nA):
        self.indices_x = {}
        self.indices_y = {}
        index = 0
        for i in np.arange(-2, 2.001, 0.001):
            new_i = round(i, 3)
            self.indices_x[new_i] = index
            self.indices_y[new_i] = index
            index += 1
        self.nS = index
        self.q = np.empty((self.nS * self.nS, nA))

    def state_to_index(self, state):
        x_index = self.indices_x[round(state[0], 3)]
        y_index = self.indices_y[round(state[1], 3)]
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

# class Q_Table():

#     def __init__(self):
#         self.Q = {}
#         # self.num_visits = {}
       
#     def __call__(self, state, action=-1):
#         s = self.convert_state(state)
#         if action == -1:
#             return self.Q[s]
#         else:
#             return self.Q[s][action]
    
#     def update(self, state, action, Q_value):
#         s = self.convert_state(state)
#         self.Q[s][action] = Q_value
    
#     def add_state(self, state):
#         s = self.convert_state(state)
#         if self.Q.get(s) == None:
#             self.Q[s] = [0, 0] # Q for no flap (action 0), Q for flap (action 1)

#     def convert_state(self, state):
#         return (round(state[0], 3), round(state[1], 3))

#     def get_table(self):
#         return self.Q

    # def increment_num_visits(self, state, action):
    #     s = self.convert_state(state)
    #     if self.num_visits.get((s, action)) == None:
    #         self.num_visits[(s, 0)] = 1
    #         self.num_visits[(s, 1)] = 1
    #     else:
    #         self.num_visits[(s, 0)] += 1
    #         self.num_visits[(s, 1)] += 1

    # def get_num_visits(self, state, action):
    #     s = self.convert_state(state)
    #     return self.num_visits[(s, action)]

# def reward_func(R, done, score, score_prev):
#     # Pass a pipe (ie. increment score) = 1
#     # Collide = -1000
#     # Any other state of being alive = 0.1
#     if not done:
#         if score > score_prev:
#             R = 1 # Passed pipe
#         else:
#             R = 0
#     else:
#         # Crashed
#         R = -100
#     return R

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
    num_episode:int,
) -> np.array:

    Q = Q_Table(num_actions)

    def epsilon_greedy_policy(s,epsilon=0.1):
        nA = env.action_space.n
        Q_vals = Q.get_q_at_pos(s)
        # Q = np.dot(w, X(s,done,w))
        # print(Q)
        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            # if(Q[0] == Q[1]):
            #     return np.random.randint(nA)
            return np.argmax(Q_vals)

    mean_scores = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    episode = 0
    while np.mean(mean_scores[-1:]) < 3 and episode < 30000:
        print("Episode:", episode, "Avg score: ", np.mean(mean_scores[-1:]), end='\r')
        done = False
        S = env.reset()
        score = 0
        score_prev = 0
        while not done:
            A = epsilon_greedy_policy(S)
            S_prime, R, done, info = env.step(A) # R is always 1, so it's irrelevant
            score = info['score']
            R, score_prev = reward_func(R, done, score, score_prev)
            max_Q = max(Q.get_q_at_pos(S_prime, 0), Q.get_q_at_pos(S_prime, 1))
            current_Q_value = Q.get_q_at_pos(S,A)
            new_Q_value = (current_Q_value) + (alpha * (R + (gamma * max_Q) - current_Q_value))
            Q.set_q_value(S, A, new_Q_value)
            S = S_prime
        mean_scores.append(score)
        episode += 1
    # print("AVG SCORE: " + str(np.mean(scores)))
    return Q, np.mean(mean_scores[-1:])