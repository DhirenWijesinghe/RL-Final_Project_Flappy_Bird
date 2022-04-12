import numpy as np
import math

class Q_Table():

    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 tile_width:np.array):

        self.num_rows = math.ceil((state_high[0] - state_low[0]) / tile_width[0]) + 1
        self.num_cols = math.ceil((state_high[1] - state_low[1]) / tile_width[1]) + 1
        print(self.num_rows, self.num_cols)
        self.tile_width = tile_width

        self.Q_table = np.zeros((self.num_rows, self.num_cols, num_actions))

        self.state_low = state_low
        self.state_high = state_high

    def __call__(self, State=(-1, -1), Action=-1):
        
        if State[0] == -1:
            return self.Q_table
        
        else:
            horizontal = int((State[0] + abs(self.state_low[0])) // self.tile_width[0])
            vertical = int((State[1] + abs(self.state_low[1])) // self.tile_width[1])
            if Action == -1:
                return self.Q_table[horizontal, vertical]
            else:
                return self.Q_table[horizontal, vertical, Action]
    
    def update(self, State, Action, Q_value):

        horizontal = int((State[0] + abs(self.state_low[0])) // self.tile_width[0])
        vertical = int((State[1] + abs(self.state_low[1])) // self.tile_width[1])
        self.Q_table[horizontal, vertical, Action] += Q_value
        # print(self.Q_table[horizontal, vertical, Action])

    
def Q_Learning(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    num_actions:int,
    num_episode:int,
) -> np.array:

    Reward = {1 : 0, 
                -1000: -1000}

    def epsilon_greedy_policy(S,Q,epsilon=.1):
        nA = env.action_space.n
        # Q = [np.dot(w, X(s,done,a)) for a in range(nA)]
        # Q = np.dot(w, X(s,done,w))

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q(S))

    Q = Q_Table(state_low=env.observation_space.low, 
                state_high=env.observation_space.high, 
                num_actions=num_actions,
                tile_width=[0.001, 0.001])
    
    for e in range(num_episode):

        done = False

        S = env.reset()

        while not done:

            A = epsilon_greedy_policy(S,Q)
            S_prime, R, done, info = env.step(A)
            # R = Reward[R]
            # print(R)
            max_a = np.argmax(Q(State=S_prime))
            # print(max_a)
            Q_value = alpha * (R + gamma * Q(S_prime, max_a) - Q(S,A))
            # print(Q_value)
            Q.update(S, A, Q_value)
            S = S_prime
    return Q()

