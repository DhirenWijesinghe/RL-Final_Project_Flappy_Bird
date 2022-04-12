from cmath import inf
import time
import flappy_bird_gym
from q_learning import *
env = flappy_bird_gym.make("FlappyBird-v0")

obs = env.reset()

# while True:
#     # Next action:
#     # (feed the observation to your agent here)
#     action =  env.action_space.sample() #for a random action

#     # Processing:
#     obs, reward, done, info = env.step(action)
    
#     # Rendering the game:
#     # (remove this two lines during training)
#     env.render()
#     time.sleep(1 / 30)  # FPS
    
#     # Checking if the player is still alive
#     if done:
#         break

num_actions = env.action_space.n
# num_states = 2*2*100 + 2*100
# print(num_states)
env.observation_space.high = [2, 2]
env.observation_space.low = [-2, -2]
print(env.observation_space.high, env.observation_space.low)

result = Q_Learning(env=env,gamma=0,lam=0,alpha=0.7,num_actions=num_actions,num_episode=10000)

env.close()

env.observation_space.high = [2, 2]
env.observation_space.low = [-2, -2]
obs = env.reset()


while True:
    # Next action:
    # (feed the observation to your agent here)
    # action =  env.action_space.sample() #for a random action
    s0 = int((obs[0] + abs(env.observation_space.low[0])) // 0.001)
    s1 = int((obs[1] + abs(env.observation_space.low[1])) // 0.001)
    action = np.argmax(result[s0,s1])
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