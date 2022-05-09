# A Comparison of Reinforcement Learning Agents to Reach High Scores in Flappy Bird
Written by Uma Sethuraman and Dhiren Wijesinghe
## Python Environment Setup:
1. We highly recommend using the Anaconda environment included in the main directory of this project
2. If you do not use the anaconda environment, libraries that need to be installed are:
    * numpy
    * Flappy Bird Gym Environment: https://github.com/Talendar/flappy-bird-gym
    * pandas

## Running Code

### SARSA Lambda
* To run a single game with our best performing SARSA Lambda agent, run: `python test_sarsa_lambda.py -g`
* To run a test of 2000 games on our best performing SARSA Lambda agent, run: `python test_sarsa_lambda.py -t`
* To train your own SARSA Lambda agent, run: `python test_sarsa_lambda.py -l`

### N-Step SARSA
* To run a single game with our best performing n-step SARSA agent, run: `python test_n_step_sarsa.py -g`
* To run a test of 2000 games on our best performing n-step SARSA agent, run: `python test_n_step_sarsa.py -t`
* To train your own n-step SARSA agent, run: `python test_n_step_sarsa.py -l`

### Q-Learning
* To run a single game with our best performing Q-Learning agent, run: `python test_q_learning.py -g`
* To run a test of 2000 games on our best performing Q-Learning agent, run: `python test_q_learning.py -t`
* To train your own Q-Learning agent, run: `python test_q_learning.py -l`

## Changing Learning Parameters
You must open the corresponding agent's test file, and change the parameters on the lines where the agent class is instantiated. 
