How to Run DQN CartPole Code
=============================

Requirements:
-------------
- Python 3.8+
- PyTorch
- gymnasium

Install dependencies:
---------------------
pip install torch gymnasium

Running the training:
---------------------
python3 dqn_cartpole_homework.py

After training, the code will run a 1000-step test using the trained policy
and print how many times the pole was reset (should be <= 30 for full credit).


Neural Network (DQN) : A simple 3-layer fully connected neural network.
Input size: 4 (state space) and output size: 2 (Q-values for each action) and activation: ReLU.
Expected: Over time, the model learns to output higher Q-values for good actions.
-------------------------
Epsilon-Greedy Action Selection
If the probability x, the agent explores (chooses a random action).
If the probability 1-x, it exploits (chooses the best Q-value action).
Expected: initially more exploration, then, later mostly greedy actions as the agent learns.
-------------
Plotting Function: plots how many steps the agent survived in each episode.
Also shows a 100 episode moving average to smooth out the trend.
Expected: increasing trend in duration as learning progresses.




Run code:
------------
python3 dqn_cartpole_homework.py 

Testing ended after 1000 steps with 2 resets.

Note: I tried 64, 128, 256 for neural network width. The only good results was 4-128-128-2.
I tried many number of episoded num_episodes starting from 50, 100, 1000, 500, and good results was with 600.


