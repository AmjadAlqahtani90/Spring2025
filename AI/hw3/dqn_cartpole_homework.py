# Import all necessary modules  and library 

import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#Initialize the Environment and make an Initial Random Step
env = gym.make('CartPole-v1')

# set up matplotlib
plt.ion()


observation, info = env.reset(seed=42)
print(f"observation: {observation}")
print(f"info: {info}")

action = env.action_space.sample() 
print(f"action: {action}")
observation, reward, terminated, truncated, info = env.step(action)
print(f"observation: {observation}, reward: {reward}, terminated: {terminated}, truncated: {truncated}, info: {info}")

#Constants
BATCH_SIZE = 128    #Numbers of Samples fed into the nerual network during trainig at once
GAMMA = 0.99        #The decaying factor in the bellman function. Still remember the accumulated *discounted* return?
TAU = 0.005         #Update rate of the duplicate network
LR = 1e-4           #Learning rate of your Q - network

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
print(state)
n_observations = len(state)

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Replay Memory

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        #TODO : Define the layer of your Q network here. Think about the shapes when you define it.
        #What is the shape of the input? What should be the shape of the output?
        # ---- ---- ----
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, n_actions)
        # ---- ---- ----

    def forward(self, x):
        #TODO : Define how the network should process your input to produce an output
        #Should return a tensor
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.output(x)

def select_action(state):
    #TODO : Implement an epsilon-greedy policy that
    #Picks an random action with a small posibility
    #and acts according to the Q values otherwise
    
    # ---- ---- ---- to the end
    global steps_done
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000

    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_durations = []
steps_done = 0

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode Number (Training Iteration)')
    plt.ylabel('Episode Duration (Steps Survived)')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    #TODO : Calcualte the observed the Q value (Q_observed = immediate reward + gamma * max(Q(s_t+1)))
    #TODO : Pick an appropiate loss function and calculate the loss
    #TODO : Name your calculated loss "loss"
    
    # ---- ---- ----
    # Compute Q_observed = r + gamma * max(Q(s_{t+1}, a)) using target_net
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = reward_batch + (GAMMA * next_state_values)

    # Compute loss using Huber loss (SmoothL1Loss)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)
    # ---- ---- ----

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

#Creating to instances of the Q-network.
#Policy net is trained online directly by loss function
#Target network updates slower and provides a more stable target
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 600

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0
    num_steps = 0
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    if i_episode % 10 == 0:
        print(i_episode)

env.close()

TestEnv = gym.make("CartPole-v1", render_mode="human")
observation, info = TestEnv.reset(seed=42)

end_count = 0
steps = 0
test_durations = []
test_step = 0

while steps < 1000 and end_count < 30:
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = policy_net(state).argmax(1).view(1, 1)  # greedy policy

    observation, reward, terminated, truncated, _ = TestEnv.step(action.item())
    steps += 1
    test_step += 1

    if terminated or truncated:
        end_count += 1
        test_durations.append(test_step)  # store duration for this episode
        test_step = 0  # reset episode step counter
        if end_count < 30:
            observation, info = TestEnv.reset()
        else:
            break

TestEnv.close()
print(f"Testing ended after {steps} steps with {end_count} resets.")


plot_durations(show_result=True)
plt.ioff()
plt.show()


# Plot testing results
plt.figure()
plt.title("Testing Performance (Max 30 Episodes / 1000 Steps)")
plt.xlabel("Episode")
plt.ylabel("Duration")
plt.plot(test_durations, marker='o')
plt.grid(True)
plt.show()


