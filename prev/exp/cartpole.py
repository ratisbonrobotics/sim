import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_contr import suite

# Define the neural network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

# Load the environment
env = suite.load(domain_name="cartpole", task_name="swingup", visualize_reward=True)

# Define the hyperparameters
learning_rate = 0.01
num_episodes = 1000
discount_factor = 0.99

# Create the policy network
state_size = env.observation_spec()['position'].shape[0] + env.observation_spec()['velocity'].shape[0]
action_size = env.action_spec().shape[0]
policy_net = PolicyNetwork(state_size, action_size)

# Define the optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Training loop
for episode in range(num_episodes):
    time_step = env.reset()
    state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
    episode_reward = 0

    while not time_step.last():
        state_tensor = torch.FloatTensor(state)
        action = policy_net(state_tensor).detach().numpy()
        time_step = env.step(action)
        next_state = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
        reward = time_step.reward
        episode_reward += reward

        # Calculate the loss and update the policy network
        action_tensor = torch.FloatTensor(action)
        action_tensor.requires_grad = True
        loss = -torch.log(action_tensor) * reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

    print(f"Episode {episode+1}: Reward = {episode_reward}")