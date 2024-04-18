#! python3

import argparse
import collections
import random

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class ReplayMemory():
    def __init__(self, memory_size, batch_size):
        # define init params
        # use collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory = collections.deque(maxlen=memory_size)
        self.batch_size = batch_size
        # END STUDENT SOLUTION
        pass


    def sample_batch(self):
        # randomly chooses from the collections.deque
        # BEGIN STUDENT SOLUTION
        return random.sample(self.memory, self.batch_size)
        # END STUDENT SOLUTION
        pass


    def append(self, transition):
        # append to the collections.deque
        # BEGIN STUDENT SOLUTION
        self.memory.append(transition)
        # END STUDENT SOLUTION
        pass



class DeepQNetwork(nn.Module):
    def __init__(self, state_size, action_size, lr_q_net=2e-4, gamma=0.99, epsilon=0.05, target_update=50, burn_in=10000, replay_buffer_size=50000, replay_buffer_batch_size=32, device='cpu'):
        super(DeepQNetwork, self).__init__()

        # define init params
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma
        self.epsilon = epsilon

        self.target_update = target_update

        self.burn_in = burn_in

        self.device = device

        hidden_layer_size = 256

        # q network
        q_net_init = lambda: nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size)
            # END STUDENT SOLUTION
        )

        # initialize replay buffer, networks, optimizer, move networks to device
        # BEGIN STUDENT SOLUTION
        self.q_net = q_net_init().to(self.device)
        self.target_q_net = q_net_init().to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr_q_net)
        self.replay_memory = ReplayMemory(replay_buffer_size, replay_buffer_batch_size)
        # END STUDENT SOLUTION


    def forward(self, state):
        return(self.q_net(state), self.target(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using epsilon greedy, else get the argmax
        # BEGIN STUDENT SOLUTION
        if stochastic and random.random() < self.epsilon:
            action = random.choice(range(self.action_size))
            return action
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                return self.q_net(state).argmax().item()

        # END STUDENT SOLUTION
        pass


    def train(self):
        # train the agent using the replay buffer
        # BEGIN STUDENT SOLUTION

        if len(self.replay_memory.memory) < self.replay_memory.batch_size:
            return
        
        transitions = self.replay_memory.sample_batch()
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        current_q_values = self.q_net(states).gather(1, actions)
        next_q_values = self.target_q_net(next_states).detach().max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # END STUDENT SOLUTION
        pass


    def run(self, env, max_steps, num_episodes, train, init_buffer):
        total_rewards = []

        # initialize replay buffer
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            state = env.reset()[0]
            episode_reward = 0
            for step in range(max_steps):
                action = self.get_action(state, train)
                next_state, reward, done, info, _ = env.step(action)
                self.replay_memory.append((state, action, reward, next_state, done))
                episode_reward += reward
                state = next_state
                if train:
                    self.train()
                if step % self.target_update == 0:
                    self.target_q_net.load_state_dict(self.q_net.state_dict())
                if done:
                    break
            total_rewards.append(episode_reward)
            print(f'Episode {episode} reward: {episode_reward}')
        # END STUDENT SOLUTION
        return(total_rewards)




def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    atr = []
    graph_every = 20
    total_rewards_agents = []
    for agent in agents:
        total_rewards = agent.run(env, max_steps, num_episodes, True, True)
        total_rewards_agents.append(total_rewards)

    for total_rewards in total_rewards_agents:
        average_total_rewards = []
        for i in range(0, len(total_rewards), graph_every):
            average_total_rewards.append(sum(total_rewards[i:i+20])/graph_every)
        atr.append(average_total_rewards)

    atr = torch.tensor(atr, dtype=torch.float)
    min_values, _ = torch.min(atr, dim=0)
    min_total_rewards = min_values.view(-1)
    max_values, _ = torch.max(atr, dim=0)
    max_total_rewards = max_values.view(-1)
    average_total_rewards = torch.mean(atr, dim=0)
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards, max_total_rewards, alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    fig.savefig(f'./graphs/{graph_name}.png')
    plt.close(fig)
    print(f'Finished: {graph_name}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agent on the initialized agents
    # BEGIN STUDENT SOLUTION
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agents = []
    for i in range(5):
        agents.append(DeepQNetwork(state_size, action_size))
    graph_agents('DQN', agents, env, args.max_steps, args.num_episodes)

    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
