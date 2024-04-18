#! python3

import argparse
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np # NOTE only imported because https://github.com/pytorch/pytorch/issues/13918
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class PolicyGradient(nn.Module):
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, mode='REINFORCE', n=100, gamma=0.99, device='cpu'):
        super(PolicyGradient, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.mode = mode
        self.n = n
        self.gamma = gamma

        self.device = device

        hidden_layer_size = 256

        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, action_size),
            # BEGIN STUDENT SOLUTION
            nn.Softmax(dim=-1)
            # END STUDENT SOLUTION
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_size, hidden_layer_size),
            nn.ReLU(),
            # BEGIN STUDENT SOLUTION
            nn.Linear(hidden_layer_size, action_size),
            nn.Softmax(dim=-1)
            # END STUDENT SOLUTION
        )

        # initialize networks, optimizers, move networks to device
        # BEGIN STUDENT SOLUTION
        self.actor.to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        if ('REINFORCE' != self.mode):
            self.critic.to(self.device)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        # END STUDENT SOLUTION


    def forward(self, state):
        if 'REINFORCE' == self.mode:
            return(self.actor(state))
        return(self.actor(state), self.critic(state))


    def get_action(self, state, stochastic):
        # if stochastic, sample using the action probabilities, else get the argmax
        # BEGIN STUDENT SOLUTION
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if ('REINFORCE' != self.mode):
            action_probs, _ = self.forward(state)
        else:
            action_probs = self.forward(state)

        if stochastic:
            distribution = torch.distributions.Categorical(action_probs)
            action = distribution.sample()
            return action.item()
        else:
            return torch.argmax(action_probs).item()
        # END STUDENT SOLUTION


    def calculate_n_step_bootstrap(self, rewards_tensor, values):
        # calculate n step bootstrap
        # BEGIN STUDENT SOLUTION
        T = len(rewards_tensor)
        N = self.n
        Gs = []
        for t in range(T):
            if t + N < T:
                V_end = values[t + N]
            else:
                V_end = 0
            G_t = 0
            upper_bound = min(t+N-1, T)
            for k in range(t, upper_bound):
                G_t += (self.gamma**(k-t)) * rewards_tensor[k]
            G_t += V_end * (self.gamma**(N))
            Gs.append(G_t)
        return torch.FloatTensor(Gs).to(self.device)
        # END STUDENT SOLUTION



    #Custom reward calc function forreinforce and reinforce with baseline
    def cust_reward_calc(self, rewards_tensor):
        sz = len(rewards_tensor)
        Gs = []
        for t in range(sz):
            G_t = 0
            for k in range(t, sz):
                G_t += (self.gamma**(k-t)) * rewards_tensor[k]
            Gs.append(G_t)
        return torch.FloatTensor(Gs).to(self.device)


    def train(self, states, actions, rewards):
        # train the agent using states, actions, and rewards
        # BEGIN STUDENT SOLUTION

        self.optimizer_actor.zero_grad()
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        if ('REINFORCE' == self.mode):
            Gs = self.cust_reward_calc(rewards)
        elif('REINFORCE_WITH_BASELINE' == self.mode):
            self.optimizer_critic.zero_grad()
            actor_probs, critic_values = self.forward(torch.FloatTensor(states).to(self.device))
            critic_values = critic_values.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device)).squeeze(1)
            Gs = self.cust_reward_calc(rewards)
        else:
            self.optimizer_critic.zero_grad()
            actor_probs, critic_values = self.forward(torch.FloatTensor(states).to(self.device))
            critic_values = critic_values.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device)).squeeze(1)
            Gs = self.calculate_n_step_bootstrap(rewards, critic_values)

        if ('REINFORCE' == self.mode):
            log_prob = self.forward(torch.FloatTensor(states).to(self.device))
            log_prob = log_prob.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device)).squeeze(1).log()
            loss = -1 * (log_prob * Gs).mean()
            loss.backward()
            self.optimizer_actor.step()
        else:
            actor_log_prob = actor_probs.gather(1, torch.LongTensor(actions).unsqueeze(1).to(self.device)).squeeze(1).log()
            diff = Gs - critic_values
            actor_loss = -1 * (diff * actor_log_prob).mean()
            actor_loss.backward(retain_graph=True)
            self.optimizer_actor.step()
            critic_loss = (diff**2).mean()
            critic_loss.backward()
            self.optimizer_critic.step()


        # END STUDENT SOLUTION



    def run(self, env, max_steps, num_episodes, train):
        total_rewards = []
        # run the agent through the environment num_episodes times for at most max steps
        # BEGIN STUDENT SOLUTION
        for episode in range(num_episodes):
            state = env.reset()[0]
            states = []
            actions = []
            rewards = []
            for step in range(max_steps):
                action = self.get_action(state, train)
                next_state, reward, done, info , _ = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                if done:
                    break
            if train:
                self.train(states, actions, rewards)
            else:
                total_rewards.append(sum(rewards))
        # END STUDENT SOLUTION
        return(total_rewards)



def graph_agents(graph_name, agents, env, max_steps, num_episodes):
    print(f'Starting: {graph_name}')

    # graph the data mentioned in the homework pdf
    # BEGIN STUDENT SOLUTION
    atr = []
    for agent in agents:
        average_total_rewards_agent = []
        iters = int(num_episodes/100)
        for i in tqdm(range(iters)):
            tr_train = agent.run(env, max_steps, 100, True)
            tr_test = agent.run(env, max_steps, 20, False)
            tr_test = sum(tr_test) / len(tr_test)
            average_total_rewards_agent.append(tr_test)
        atr.append(average_total_rewards_agent)

    atr = torch.tensor(atr, dtype=torch.float)
    min_values, _ = torch.min(atr, dim=0)
    min_total_rewards = min_values.view(-1)
    max_values, _ = torch.max(atr, dim=0)
    max_total_rewards = max_values.view(-1)

    average_total_rewards = torch.mean(atr, dim=0)
    graph_every = 100
    # END STUDENT SOLUTION

    # plot the total rewards
    xs = [i * graph_every for i in range(len(average_total_rewards))]
    fig, ax = plt.subplots()
    plt.fill_between(xs, min_total_rewards.tolist(), max_total_rewards.tolist(), alpha=0.1)
    ax.plot(xs, average_total_rewards)
    ax.set_ylim(-max_steps * 0.01, max_steps * 1.1)
    ax.set_title(graph_name, fontsize=10)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Total Reward')
    #fig.savefig(f'./graphs/{graph_name}.png')
    #plt.close(fig)
    plt.show()
    print(f'Finished: {graph_name}')



def parse_args():
    mode_choices = ['REINFORCE', 'REINFORCE_WITH_BASELINE', 'A2C']

    parser = argparse.ArgumentParser(description='Train an agent.')
    parser.add_argument('--mode', type=str, default='REINFORCE', choices=mode_choices, help='Mode to run the agent in')
    parser.add_argument('--n', type=int, default=64, help='The n to use for n step A2C')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs to average over for graph')
    parser.add_argument('--num_episodes', type=int, default=3500, help='Number of episodes to train for')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps in the environment')
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Environment name')
    return parser.parse_args()



def main():
    args = parse_args()

    # init args, agents, and call graph_agents on the initialized agents
    # BEGIN STUDENT SOLUTION
    #init agent
    # get states and action sizes
    env = gym.make(args.env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = PolicyGradient(state_size, action_size, mode=args.mode, n=args.n)
    agent.run(env, args.max_steps, args.num_episodes, True)
    # END STUDENT SOLUTION



if '__main__' == __name__:
    main()
