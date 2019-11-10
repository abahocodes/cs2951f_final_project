## Based on https://towardsdatascience.com/double-deep-q-networks-905dd8325412
## Please read to follow the code below, Double Q-learning in Hasselt et al., 2015 is used

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import argparse

from clevr_robot_env import ClevrEnv
import sys
import sys.path.append("../clevr_env_robot")
from networks import DQN
from replay_buffer import ReplayBuffer

MAX_EPISODES = 5000

class DoubleDQN:
    def __init__(self, env, replay_buffer, tau=0.01, gamma=0.99, epsilon=0.9):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(observation_space_shape, action_space_shape).to(self.device)
        self.target_model = DQN(observation_space_shape, action_space_shape).to(self.device)
        self.replay_buffer = replay_buffer

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state):
        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn() < self.epsilon):
            return self.env.action_space.sample()
        return action
    
    def compute_loss(self, batch):     
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones)

        # resize tensors
        actions = actions.view(actions.size(0))
        dones = dones.view(dones.size(0))

        # compute loss
        curr_Q = self.model.forward(states).gather(1, actions.view(actions.size(0), 1))
        next_Q = self.target_model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.gamma * max_next_Q

        loss = F.mse_loss(curr_Q, expected_Q.detach())

        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="[test|train]")
args = parser.parse_args()

env = ClevrEnv()

def train():
    # for episode in range(MAX_EPISODES):
    #     print("training episode: ", episode)
    pass



def main():
    if args.mode == "train":
        train()
    else:
        test()

if __name__ == "__main__":
    main()