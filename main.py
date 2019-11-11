## Based on https://towardsdatascience.com/double-deep-q-networks-905dd8325412
## Please read to follow the code below, Double Q-learning in Hasselt et al., 2015 is used

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import argparse

import sys, os.path
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','clevr_robot_env'))
from clevr_robot_env import ClevrEnv
from networks import DQN, Encoder
from replay_buffer import ReplayBuffer

MAX_EPISODES = 50
REPLAY_BUFFER_SIZE = 100 

class DoubleDQN:
    def __init__(self, env, replay_buffer, tau=0.01, gamma=0.99, epsilon=0.9):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.embedding_size = 64
        self.hidden_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = self.env.get_obs().shape
        self.action_shape = 40
        self.model = DQN(self.obs_shape, self.action_shape).to(self.device)
        self.target_model = DQN(self.obs_shape, self.action_shape).to(self.device)
        self.encoder = Encoder(self.embedding_size, self.hidden_size)
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

def train(env, agent):
    episode_rewards = []
    for episode in range(MAX_EPISODES):
        state = env.reset()
        goal, goal_program = env.sample_goal()
        env.set_goal(goal, goal_program)
        episode_reward = 0
        max_steps = len(agent.replay_buffer)

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": reward " + str(episode_reward))
                break

            state = next_state
    return episode_rewards

def test(env, agent):
    pass

def main():
    env = ClevrEnv(action_type="perfect", obs_type='order_invariant', direct_obs=True)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    agent = DoubleDQN(env, replay_buffer)

    if args.mode == "train":
        train(env, agent)
    else:
        test(env, agent)

if __name__ == "__main__":
    main()