## Based on https://towardsdatascience.com/double-deep-q-networks-905dd8325412
## Please read to follow the code below, Double Q-learning in Hasselt et al., 2015 is used

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np

import argparse

import sys, os.path
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..','clevr_robot_env'))
from clevr_robot_env import ClevrEnv
from networks import DQN, Encoder
from replay_buffer import ReplayBuffer
from transition import Transition
from util import relabel_future_instructions

MAX_EPISODES = 50
REPLAY_BUFFER_SIZE = 100
BATCH_SIZE = 30

class DoubleDQN:
    def __init__(self, env, tau=0.01, gamma=0.99, epsilon=0.9):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.embedding_size = 64
        self.hidden_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_shape = self.env.get_obs().shape
        self.action_shape = 40 // 5
        self.model = DQN(self.obs_shape, self.action_shape).to(self.device)
        self.target_model = DQN(self.obs_shape, self.action_shape).to(self.device)
        self.encoder = Encoder(self.embedding_size, self.hidden_size)

        # hard copy model parameters to target model parameters
        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_action(self, state, goal):
        assert len(state.shape) == 2 # This function should not be called during update

        if(np.random.randn() < self.epsilon):
            q_values = self.model.forward(state, goal)
            idx = torch.argmax(q_values).detach().numpy()
        else:
            idx = self.env.action_space.sample()

        obj_selection = idx // 8
        direction_selection = idx % 8

        return int(obj_selection), int(direction_selection)
        
    
    def compute_loss(self, batch):     
        states, actions, rewards, next_states, _, dones = batch
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

    def update(self, replay_buffer, batch_size):
        batch = replay_buffer.sample(batch_size)
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
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    for episode in range(MAX_EPISODES):
        state = env.reset()
        goal, goal_program = env.sample_goal()
        print("running episode: ", episode)
        print("episode goal: ", goal)
        env.set_goal(goal, goal_program)
        episode_reward = 0
        max_steps = replay_buffer.max_size()
        trajectory = []

        for step in range(max_steps):
            action = agent.get_action(state, goal)
            print("choosing action: ", action)
            next_state, reward, done, _ = env.step(action, record_achieved_goal=True)
            achieved_goals = env.get_achieved_goals()
            print("num achieved goals: ", len(achieved_goals))
            transition = Transition(state, action, goal, reward, next_state, achieved_goals, done)
            trajectory.append(transition)
            episode_reward += reward

            # if len(agent.replay_buffer) > BATCH_SIZE:
            #     agent.update(BATCH_SIZE)   

            if done or step == max_steps-1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": reward " + str(episode_reward))
                break

            state = next_state

        for step in range(len(trajectory)): # T == length of trajectory?
            replay_buffer.add(trajectory[step])
            print("just finished adding all transitions")
            for goal_prime in trajectory[step].satisfied_goals_t:
                transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, 1, trajectory[step].next_state, trajectory[step].satisfied_goals_t, trajectory[step].done)
                replay_buffer.add(transition)
            print("just finished adding all achieved goals")
            deltas = relabel_future_instructions(trajectory, step, 4, 30)
            for delta in deltas:
                goal_prime, reward_prime = delta
                transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, reward_prime, trajectory[step].next_state, trajectory[step].satisfied_goals_t, trajectory[step].done)
                replay_buffer.add(transition)    
            print("just finished adding all deltas")

        agent.update(replay_buffer, BATCH_SIZE)   
        print("completed episode: ", episode)
    return episode_rewards

def test(env, agent):
    pass

def main():
    env = ClevrEnv(action_type="perfect", obs_type='order_invariant', direct_obs=True)
    # replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    agent = DoubleDQN(env)

    if args.mode == "train":
        train(env, agent)
    else:
        test(env, agent)

if __name__ == "__main__":
    main()