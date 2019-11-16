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

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

REPLAY_BUFFER_SIZE = 2e6
BATCH_SIZE = 32
EPOCH = 50
EPISODES = 50
STEPS = 100
UPDATE_STEPS = 100

class DoubleDQN:
    def __init__(self, env, tau=0.01, gamma=0.9, epsilon=1.0):
        self.env = env
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.embedding_size = 50
        self.hidden_size = 50
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
        states, actions, goals, rewards, next_states, satisfied_goals, dones = batch

        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        curr_Q = self.model.forward(states, goals) 

        curr_Q_prev_actions = [curr_Q[batch, actions[batch][0], actions[batch][1]] for batch in range(len(states))] # TODO: Use pytorch gather
        curr_Q_prev_actions = torch.stack(curr_Q_prev_actions)

        next_Q = self.target_model.forward(next_states, goals) 
        
        next_Q_max_actions = torch.max(next_Q, -1).values
        next_Q_max_actions = torch.max(next_Q_max_actions, -1).values

        next_Q_max_actions = rewards + (1 - dones) * self.gamma * next_Q_max_actions

        loss = F.mse_loss(curr_Q_prev_actions, next_Q_max_actions.detach())

        return loss

    def update(self, replay_buffer, batch_size):
        for _ in range(UPDATE_STEPS):
            batch = replay_buffer.sample(batch_size)
            loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target_net(self): # TODO: Check this function
         # target network update
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help="[test|train]")
args = parser.parse_args()

def train(env, agent):
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    for cycle in range(EPOCH):
        epoch_reward = 0.0
        agent.update_target_net() # target network update
        for episode in range(EPISODES):
            state = env.reset()
            goal, goal_program = env.sample_goal()
            env.set_goal(goal, goal_program)
            episode_reward = 0
            trajectory = []

            for step in range(STEPS):
                action = agent.get_action(state, goal)
                next_state, reward, done, _ = env.step(action, record_achieved_goal=True)
                achieved_goals = env.get_achieved_goals()
                transition = Transition(state, action, goal, reward, next_state, achieved_goals, done)
                trajectory.append(transition)
                episode_reward += reward

                if done:
                    break

                state = next_state
            
            for step in range(len(trajectory)):
                replay_buffer.add(trajectory[step])
                for goal_prime in trajectory[step].satisfied_goals_t:
                    transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, 1.0, trajectory[step].next_state, trajectory[step].satisfied_goals_t, trajectory[step].done)
                    replay_buffer.add(transition)
                deltas = relabel_future_instructions(trajectory, step, 4, 0.9)
                for delta in deltas:
                    goal_prime, reward_prime = delta
                    transition = Transition(trajectory[step].current_state, trajectory[step].action, goal_prime, reward_prime, trajectory[step].next_state, trajectory[step].satisfied_goals_t, trajectory[step].done)
                    replay_buffer.add(transition)    

            epoch_reward += episode_reward

            logging.error("[Episode] " + str(episode) + ": reward " + str(episode_reward))

            agent.update(replay_buffer, BATCH_SIZE)   

        logging.error("[Epoch] " + str(cycle) + ": average reward " + str(epoch_reward/STEPS))

        if cycle > 9:
            agent.epsilon *= 0.993
            if agent.epsilon < 0.1:
                agent.epsilon = 0.1

        torch.save(agent, 'agent.npy')

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